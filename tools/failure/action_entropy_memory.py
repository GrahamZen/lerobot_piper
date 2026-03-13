from collections import deque
from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_IMAGES


def compute_action_entropy_gpu(
    action_samples: torch.Tensor,
    bandwidth: float = 1.0,
    min_density: float = 1e-8,
) -> torch.Tensor:
    """Estimate action entropy via isotropic Gaussian KDE.

    Matches the KDE.kde_entropy / gaussian_kernel implementation in
    DemoSpeedup's robobase/robobase/utils.py:
      - isotropic bandwidth = 1 (hardcoded scalar, same for all dims)
      - squared Euclidean distance summed over all action dims
      - kernel: exp(-dist / (2 * bandwidth^2))
      - density = sum_over_neighbors / M
      - entropy = -mean(log(density + 1e-8))

    Args:
        action_samples: Tensor of shape (M, D), where M is the number of overlapping
            predictions for the same target timestep.

    Returns:
        Scalar entropy estimate.
    """
    if action_samples.ndim != 2:
        raise ValueError(f"Expected action_samples to have shape (M, D), got {tuple(action_samples.shape)}")

    num_samples = action_samples.shape[0]
    if num_samples == 0:
        raise ValueError("Expected at least one action sample to estimate entropy")

    x_i = action_samples.unsqueeze(1)  # (M, 1, D)
    x_j = action_samples.unsqueeze(0)  # (1, M, D)
    distances = torch.sum((x_i - x_j) ** 2, dim=-1)  # (M, M) squared Euclidean
    kernel_values = torch.exp(-distances / (2 * bandwidth**2))  # (M, M)

    density = kernel_values.sum(dim=1) / num_samples  # (M,)
    return -(torch.log(density + min_density)).mean()


def _collect_overlapping_predictions(
    recent_predictions: deque[tuple[int, torch.Tensor]],
    target_step: int,
    chunk_size: int,
) -> torch.Tensor:
    """Collect all action samples for *target_step* from overlapping source predictions.

    Each entry in *recent_predictions* stores (chunk_size, D).

    Returns tensor of shape (K, D).
    """
    overlapping_predictions = []
    for source_step, source_samples in recent_predictions:
        horizon = target_step - source_step
        if 0 <= horizon < chunk_size:
            # (chunk_size, D) -> (1, D)
            overlapping_predictions.append(source_samples[horizon : horizon + 1, :])

    if not overlapping_predictions:
        raise RuntimeError(f"No overlapping predictions found for target step {target_step}")

    return torch.cat(overlapping_predictions, dim=0)  # (K, D)


def _collate_preprocessed_batch(preprocessed_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate list of preprocessed single-item batches (each with batch dim=1)."""
    batch: dict[str, Any] = {}
    keys = preprocessed_items[0].keys()
    for key in keys:
        values = [item[key] for item in preprocessed_items]
        if isinstance(values[0], torch.Tensor):
            # Some metadata entries are scalar tensors; stack them into a batch axis.
            if values[0].ndim == 0:
                batch[key] = torch.stack(values, dim=0)
            else:
                batch[key] = torch.cat(values, dim=0)
        else:
            batch[key] = values[0]
    return batch


class _IndexedSubset(Dataset):
    """Wrap Subset to keep original dataset index for downstream bookkeeping."""

    def __init__(self, dataset, indices: Sequence[int]):
        self.subset = Subset(dataset, indices)
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return {
            "step_index": int(self.indices[idx]),
            "item": self.subset[idx],
        }


@torch.no_grad()
def precompute_action_entropy_in_memory(
    dataset,
    policy: ACTPolicy,
    preprocessor,
    indices: Sequence[int],
    batch_size: int = 16,
) -> tuple[dict[int, float], dict[int, float]]:
    """Precompute entropy/max_diff for dataset indices.

    Uses ACT inference path directly (latent z=0) and computes KDE entropy over
    overlapping predictions for the same target timestep.
    """
    if not indices:
        return {}, {}

    ordered_indices = sorted({int(idx) for idx in indices})

    policy.eval()
    batch_size = max(1, int(batch_size))
    chunk_size = int(policy.config.chunk_size)

    # Build episode-start guard so contiguous global indices do not leak overlap
    # predictions across episode boundaries.
    episode_start_steps: set[int] = set()
    if getattr(dataset.meta, "episodes", None) is None:
        from lerobot.datasets.utils import load_episodes

        dataset.meta.episodes = load_episodes(dataset.root)
    for ep_meta in dataset.meta.episodes:
        from_idx = int(
            ep_meta["dataset_from_index"]
            if not isinstance(ep_meta["dataset_from_index"], list)
            else ep_meta["dataset_from_index"][0]
        )
        episode_start_steps.add(from_idx)
    first_stream_step = ordered_indices[0]

    device = next(policy.parameters()).device
    entropy_by_step: dict[int, float] = {}
    max_diff_by_step: dict[int, float] = {}

    fast_compute_entropy = compute_action_entropy_gpu
    compiled_entropy_enabled = hasattr(torch, "compile")
    if compiled_entropy_enabled:
        fast_compute_entropy = torch.compile(compute_action_entropy_gpu, mode="reduce-overhead")

    current_batch_size = batch_size
    progress = tqdm(total=len(ordered_indices), desc="Precompute action entropy")

    while True:
        indexed_subset = _IndexedSubset(dataset, ordered_indices)

        def _worker_collate(batch_items):
            step_indices = [int(row["step_index"]) for row in batch_items]
            raw_items = [row["item"] for row in batch_items]
            return raw_items, step_indices

        loader = DataLoader(
            indexed_subset,
            batch_size=current_batch_size,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_worker_collate,
            shuffle=False,
        )

        try:
            recent_predictions: deque[tuple[int, torch.Tensor]] = deque()
            for raw_items, step_indices in loader:
                preprocessed = [preprocessor(item) for item in raw_items]
                batch = _collate_preprocessed_batch(preprocessed)

                batch_n: dict[str, Any] = {}
                for key, val in batch.items():
                    if isinstance(val, torch.Tensor):
                        batch_n[key] = val.to(device, non_blocking=True)
                    else:
                        batch_n[key] = val

                if policy.config.image_features:
                    batch_n[OBS_IMAGES] = [batch_n[key] for key in policy.config.image_features]

                # ACT inference uses latent_sample=zeros internally.
                all_actions, _ = policy.model(batch_n)  # (B, chunk_size, action_dim)

                for offset, step in enumerate(step_indices):
                    step = int(step)
                    if step in episode_start_steps and step != first_stream_step:
                        recent_predictions.clear()
                    if recent_predictions and step != recent_predictions[-1][0] + 1:
                        recent_predictions.clear()
                    # Store (chunk_size, D) — one prediction chunk per source step.
                    recent_predictions.append((step, all_actions[offset]))
                    while recent_predictions and step - recent_predictions[0][0] >= chunk_size:
                        recent_predictions.popleft()

                    overlapping_predictions = _collect_overlapping_predictions(
                        recent_predictions=recent_predictions,
                        target_step=step,
                        chunk_size=chunk_size,
                    )
                    if overlapping_predictions.shape[0] < 3:
                        entropy_by_step[step] = float("nan")
                        max_diff_by_step[step] = float("nan")
                        progress.update(1)
                        continue

                    max_diff = torch.abs(overlapping_predictions - overlapping_predictions[0:1]).max()
                    try:
                        entropy_value = fast_compute_entropy(overlapping_predictions)
                    except RuntimeError as exc:
                        compile_runtime_failed = any(
                            token in str(exc).lower()
                            for token in ("inductor", "triton", "torch.compile", "torch._dynamo")
                        )
                        if compiled_entropy_enabled and compile_runtime_failed:
                            print(
                                "[WARN] Compiled entropy kernel failed at runtime, "
                                "falling back to eager implementation."
                            )
                            fast_compute_entropy = compute_action_entropy_gpu
                            compiled_entropy_enabled = False
                            entropy_value = fast_compute_entropy(overlapping_predictions)
                        else:
                            raise

                    entropy_by_step[step] = float(entropy_value.detach().cpu().item())
                    max_diff_by_step[step] = float(max_diff.detach().cpu().item())
                    progress.update(1)
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or current_batch_size <= 1:
                progress.close()
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            current_batch_size = max(1, current_batch_size // 2)
            entropy_by_step.clear()
            max_diff_by_step.clear()
            progress.n = 0
            progress.refresh()
            print(f"[WARN] OOM during entropy precompute, restarting with batch_size={current_batch_size}.")

    progress.close()

    return entropy_by_step, max_diff_by_step
