import math
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
    min_bandwidth: float = 1e-5,
    min_density: float = 1e-35,
) -> torch.Tensor:
    """Estimate paper-style conditional action entropy from overlapping predictions.

    Args:
        action_samples: Tensor of shape (M, D), where M is the number of overlapping
            predictions for the same target timestep collected from the past K chunks.

    Returns:
        Scalar entropy estimate matching the paper's KDE + -sum(p log p) procedure.
    """
    if action_samples.ndim != 2:
        raise ValueError(f"Expected action_samples to have shape (M, D), got {tuple(action_samples.shape)}")

    sample_count, action_dim = action_samples.shape
    if sample_count == 0:
        raise ValueError("Expected at least one action sample to estimate entropy")

    dtype = action_samples.dtype
    device = action_samples.device
    min_bw_t = torch.tensor(min_bandwidth, dtype=dtype, device=device)
    min_density_t = torch.tensor(min_density, dtype=dtype, device=device)
    gaussian_norm_t = torch.tensor(math.sqrt(2.0 * math.pi), dtype=dtype, device=device)

    sigma = torch.std(action_samples, dim=0, unbiased=sample_count > 1)
    sigma = torch.clamp(sigma, min=min_bw_t)
    bandwidth = torch.clamp(1.06 * sigma * (sample_count ** (-0.2)), min=min_bw_t)

    squared_mahalanobis = torch.zeros(
        (sample_count, sample_count),
        dtype=dtype,
        device=device,
    )
    for dim_idx in range(action_dim):
        vals = action_samples[:, dim_idx]
        diffs = vals.unsqueeze(1) - vals.unsqueeze(0)
        squared_mahalanobis = squared_mahalanobis + (diffs / bandwidth[dim_idx]) ** 2

    log_kernel_norm = torch.log(bandwidth * gaussian_norm_t).sum()
    kernel_vals = torch.exp(-0.5 * squared_mahalanobis - log_kernel_norm)
    densities = torch.clamp(kernel_vals.mean(dim=1), min=min_density_t)
    return -torch.log(densities).mean()


def _collect_overlapping_predictions(
    recent_predictions: deque[tuple[int, torch.Tensor]],
    target_step: int,
    chunk_size: int,
) -> torch.Tensor:
    overlapping_predictions = []
    for source_step, source_actions in recent_predictions:
        horizon = target_step - source_step
        if 0 <= horizon < chunk_size:
            overlapping_predictions.append(source_actions[horizon : horizon + 1, :])

    if not overlapping_predictions:
        raise RuntimeError(f"No overlapping predictions found for target step {target_step}")

    return torch.cat(overlapping_predictions, dim=0)


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
    """Precompute entropy/max_diff for dataset indices with single-sample forward pass."""
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
                effective_batch_size = len(step_indices)

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

                actions, _ = policy.model(batch_n)
                _, _, action_dim = actions.shape
                actions = actions.view(effective_batch_size, chunk_size, action_dim)

                for offset, step in enumerate(step_indices):
                    step = int(step)
                    if step in episode_start_steps and step != first_stream_step:
                        recent_predictions.clear()
                    if recent_predictions and step != recent_predictions[-1][0] + 1:
                        recent_predictions.clear()
                    recent_predictions.append((step, actions[offset]))
                    while recent_predictions and step - recent_predictions[0][0] >= chunk_size:
                        recent_predictions.popleft()

                    overlapping_predictions = _collect_overlapping_predictions(
                        recent_predictions=recent_predictions,
                        target_step=step,
                        chunk_size=chunk_size,
                    )
                    if overlapping_predictions.shape[0] < 3:
                        entropy_by_step[step] = float("nan")  # 替换掉 0.0
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
