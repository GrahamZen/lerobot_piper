"""Visualise a LeRobot dataset with failure-handling metrics in Rerun.

All metrics are read directly from failure_metrics.jsonl — no model loading,
no forward passes, no recomputation of any kind.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))

from offline_utils import _extract_pretrained_path, get_episode_bounds, load_failure_metrics_jsonl

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.failure_handling.config import FailureConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class PersistentTIDEDetector:
    def __init__(self, calibration_data, decay_lambda=0.95, k=1.0, threshold_c=5.0):
        self.calibration_data = calibration_data
        self.decay_lambda = decay_lambda
        self.k = k
        self.threshold_c = threshold_c
        self.c_t = 0.0

    def update(self, raw_tide, current_action, prev_action):
        act_diff = np.linalg.norm(current_action - prev_action)
        current_regime = "1" if act_diff > 0.1 else "0"

        calib = self.calibration_data.get(current_regime, {"mean": 0.0, "std": 1.0})
        mu = calib["mean"]
        sigma = calib["std"]

        # 1. Phase-aware Uncertainty Normalization (条件归一化)
        n_tide = max(0.0, (raw_tide - mu) / sigma)

        # 2. Temporal Persistence Statistic (CUSUM 累积)
        self.c_t = max(0.0, self.decay_lambda * self.c_t + n_tide - self.k)

        # 3. 最终判决
        is_failure = self.c_t > self.threshold_c

        return is_failure, self.c_t, n_tide, current_regime

    def reset(self):
        self.c_t = 0.0


# All jsonl keys that may contain per-step similarity values.
# List values → per-slot curves.  Scalar values → single curve (slot 0).
_RECORDED_SIM_KEYS = [
    "resnet_feat_similarity",  # ResNetCheckpointStrategy  — list, one per slot
    "act_fused_similarity",  # ACTFusedCheckpointStrategy — list, one per slot
    "encoder_out_similarity",  # CheckpointStrategy feature_type=encoder_out
    "backbone_similarity",  # CheckpointStrategy feature_type=backbone
    "resnet_flat_sim",  # ResNetFlatCheckpointStrategy — full (N,) vector
    "act_fused_flat_sim",  # ACTFusedFlatCheckpointStrategy — full (N,) vector
    "encoder_out_flat_sim",  # FlatCheckpointStrategy feature_type=encoder_out
    "backbone_flat_sim",  # FlatCheckpointStrategy feature_type=backbone
    "resnet_flat_sim_max",  # fallback for old recordings (scalar only)
    "act_fused_flat_sim_max",  # fallback for old recordings (scalar only)
]


def _resolve_pretrained_path(dataset_root: Path) -> Path | None:
    record_config_path = dataset_root / "meta" / "record_config.json"
    if not record_config_path.exists():
        return None
    try:
        with record_config_path.open("r", encoding="utf-8") as f:
            record_config = json.load(f)
    except Exception:
        return None
    return _extract_pretrained_path(record_config)


def _log_stitched_cameras(item: dict, camera_keys: list[str]) -> None:
    """Stitch all cameras left-to-right and log as a single image at cameras/stitched."""
    cam_imgs: list[np.ndarray] = []
    for cam_key in camera_keys:
        img = item.get(cam_key)
        if img is None:
            continue
        if isinstance(img, dict) and "bytes" in img:
            continue  # encoded images cannot be stitched
        arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
        if arr.ndim == 3 and arr.shape[0] <= 4:  # (C, H, W) → (H, W, C)
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        cam_imgs.append(arr)

    if not cam_imgs:
        return

    target_h = cam_imgs[0].shape[0]
    rows: list[np.ndarray] = []
    for img in cam_imgs:
        if img.shape[0] != target_h:
            from PIL import Image as PILImage

            w = int(img.shape[1] * target_h / img.shape[0])
            img = np.array(PILImage.fromarray(img).resize((w, target_h), PILImage.BILINEAR))
        rows.append(img)
    rr.log("cameras/stitched", rr.Image(np.concatenate(rows, axis=1)))


def _log_recorded_similarity(row: dict, rec_peaks: dict) -> None:
    """Log recorded similarity values from a jsonl row; mark new per-episode peaks."""
    for key in _RECORDED_SIM_KEYS:
        val = row.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            for idx, v in enumerate(val):
                v = float(v)
                rr.log(f"metrics/similarity/slot_{idx}", rr.Scalars(v))
                if v > rec_peaks.get(idx, float("-inf")):
                    rec_peaks[idx] = v
                    rr.log(f"metrics/similarity/peak_{idx}", rr.Scalars(v))
        else:
            v = float(val)
            rr.log("metrics/similarity/slot_0", rr.Scalars(v))
            if v > rec_peaks.get(0, float("-inf")):
                rec_peaks[0] = v
                rr.log("metrics/similarity/peak_0", rr.Scalars(v))
        break  # use only the first matching key found


def _stitch_checkpoint_image(
    dataset: "LeRobotDataset", camera_keys: list[str], frame_idx: int
) -> "np.ndarray | None":
    """Load camera images for *frame_idx* and stitch them horizontally into one array."""
    try:
        item = dataset[frame_idx]
    except Exception:
        return None

    cam_imgs: list[np.ndarray] = []
    for cam_key in camera_keys:
        img_tensor = item.get(cam_key)
        if img_tensor is None:
            continue
        if img_tensor.ndim == 4:
            img_tensor = img_tensor[0]
        if img_tensor.is_floating_point():
            img_np = (img_tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        else:
            img_np = img_tensor.permute(1, 2, 0).numpy()
        cam_imgs.append(img_np)

    if not cam_imgs:
        return None

    target_h = cam_imgs[0].shape[0]
    rows: list[np.ndarray] = []
    for img in cam_imgs:
        if img.shape[0] != target_h:
            from PIL import Image as PILImage

            w = int(img.shape[1] * target_h / img.shape[0])
            img = np.array(PILImage.fromarray(img).resize((w, target_h), PILImage.BILINEAR))
        rows.append(img)
    return np.concatenate(rows, axis=1)


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------


def visualize_dataset(
    repo_id: str, root: str | None = None, stride: int = 7, num_episode: int | None = None
) -> None:
    try:
        dataset = LeRobotDataset(repo_id, root=root)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    if dataset.meta.episodes is None:
        from lerobot.datasets.utils import load_episodes

        dataset.meta.episodes = load_episodes(dataset.root)

    total_episodes = len(dataset.meta.episodes)
    if num_episode is not None and int(num_episode) <= 0:
        raise ValueError("num_episode must be a positive integer.")
    episodes_to_visualize = total_episodes if num_episode is None else min(int(num_episode), total_episodes)
    print(f"Visualizing {episodes_to_visualize}/{total_episodes} episode(s).")

    dataset_root = Path(dataset.root)
    model_path = _resolve_pretrained_path(dataset_root)
    if not model_path:
        raise ValueError(f"Could not resolve pretrained model path for {dataset_root}")

    # --- Config (for failure_threshold only — no model loaded) ---
    failure_handling_json_path = model_path / "failure_handling.json"
    config = FailureConfig.from_json(failure_handling_json_path)
    print(f"[INFO] failure_threshold={config.detector.failure_threshold}")

    with failure_handling_json_path.open("r", encoding="utf-8") as f:
        raw_fh_config = json.load(f)
    detector_cfg = raw_fh_config.get("detector", {})
    calibration_data = detector_cfg.get("calibration_data")
    cusum_threshold = detector_cfg.get("cusum_threshold", 5.0)
    if not calibration_data:
        print("[WARN] calibration_data not found in failure_handling.json, using fallback stats.")
        calibration_data = {"0": {"mean": 0.0, "std": 1.0}, "1": {"mean": 0.0, "std": 1.0}}

    # --- Recorded metrics ---
    recorded_metrics = load_failure_metrics_jsonl(dataset_root)
    print(f"[INFO] Loaded {len(recorded_metrics)} recorded metric rows.")
    if recorded_metrics:
        sample = next(iter(recorded_metrics.items()))
        print(f"[INFO] Sample row (key={sample[0]}): {list(sample[1].keys())}")

    tide_detector = PersistentTIDEDetector(
        calibration_data=calibration_data,
        decay_lambda=0.95,
        k=1.0,
        threshold_c=cusum_threshold,
    )

    # --- Rerun blueprint ---
    camera_keys = dataset.meta.camera_keys
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Tabs(
                rrb.TimeSeriesView(
                    name="TD Smoothed",
                    contents=["metrics/td_smoothed/**", "metrics/failure_threshold"],
                ),
                rrb.TimeSeriesView(
                    name="TD Raw",
                    contents=["metrics/td_raw", "metrics/failure_threshold"],
                ),
                rrb.TimeSeriesView(
                    name="TD All",
                    contents=["metrics/td_raw", "metrics/td_smoothed/**", "metrics/failure_threshold"],
                ),
            ),
            rrb.TimeSeriesView(
                name="TIDE Detector",
                contents=["metrics/detector/**"],
            ),
            rrb.TimeSeriesView(
                name="Similarity",
                contents=["metrics/similarity/**"],
            ),
            rrb.TimeSeriesView(
                name="Checkpoint Selection",
                contents=["metrics/checkpoint/**"],
            ),
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.TextDocumentView(name="Episode ID", origin="overlay/episode_id"),
                    rrb.Spatial2DView(name="Cameras", origin="cameras/stitched"),
                    row_shares=[1, 5],
                ),
                rrb.Spatial2DView(name="Checkpoint Frame", origin="checkpoint_view/stitched"),
                column_shares=[1, 1],
            ),
            row_shares=[1, 1, 1, 1, 1.5],
        ),
        collapse_panels=True,
    )

    rr.init("LeRobot Dataset Visualiser", spawn=True)
    rr.send_blueprint(blueprint)
    rr.log(
        "metrics/td_smoothed/failed_markers",
        rr.SeriesPoints(colors=[255, 0, 0], markers="diamond", marker_sizes=5.0),
        static=True,
    )
    rr.log(
        "metrics/failure_threshold",
        rr.SeriesLines(colors=[255, 165, 0]),
        static=True,
    )
    rr.log(
        "metrics/detector/failed_markers",
        rr.SeriesPoints(colors=[255, 0, 0], markers="diamond", marker_sizes=5.0),
        static=True,
    )
    rr.log(
        "metrics/detector/threshold_C",
        rr.SeriesLines(colors=[255, 165, 0]),
        static=True,
    )

    # --- Streaming loop ---
    # Collect all (episode_idx, frame_idx) pairs upfront so a single DataLoader
    # handles every episode — avoids worker-process lifecycle issues that arise
    # when creating a new DataLoader per episode in a tight loop.
    all_frames: list[tuple[int, int]] = []  # (episode_idx, frame_idx)
    episode_from_idx: dict[int, int] = {}  # episode_idx -> global dataset start frame
    for episode_idx in range(episodes_to_visualize):
        from_idx, to_idx = get_episode_bounds(dataset, episode_idx)
        episode_from_idx[episode_idx] = from_idx
        for frame_idx in range(from_idx, to_idx, stride):
            all_frames.append((episode_idx, frame_idx))

    all_frame_indices = [f for _, f in all_frames]
    loader = DataLoader(
        Subset(dataset, all_frame_indices),
        batch_size=1,
        num_workers=4,
        prefetch_factor=2,
        collate_fn=lambda b: b[0],
        shuffle=False,
    )

    rec_peaks: dict[int, float] = {}  # slot_idx -> current-episode max
    prev_checkpoint_ts: int | None = None
    current_episode_idx = -1
    prev_act = None

    for global_step, ((episode_idx, frame_idx), item) in enumerate(zip(all_frames, loader, strict=True)):
        if episode_idx != current_episode_idx:
            # New episode — reset per-episode state
            current_episode_idx = episode_idx
            rec_peaks = {}
            prev_checkpoint_ts = None
            tide_detector.reset()
            prev_act = None
            print(f"Streaming Episode {episode_idx}/{episodes_to_visualize}...", end="\r")

        rr.set_time("step", sequence=frame_idx)
        rr.set_time("global_step", sequence=global_step)
        rr.log("overlay/episode_id", rr.TextDocument(f"{episode_idx}"), static=False)

        _log_stitched_cameras(item, camera_keys)

        row = recorded_metrics.get(frame_idx)
        if row:
            # TD metrics — read directly, no recomputation
            td_raw = float(row.get("td_raw", row.get("temporal_disagreement", 0.0)))
            td_smoothed = float(row.get("td_smoothed", 0.0))
            rr.log("metrics/td_raw", rr.Scalars(td_raw))
            rr.log("metrics/td_smoothed", rr.Scalars(td_smoothed))
            rr.log("metrics/failure_threshold", rr.Scalars(config.detector.failure_threshold))
            if td_smoothed > config.detector.failure_threshold:
                rr.log("metrics/td_smoothed/failed_markers", rr.Scalars(td_smoothed))

            # Run TIDE Detector
            curr_act = item["action"].numpy()
            if prev_act is None:
                prev_act = curr_act

            is_failure, c_t, n_tide, regime = tide_detector.update(td_raw, curr_act, prev_act)
            rr.log("metrics/detector/C_t", rr.Scalars(c_t))
            rr.log("metrics/detector/nTIDE", rr.Scalars(n_tide))
            rr.log("metrics/detector/threshold_C", rr.Scalars(tide_detector.threshold_c))
            rr.log("metrics/detector/regime", rr.Scalars(float(regime)))
            if is_failure:
                rr.log("metrics/detector/failed_markers", rr.Scalars(c_t))

            prev_act = curr_act

            # Similarity — read directly
            _log_recorded_similarity(row, rec_peaks)

            # Checkpoint timestep — best_slot_timestep is episode-local; convert to global
            best_local = int(row.get("best_slot_timestep", -1))
            if best_local >= 0:
                best_global = episode_from_idx[episode_idx] + best_local
                rr.log("metrics/checkpoint/best_slot_timestep", rr.Scalars(float(best_global)))
                if best_global != prev_checkpoint_ts:
                    prev_checkpoint_ts = best_global
                    stitched = _stitch_checkpoint_image(dataset, camera_keys, best_global)
                    if stitched is not None:
                        rr.log("checkpoint_view/stitched", rr.Image(stitched))

    print("\nDone streaming to Rerun.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream LeRobot dataset directly to Rerun.")
    parser.add_argument("--repo_id", type=str, required=True, help="Dataset repository ID")
    parser.add_argument("--root", type=str, default=None, help="Dataset root directory")
    parser.add_argument("--stride", type=int, default=7, help="Frame stride (controls speed)")
    parser.add_argument(
        "--num_episode", type=int, default=None, help="Visualise first N episodes (default: all)"
    )
    args = parser.parse_args()
    visualize_dataset(args.repo_id, args.root, args.stride, num_episode=args.num_episode)
