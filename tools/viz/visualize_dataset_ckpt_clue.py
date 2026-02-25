"""
This script streams an entire LeRobot dataset to Rerun for continuous visualization.
It renders camera streams and the robot's 3D posture across all episodes in a single timeline.

Key Features:
- Continuous playback: Automatically scrolls through all episodes in the dataset.
- 3D Visualization: Uses forward kinematics to render the robot's arm and gripper state.
- Global Timeline: Displays a "global_step" timeline for the entire dataset plus per-episode steps.
- Episode Overlays: Shows the current episode index as a large 3D text overlay.
- Stride control: Adjust the playback speed using the --stride argument.

Usage:
    python tools/viz/visualize_dataset.py --repo_id local/lerobot_pick_and_place --stride 7
"""

import argparse
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# =========================
# Checkpoint Detection Config
# =========================
CHECKPOINT_SIGNAL_CONFIG = {
    "window_size": 31,
    "eval_delay": 15,
    "valley_lookback": 8,
    "valley_lookahead": 8,
    "smoothing_sigma": 2.0,
    "valley_prominence": None,
}


def load_checkpoint_signal_config_from_dataset(dataset_root):
    record_config_path = Path(dataset_root) / "meta" / "record_config.json"
    if not record_config_path.exists():
        print(
            f"Warning: record_config.json not found at {record_config_path}. Cannot load checkpoint signal config."
        )
        return {}

    try:
        with open(record_config_path) as f:
            record_config = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to parse record config at {record_config_path}: {e}")
        return {}

    pretrained_path = record_config.get("pretrained_path")
    if not pretrained_path and isinstance(record_config.get("policy"), dict):
        pretrained_path = record_config["policy"].get("pretrained_path")

    if not pretrained_path:
        for key in ("model", "train", "training"):
            section = record_config.get(key)
            if isinstance(section, dict) and section.get("pretrained_path"):
                pretrained_path = section["pretrained_path"]
                break

    if not pretrained_path:
        print(f"Warning: 'pretrained_path' not found in {record_config_path}")
        return {}

    pretrained_path = Path(pretrained_path).expanduser()
    failure_handling_json_path = pretrained_path / "failure_handling.json"
    if not failure_handling_json_path.exists():
        nested_candidate = pretrained_path / "pretrained_model" / "failure_handling.json"
        if nested_candidate.exists():
            failure_handling_json_path = nested_candidate
        else:
            print(
                "Warning: failure handling config not found. Tried "
                f"{failure_handling_json_path} and {nested_candidate}"
            )
            return {}

    try:
        with open(failure_handling_json_path) as f:
            failure_handling = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to parse failure handling config at {failure_handling_json_path}: {e}")
        return {}

    print(f"Loaded failure handling config from {failure_handling_json_path}")
    return failure_handling


def build_failed_points_table_markdown(
    failed_steps, failure_metrics, previous_checkpoint_by_step, cp_threshold
):
    title = "# Failed Points Summary"
    threshold_line = (
        f"\n\n- `cp_threshold`: `{cp_threshold:.6f}`"
        if cp_threshold is not None
        else "\n\n- `cp_threshold`: `N/A`"
    )

    table_header = (
        "\n\n| # | failed_step (x) | temporal_disagreement (y_td) | previous_checkpoint_step (y_cp) |\n"
        "| :--- | :--- | :--- | :--- |"
    )

    if not failed_steps:
        return f"{title}{threshold_line}{table_header}\n| - | - | - | - |"

    rows = []
    for idx, step in enumerate(failed_steps, start=1):
        td = float(failure_metrics.get(step, {}).get("temporal_disagreement", 0.0))
        prev_cp = previous_checkpoint_by_step.get(step, np.nan)
        prev_cp_text = "-" if (isinstance(prev_cp, float) and np.isnan(prev_cp)) else f"{float(prev_cp):.0f}"
        rows.append(f"| {idx} | {int(step)} | {td:.6f} | {prev_cp_text} |")

    return f"{title}{threshold_line}{table_header}\n" + "\n".join(rows)


def build_checkpoint_series(
    failure_metrics,
    window_size=31,
    eval_delay=15,
    valley_lookback=8,
    valley_lookahead=8,
    smoothing_sigma=2.0,
    valley_prominence=None,
):
    if not failure_metrics:
        return {}, {}, {}

    steps = np.array(sorted(failure_metrics.keys()), dtype=np.int64)

    smoothed_by_step = {}
    checkpoint_flag_by_step = {}
    previous_checkpoint_by_step = {}

    eval_delay = max(eval_delay, valley_lookahead)
    min_required = eval_delay + valley_lookback + 1
    window_size = max(window_size, min_required)

    recent_raw = deque(maxlen=window_size)
    recent_steps = deque(maxlen=window_size)

    checkpoint_set = set()
    latest_checkpoint = np.nan

    for step in steps:
        step_int = int(step)
        disagreement = float(failure_metrics[step_int].get("temporal_disagreement", 0.0))

        recent_raw.append(disagreement)
        recent_steps.append(step_int)

        if len(recent_raw) >= min_required:
            smoothed_window = gaussian_filter1d(np.array(recent_raw), sigma=smoothing_sigma)

            eval_idx = len(recent_raw) - 1 - eval_delay
            eval_val = smoothed_window[eval_idx]
            eval_step = recent_steps[eval_idx]

            smoothed_by_step[eval_step] = eval_val

            past_vals = smoothed_window[eval_idx - valley_lookback : eval_idx]
            future_vals = smoothed_window[eval_idx + 1 : eval_idx + 1 + valley_lookahead]

            current_prominence = valley_prominence
            if current_prominence is None:
                current_prominence = max(1e-6, 0.35 * float(np.std(smoothed_window)))

            is_valley = True

            if eval_val > min(past_vals) or eval_val >= min(future_vals):
                is_valley = False
            else:
                if (
                    max(past_vals) - eval_val < current_prominence
                    or max(future_vals) - eval_val < current_prominence
                ):
                    is_valley = False

            if is_valley:
                checkpoint_set.add(eval_step)
                latest_checkpoint = float(eval_step)

        checkpoint_flag_by_step[step_int] = 1.0 if step_int in checkpoint_set else 0.0
        previous_checkpoint_by_step[step_int] = latest_checkpoint

    for step in steps:
        if step not in smoothed_by_step:
            smoothed_by_step[step] = float(failure_metrics[step].get("temporal_disagreement", 0.0))

    return smoothed_by_step, previous_checkpoint_by_step, checkpoint_flag_by_step


class PiperFK:
    def __init__(self):
        self.RADIAN = 180 / math.pi
        self.PI = math.pi
        # DH parameters from piper_fk.py (dh_is_offset=1) - Converted to Meters
        self._a = [x / 1000.0 for x in [0, 0, 285.03, -21.98, 0, 0]]
        self._alpha = [0, -self.PI / 2, 0, self.PI / 2, -self.PI / 2, self.PI / 2]
        # Offset override from code logic if dh_is_offset=1
        self._theta_offset = [0, -self.PI * 172.22 / 180, -102.78 / 180 * self.PI, 0, 0, 0]
        self._d = [x / 1000.0 for x in [123, 0, 0, 250.75, 0, 91]]

        # Load mesh paths
        self.mesh_dir = Path("assets/piper_description/meshes")
        self.links = [
            "base_link",
            "link1",
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
            "gripper_base",
            "link7",
            "link8",
        ]
        # Colors (RGB)
        self.gray = [128, 128, 128]
        self.link_colors = {
            "base_link": self.gray,
            "link1": self.gray,
            "link2": self.gray,
            "link3": self.gray,
            "link4": self.gray,
            "link5": self.gray,
            "link6": self.gray,
            "gripper_base": self.gray,
            "link7": self.gray,
            "link8": self.gray,
        }

    def _link_transform(self, alpha, a, theta, d):
        ca, sa = math.cos(alpha), math.sin(alpha)
        ct, st = math.cos(theta), math.sin(theta)
        return np.array(
            [[ct, -st, 0, a], [st * ca, ct * ca, -sa, -sa * d], [st * sa, ct * sa, ca, ca * d], [0, 0, 0, 1]]
        )

    def get_transforms(self, joints, gripper_val=0):
        # joints: 6 angles
        # gripper_val: distance in meters
        transforms = {}
        transform = np.eye(4)
        transforms["base_link"] = transform.copy()

        # FK for 6 joints
        for i in range(6):
            theta = joints[i] + self._theta_offset[i]
            t_i = self._link_transform(self._alpha[i], self._a[i], theta, self._d[i])
            transform = transform @ t_i
            transforms[f"link{i + 1}"] = transform.copy()

        # Gripper Base (Fixed to link6)
        t_gripper_base = transform.copy()  # joint6_to_gripper_base origin is 0 0 0
        transforms["gripper_base"] = t_gripper_base

        # Gripper Fingers
        # Joint 7: origin 0 0 0.1358, rpy 1.5708 0 0. Prismatic z
        # Helper for RPY + XYZ fixed transform
        def get_fixed(xyz, rpy):
            cx, sx = math.cos(rpy[0]), math.sin(rpy[0])
            cy, sy = math.cos(rpy[1]), math.sin(rpy[1])
            cz, sz = math.cos(rpy[2]), math.sin(rpy[2])

            r_x = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]])
            r_y = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]])
            r_z = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

            t_r = np.eye(4)
            t_r[:3, 3] = xyz
            return t_r @ r_z @ r_y @ r_x

        # Link 7 (Left Finger)
        j7_origin = get_fixed([0, 0, 0.1358], [1.5708, 0, 0])

        # User specified gripper_val is the total distance between fingers.
        half_dist = gripper_val / 2.0
        j7_pos = max(0, min(half_dist, 0.035))

        t_prismatic = np.eye(4)
        t_prismatic[2, 3] = j7_pos
        transforms["link7"] = t_gripper_base @ j7_origin @ t_prismatic

        # Link 8 (Right Finger)
        j8_origin = get_fixed([0, 0, 0.1358], [1.5708, 0, -3.1416])
        j8_pos = j7_pos  # Symmetric movement in rotated frame

        t_prismatic8 = np.eye(4)
        t_prismatic8[2, 3] = j8_pos
        transforms["link8"] = t_gripper_base @ j8_origin @ t_prismatic8

        return transforms

    def log_initial_meshes(self, prefix="simulation"):
        # Log parsed meshes as Asset3D once
        for link in self.links:
            mesh_path = self.mesh_dir / f"{link}.STL"
            if mesh_path.exists():
                color = self.link_colors.get(link, [200, 200, 200])
                rr.log(f"{prefix}/{link}", rr.Asset3D(path=mesh_path, albedo_factor=color), static=True)


def visualize_dataset(repo_id, root=None, stride=7, checkpoint_signal_config=None):
    # Resolve local path to avoid HF Hub 401 errors
    expanded_root = Path(root).expanduser() if root else None
    dataset_path = Path(repo_id).expanduser()
    if not dataset_path.is_dir() and expanded_root:
        dataset_path = expanded_root / repo_id

    if dataset_path.is_dir():
        print(f"Found local dataset at: {dataset_path}")
        # Passing an absolute path as repo_id tells LeRobotDataset it's local
        repo_id = str(dataset_path)
        root = None
    else:
        root = expanded_root
        print(f"Loading dataset: {repo_id} (root: {root})")

    try:
        dataset = LeRobotDataset(repo_id, root=root)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Ensure episodes metadata
    if dataset.meta.episodes is None:
        try:
            from lerobot.datasets.utils import load_episodes

            dataset.meta.episodes = load_episodes(dataset.root)
        except Exception as e:
            print(f"Warning: Could not load episodes metadata: {e}")

    total_episodes = len(dataset.meta.episodes)
    print(f"Total episodes available: {total_episodes}")

    # Load failure metrics if they exist
    failure_metrics = {}
    metrics_path = Path(dataset.root) / "failure_metrics.jsonl"
    if metrics_path.exists():
        print(f"Found failure metrics at {metrics_path}")
        with open(metrics_path) as f:
            for line in f:
                try:
                    m = json.loads(line)
                    # Use step as the key for lookup
                    if "step" in m:
                        failure_metrics[int(m["step"])] = m
                except (KeyError, ValueError, TypeError):
                    continue
        print(f"Loaded {len(failure_metrics)} failure metric entries.")

    smoothed_td_by_step = {}
    previous_checkpoint_by_step = {}
    checkpoint_flag_by_step = {}
    failed_steps = []
    failed_step_set = set()
    cp_threshold = None

    failure_handling_cfg = load_checkpoint_signal_config_from_dataset(dataset.root)
    dataset_signal_cfg = {
        key: failure_handling_cfg[key] for key in CHECKPOINT_SIGNAL_CONFIG if key in failure_handling_cfg
    }
    if "cp_threshold" in failure_handling_cfg:
        try:
            cp_threshold = float(failure_handling_cfg["cp_threshold"])
            print(f"Loaded cp_threshold from failure_handling.json: {cp_threshold:.6f}")
        except (TypeError, ValueError):
            print(
                "Warning: cp_threshold exists in failure_handling config but is not a valid float. "
                f"Got: {failure_handling_cfg['cp_threshold']}"
            )

    signal_cfg = dict(CHECKPOINT_SIGNAL_CONFIG)
    if dataset_signal_cfg:
        signal_cfg.update(dataset_signal_cfg)
    if checkpoint_signal_config:
        signal_cfg.update(checkpoint_signal_config)

    effective_cfg_dump = {
        "window_size": signal_cfg["window_size"],
        "eval_delay": signal_cfg["eval_delay"],
        "valley_lookback": signal_cfg["valley_lookback"],
        "valley_lookahead": signal_cfg["valley_lookahead"],
        "smoothing_sigma": signal_cfg["smoothing_sigma"],
        "valley_prominence": signal_cfg["valley_prominence"],
        "cp_threshold": cp_threshold,
        "is_failing_rule": "temporal_disagreement > cp_threshold",
    }
    print("[Config Dump] Effective in-use config:")
    print(json.dumps(effective_cfg_dump, indent=2, ensure_ascii=False, sort_keys=True))

    if failure_metrics:
        smoothed_td_by_step, previous_checkpoint_by_step, checkpoint_flag_by_step = build_checkpoint_series(
            failure_metrics,
            window_size=signal_cfg["window_size"],
            eval_delay=signal_cfg["eval_delay"],
            valley_lookback=signal_cfg["valley_lookback"],
            valley_lookahead=signal_cfg["valley_lookahead"],
            smoothing_sigma=signal_cfg["smoothing_sigma"],
            valley_prominence=signal_cfg["valley_prominence"],
        )
        num_checkpoints = int(sum(checkpoint_flag_by_step.values()))
        print(
            f"Detected {num_checkpoints} checkpoint valleys from sliding-window Temporal Disagreement "
            f"(window_size={signal_cfg['window_size']}, lookback={signal_cfg['valley_lookback']}, "
            f"lookahead={signal_cfg['valley_lookahead']}, prominence={signal_cfg['valley_prominence']})."
        )

        if cp_threshold is not None:
            failed_steps = [
                step
                for step in sorted(failure_metrics.keys())
                if float(failure_metrics[step].get("temporal_disagreement", 0.0)) > cp_threshold
            ]
            failed_step_set = set(failed_steps)
            print(
                f"Detected {len(failed_steps)} failing points from is_failing rule "
                f"(temporal_disagreement > cp_threshold={cp_threshold:.6f})."
            )
        else:
            print("Warning: cp_threshold not found. Failed-point markers and table will be empty.")

    print("Initializing Rerun...")

    # Discover camera names from dataset metadata
    camera_names = []
    for key in dataset.meta.camera_keys:
        # key is like "observation.images.left" -> extract "left"
        camera_names.append(key.replace("observation.images.", ""))
    print(f"Camera views: {camera_names}")

    # Build blueprint with metrics panels if failure_metrics exist
    if failure_metrics:
        visible_camera_names = [
            cam for cam in camera_names if "left" not in cam.lower() and "right" not in cam.lower()
        ]
        camera_views = [rrb.Spatial2DView(origin=f"cameras/{cam}") for cam in visible_camera_names]
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="simulation"),
                    rrb.Vertical(*camera_views) if camera_views else rrb.Spatial3DView(origin="simulation"),
                    column_shares=[2, 1],
                ),
                # === Modified part: 3x2 layout for metrics ===
                rrb.Horizontal(
                    rrb.Vertical(
                        rrb.Horizontal(
                            rrb.TimeSeriesView(
                                name="Temporal Disagreement", origin="metrics/temporal_disagreement"
                            ),
                            rrb.TimeSeriesView(
                                name="Smoothed Temporal Disagreement",
                                origin="metrics/temporal_disagreement_smoothed",
                            ),
                        ),
                        rrb.Horizontal(
                            rrb.TimeSeriesView(name="Following Error", origin="metrics/following_error"),
                            rrb.TimeSeriesView(
                                name="Previous Checkpoint Step", origin="metrics/previous_checkpoint_step"
                            ),
                        ),
                        rrb.Horizontal(
                            rrb.TimeSeriesView(name="Attention Entropy", origin="metrics/attention_entropy"),
                            rrb.TimeSeriesView(
                                name="Mahalanobis Distance", origin="metrics/mahalanobis_distance"
                            ),
                        ),
                        rrb.Horizontal(
                            rrb.TimeSeriesView(name="Endpoint Shift", origin="metrics/endpoint_shift"),
                            rrb.TimeSeriesView(name="Action Jerk", origin="metrics/action_jerk"),
                        ),
                        rrb.Horizontal(
                            rrb.TimeSeriesView(
                                name="Attention Entropy Downward Slope",
                                origin="metrics/attention_entropy_downward_slope",
                            ),
                            rrb.TimeSeriesView(name="Checkpoint Flag", origin="metrics/checkpoint_flag"),
                        ),
                    ),
                    rrb.TextDocumentView(origin="summary/failed_points_table", name="Failed Points Table"),
                    column_shares=[3, 1],
                ),
                row_shares=[3, 3],
            ),
            collapse_panels=True,
        )
    else:
        blueprint = None

    rr.init("LeRobot Dataset Visualizer", spawn=True)
    if blueprint:
        rr.send_blueprint(blueprint)

    if failure_metrics:
        rr.log(
            "metrics/temporal_disagreement/failed_markers",
            rr.SeriesPoints(colors=[255, 0, 0], markers="circle", marker_sizes=6.0),
            static=True,
        )
        rr.log(
            "metrics/previous_checkpoint_step/failed_markers",
            rr.SeriesPoints(colors=[255, 0, 0], markers="circle", marker_sizes=6.0),
            static=True,
        )
        failed_table_md = build_failed_points_table_markdown(
            failed_steps,
            failure_metrics,
            previous_checkpoint_by_step,
            cp_threshold,
        )
        rr.log(
            "summary/failed_points_table",
            rr.TextDocument(failed_table_md, media_type=rr.MediaType.MARKDOWN),
            static=True,
        )

    fk = PiperFK()
    print("Logging initial meshes...")
    fk.log_initial_meshes("simulation/left_arm")
    fk.log_initial_meshes("simulation/right_arm")

    global_step = 0
    prev_attention_entropy = None
    prev_attention_step = None

    for episode_idx in range(total_episodes):
        print(f"Streaming Episode {episode_idx}/{total_episodes}...", end="\r")

        ep_meta = dataset.meta.episodes[episode_idx]
        from_idx = int(
            ep_meta["dataset_from_index"]
            if not isinstance(ep_meta["dataset_from_index"], list)
            else ep_meta["dataset_from_index"][0]
        )
        to_idx = int(
            ep_meta["dataset_to_index"]
            if not isinstance(ep_meta["dataset_to_index"], list)
            else ep_meta["dataset_to_index"][0]
        )

        # Log Episode ID Overlay
        # Using a separate log call at the start of the episode ensures it's visible even if we skip the first step?
        # But we log it continuously or just once? TextDocument is stateful.
        # Log it at the current global_step so it appears at the right time.

        for i in range(from_idx, to_idx, stride):
            # Set timelines - ONLY global_step to ensure continuous playback
            rr.set_time_sequence("global_step", global_step)
            global_step += 1

            # Update Overlay (Redundant to log every step but ensures it's always there)
            rr.log("overlay/episode_id", rr.TextDocument(f"# Episode {episode_idx}"), static=False)

            try:
                item = dataset[i]
            except Exception as e:
                print(f"Error loading frame {i}: {e}")
                continue

            # 1. Action vector (Simulation)
            # Find action key
            act = None
            if "action" in item:
                val = item["action"]
                if hasattr(val, "numpy"):
                    val = val.numpy()
                act = np.array(val)

            if act is not None:
                rr.log("vectors/action_tensor", rr.Tensor(act))

                if len(act) >= 14:
                    # Left Arm
                    left_joints = act[0:6]
                    left_grip = act[6]
                    left_transforms = fk.get_transforms(left_joints, left_grip)

                    t_left_base = np.eye(4)
                    t_left_base[1, 3] = 0.32

                    for link_name, t_local in left_transforms.items():
                        t_global = t_left_base @ t_local
                        rr.log(
                            f"simulation/left_arm/{link_name}",
                            rr.Transform3D(translation=t_global[:3, 3], mat3x3=t_global[:3, :3]),
                        )

                    # Right Arm
                    right_joints = act[7:13]
                    right_grip = act[13]
                    right_transforms = fk.get_transforms(right_joints, right_grip)

                    t_right_base = np.eye(4)
                    t_right_base[1, 3] = -0.32

                    for link_name, t_local in right_transforms.items():
                        t_global = t_right_base @ t_local
                        rr.log(
                            f"simulation/right_arm/{link_name}",
                            rr.Transform3D(translation=t_global[:3, 3], mat3x3=t_global[:3, :3]),
                        )

            # 2. Images
            image_keys = [k for k in item if "image" in k]
            for img_key in image_keys:
                img_data = item[img_key]
                clean_key = img_key.replace("observation.images.", "")

                # Convert to numpy/image
                if isinstance(img_data, dict) and "bytes" in img_data:
                    import io

                    img = Image.open(io.BytesIO(img_data["bytes"]))
                    rr.log(f"cameras/{clean_key}", rr.Image(img))
                else:
                    arr = img_data
                    if hasattr(arr, "numpy"):
                        arr = arr.numpy()
                    if arr.ndim == 3 and arr.shape[0] <= 4:
                        arr = np.transpose(arr, (1, 2, 0))

                    rr.log(f"cameras/{clean_key}", rr.Image(arr))

            # 3. Observation Text
            step_data = {}
            for key, value in item.items():
                if "observation.state" in key:
                    val = value
                    if hasattr(val, "tolist"):
                        val = val.tolist()
                    elif hasattr(val, "numpy"):
                        val = val.numpy().tolist()
                    step_data[key] = val
            if step_data:
                rr.log("vectors/observation_text", rr.TextDocument(str(step_data)))

            # 4. Failure Metrics (Rerun Scalars)
            if failure_metrics:
                m = failure_metrics.get(i, {})
                raw_temporal_disagreement = float(m.get("temporal_disagreement", 0.0))
                rr.log("metrics/temporal_disagreement/value", rr.Scalars(raw_temporal_disagreement))
                rr.log(
                    "metrics/temporal_disagreement_smoothed",
                    rr.Scalars(smoothed_td_by_step.get(i, raw_temporal_disagreement)),
                )
                rr.log("metrics/following_error", rr.Scalars(m.get("following_error", 0.0)))
                attention_entropy = float(m.get("attention_entropy", 0.0))
                rr.log("metrics/attention_entropy", rr.Scalars(attention_entropy))

                if prev_attention_entropy is None or prev_attention_step is None or i == prev_attention_step:
                    attention_entropy_downward_slope = 0.0
                else:
                    attention_entropy_downward_slope = (prev_attention_entropy - attention_entropy) / float(
                        i - prev_attention_step
                    )
                rr.log(
                    "metrics/attention_entropy_downward_slope",
                    rr.Scalars(attention_entropy_downward_slope),
                )

                prev_attention_entropy = attention_entropy
                prev_attention_step = i
                rr.log("metrics/mahalanobis_distance", rr.Scalars(m.get("mahalanobis_distance", 0.0)))
                rr.log("metrics/endpoint_shift", rr.Scalars(m.get("endpoint_shift", 0.0)))
                rr.log("metrics/action_jerk", rr.Scalars(m.get("action_jerk", 0.0)))
                rr.log(
                    "metrics/previous_checkpoint_step/value",
                    rr.Scalars(previous_checkpoint_by_step.get(i, np.nan)),
                )
                rr.log("metrics/checkpoint_flag", rr.Scalars(checkpoint_flag_by_step.get(i, 0.0)))

                if i in failed_step_set:
                    rr.log(
                        "metrics/temporal_disagreement/failed_markers",
                        rr.Scalars(raw_temporal_disagreement),
                    )
                    rr.log(
                        "metrics/previous_checkpoint_step/failed_markers",
                        rr.Scalars(previous_checkpoint_by_step.get(i, np.nan)),
                    )

    print("\nDone streaming to Rerun.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream LeRobot dataset directly to Rerun.")
    parser.add_argument("--repo_id", type=str, help="Dataset repository ID")
    parser.add_argument("--root", type=str, default=None, help="Dataset root")
    parser.add_argument("--stride", type=int, default=7, help="Visualization stride (speed)")

    args = parser.parse_args()
    visualize_dataset(args.repo_id, args.root, args.stride)
