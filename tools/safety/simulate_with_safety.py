"""
Run a trained ACT policy through the safety filter and visualise the result
in MuJoCo so that you can determine good constraint boundaries.

Because there is no real-robot calibration here, the robot pose in MuJoCo is
driven purely by the policy's predicted actions (forward pass on dataset
observations).  The constraint box is rendered as a semi-transparent red-orange
cube so you can visually judge which bounds are tight vs. too loose.

Dataset discovery
-----------------
By default the script reads ``<pretrained_path>/train_config.json`` and extracts
``dataset.repo_id`` and ``dataset.root`` automatically — you only need to pass
``--pretrained_path``.  If ``root`` is absent from train_config.json the dataset
is resolved under ``cache_root`` (default: ~/.cache/huggingface/lerobot).
Override either value explicitly with ``--repo_id`` / ``--dataset_root``.

Workflow
--------
1. Watch the robot move freely (no safety filter) to see where the policy
   naturally wants to go — this tells you the natural workspace envelope.
2. Edit constraint_config.json (R key hot-reloads it without restarting).
3. Enable the safety filter (--filter) and replay to see filtered motion.
4. Iterate until the box boundaries look right.

Controls
--------
  SPACE  — start / replay trajectory
  R      — hot-reload constraint_config.json and reset robot to home pose
  F      — toggle safety filter on / off at runtime

Usage
-----
    # auto-discover dataset from train_config.json (simplest)
    python tools/safety/simulate_with_safety.py \\
        --pretrained_path /path/to/model

    # same, but with safety filter on from the start
    python tools/safety/simulate_with_safety.py \\
        --pretrained_path /path/to/model \\
        --filter

    # explicit dataset override
    python tools/safety/simulate_with_safety.py \\
        --pretrained_path /path/to/model \\
        --repo_id my_dataset \\
        --dataset_root /data/my_dataset

    # override a constraint bound on the command line
    python tools/safety/simulate_with_safety.py \\
        --pretrained_path /path/to/model \\
        --constraint.z_min 0.05

    # headless (no viewer, just prints per-step constraint margin)
    python tools/safety/simulate_with_safety.py \\
        --pretrained_path /path/to/model \\
        --no_viz
"""

import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import draccus
import numpy as np
import torch

# ── project root on sys.path so local imports work ───────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "tools" / "safety") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tools" / "safety"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.safety import (
    RobotKinematicsInfo,
    enforce_safe_action,
    load_constraint_config,
    make_box_handles,
)
from mujoco_viz import PiperMuJoCoViz

# ── default paths ─────────────────────────────────────────────────────────────
DEFAULT_PRETRAINED = str(
    _REPO_ROOT
    / "outputs/train/lerobot_long_horizon_cup_torque_spact/checkpoints/last/pretrained_model"
)
DEFAULT_URDF = str(
    _REPO_ROOT / "assets/piper_dual_description/urdf/piper_dual_description.urdf"
)
DEFAULT_CONFIG = str(_REPO_ROOT / "tools/safety/constraint_config.json")
DEFAULT_CACHE_ROOT = str(Path.home() / ".cache/huggingface/lerobot")


# ── dataset discovery ─────────────────────────────────────────────────────────

def resolve_training_dataset_root(
    pretrained_path: Path, cache_root: Path
) -> tuple[Path, str]:
    """
    Derive dataset root and repo_id from ``<pretrained_path>/train_config.json``.

    Reads ``dataset.repo_id`` and optionally ``dataset.root``.
    Falls back to ``cache_root / repo_id`` when ``root`` is absent.
    """
    train_config_path = pretrained_path / "train_config.json"
    if not train_config_path.exists():
        raise FileNotFoundError(f"train_config.json not found: {train_config_path}")
    with train_config_path.open() as f:
        train_cfg = json.load(f)
    repo_id: str = train_cfg["dataset"]["repo_id"]
    root = train_cfg["dataset"].get("root")
    dataset_root = Path(root).expanduser() if root else cache_root / repo_id
    return dataset_root, repo_id


# ── CLI config ────────────────────────────────────────────────────────────────

@dataclass
class SimSafetyConfig:
    """
    Configuration for the safety-filter policy simulation.

    All constraint parameters come from the JSON file named by `config`.
    Edit that file to adjust bounds; press R in the viewer to hot-reload.

    Fields
    ------
    pretrained_path : path to a pretrained ACT model directory.
    repo_id         : dataset repo/local ID.
                      Leave empty to auto-read from train_config.json.
    dataset_root    : root directory for the LeRobot dataset.
                      Leave empty to auto-derive from train_config.json
                      (falls back to cache_root/repo_id).
    cache_root      : HuggingFace cache root used when dataset.root is absent
                      from train_config.json.
    episode_index   : which episode to use as observations (None → random).
    urdf            : robot URDF used for safety-filter kinematics + MuJoCo viz.
    config          : path to BoxConstraintConfig JSON.
    filter          : enable safety filter from the start (toggle with F key).
    no_viz          : headless mode — run once and print statistics.
    dt              : seconds between visualiser frames.
    """

    pretrained_path: str = DEFAULT_PRETRAINED
    repo_id: str = ""
    dataset_root: str = ""
    cache_root: str = DEFAULT_CACHE_ROOT
    episode_index: int | None = None
    urdf: str = DEFAULT_URDF
    config: str = DEFAULT_CONFIG
    filter: bool = False       # start with filter ON?
    no_viz: bool = False
    dt: float = 0.03


# ── helpers ───────────────────────────────────────────────────────────────────

def _pick_episode(dataset: LeRobotDataset, idx: int | None) -> tuple[int, int, int]:
    """Return (episode_idx, start_frame, length)."""
    n = len(dataset.meta.episodes)
    ep_idx = np.random.randint(n) if idx is None else idx
    ep = dataset.meta.episodes[ep_idx]
    if "index" in ep:
        start = ep["index"]
    elif "dataset_from_index" in ep:
        v = ep["dataset_from_index"]
        start = v.item() if hasattr(v, "item") else (v[0] if isinstance(v, (list, tuple, np.ndarray)) else v)
    else:
        raise KeyError("Cannot determine episode start index")
    return ep_idx, int(start), int(ep["length"])


def _run_policy_on_episode(
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
    dataset: LeRobotDataset,
    start: int,
    length: int,
) -> list[np.ndarray]:
    """
    Run the policy over an entire episode and collect predicted actions.

    Returns a flat list of per-step action arrays (logical-DOF length).
    The policy produces overlapping chunks; we concatenate them in order,
    advancing by chunk_size each time (same as simulate_episode.py).
    """
    device = next(policy.parameters()).device
    actions: list[np.ndarray] = []

    with torch.inference_mode():
        frame = start
        while frame < start + length:
            item = dataset[frame]
            batch = {k: v.unsqueeze(0).to(device) for k, v in item.items() if isinstance(v, torch.Tensor)}
            batch = preprocessor(batch)
            chunk = policy.predict_action_chunk(batch)
            chunk_np = postprocessor(chunk)[0].cpu().numpy()   # (chunk_size, action_dim)
            for step in chunk_np:
                actions.append(step)
            frame += chunk_np.shape[0]

    return actions


def _box_dict(box) -> dict:
    return {
        "x_min": box.x_min, "x_max": box.x_max,
        "y_min": box.y_min, "y_max": box.y_max,
        "z_min": box.z_min, "z_max": box.z_max,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def run(cfg: SimSafetyConfig) -> None:
    # ── 1. load model ─────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading model from {cfg.pretrained_path} …")
    policy = ACTPolicy.from_pretrained(cfg.pretrained_path)
    policy.eval()
    print("      Model loaded.")

    # ── 2. load dataset ───────────────────────────────────────────────────────
    # Resolve repo_id / dataset_root from train_config.json when not given.
    repo_id = cfg.repo_id
    dataset_root = cfg.dataset_root
    if not repo_id or not dataset_root:
        print(f"[2/4] Auto-discovering dataset from train_config.json …")
        auto_root, auto_repo_id = resolve_training_dataset_root(
            Path(cfg.pretrained_path), Path(cfg.cache_root)
        )
        if not repo_id:
            repo_id = auto_repo_id
        if not dataset_root:
            dataset_root = str(auto_root)
        print(f"      repo_id={repo_id}  root={dataset_root}")

    print(f"[2/4] Loading dataset '{repo_id}' from {dataset_root} …")
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=cfg.pretrained_path,
        dataset_stats=dataset.meta.stats,
    )
    ep_idx, start, length = _pick_episode(dataset, cfg.episode_index)
    print(f"      Episode {ep_idx}  frames {start}–{start + length - 1}  (length {length})")

    # ── 3. safety filter backend ──────────────────────────────────────────────
    print(f"[3/4] Loading safety filter (Pinocchio) …")
    box = load_constraint_config(cfg.config)
    kin = RobotKinematicsInfo(
        cfg.urdf,
        critical_link_indices=box.critical_links if box.critical_links else None,
    )
    handles = make_box_handles(box)
    filter_enabled = cfg.filter
    print(f"      nq={kin.nq}  logical_dof={kin.logical_dof}")
    print(f"      Critical links: {kin.critical_link_names}")
    print(f"      Box: x=[{box.x_min},{box.x_max}]  y=[{box.y_min},{box.y_max}]  z=[{box.z_min},{box.z_max}]")
    print(f"      Safety filter: {'ON' if filter_enabled else 'OFF'}")

    # ── 4. MuJoCo visualiser ──────────────────────────────────────────────────
    viz: PiperMuJoCoViz | None = None
    play_event   = threading.Event()
    reload_event = threading.Event()
    filter_event = threading.Event()   # toggle filter

    SPACE = 32
    R_KEY = 82
    F_KEY = 70

    if not cfg.no_viz:
        print(f"[4/4] Starting MuJoCo viewer …")
        viz = PiperMuJoCoViz(cfg.urdf, constraint_box=_box_dict(box))
        def _on_key(keycode: int) -> None:
            if keycode == SPACE and not play_event.is_set():
                play_event.set()
            elif keycode == R_KEY:
                reload_event.set()
            elif keycode == F_KEY:
                filter_event.set()
        viz.start(key_callback=_on_key)
        print(f"      SPACE = play  |  R = reload JSON + reset  |  F = toggle filter")
    else:
        print("[4/4] Headless mode — no viewer.")

    # ── helpers used inside the loop ──────────────────────────────────────────
    q_home = np.zeros(kin.logical_dof)

    def _do_reload() -> None:
        nonlocal box, handles
        try:
            new_box = load_constraint_config(cfg.config)
        except Exception as exc:
            print(f"[reload] ERROR: {exc}", flush=True)
            return
        box = new_box
        handles[:] = make_box_handles(new_box)
        if viz is not None:
            viz.update_constraint_box(_box_dict(new_box))
            viz.set_qpos(list(q_home))
        print(f"[reload] Box updated: "
              f"x=[{new_box.x_min},{new_box.x_max}]  "
              f"y=[{new_box.y_min},{new_box.y_max}]  "
              f"z=[{new_box.z_min},{new_box.z_max}]", flush=True)

    def _wait_for_space(label: str) -> bool:
        """Block until SPACE, handle R/F in the meantime. Returns False if viewer closed."""
        nonlocal filter_enabled
        if viz is None:
            return True
        print(f"\n  >>> SPACE = {label}  |  R = reload JSON  |  F = toggle filter <<<", flush=True)
        while not play_event.is_set():
            if not viz.is_running():
                return False
            if reload_event.is_set():
                reload_event.clear()
                _do_reload()
            if filter_event.is_set():
                filter_event.clear()
                filter_enabled = not filter_enabled
                print(f"[toggle] Safety filter: {'ON' if filter_enabled else 'OFF'}", flush=True)
            time.sleep(0.05)
        play_event.clear()
        return viz.is_running()

    # ── replay loop ───────────────────────────────────────────────────────────
    run_index = 0
    while True:
        # Generate fresh actions each run (re-runs the policy)
        print(f"\n[run {run_index + 1}] Running policy on episode {ep_idx} …", flush=True)
        trajectory = _run_policy_on_episode(
            policy, preprocessor, postprocessor, dataset, start, length
        )
        n_steps = len(trajectory)
        action_dim = trajectory[0].shape[0]
        print(f"      {n_steps} steps  action_dim={action_dim}  "
              f"filter={'ON' if filter_enabled else 'OFF'}")

        if viz is not None:
            if not _wait_for_space("start" if run_index == 0 else "replay"):
                break

        # per-step stats
        n_filtered    = 0
        min_h_safe    = np.inf
        max_violation = 0.0

        print(f"\n{'Step':>5}  {'h_raw_min':>10}  {'h_safe_min':>10}  {'filtered':>8}")
        print("-" * 47)

        for i, q_raw in enumerate(trajectory):
            # Compute raw constraint margin
            kin.compute_kinematics_and_jacobians(kin.expand(q_raw))
            h_raw_min = min(
                h(kin.get_frame_position(fid))
                for fid in kin.critical_frame_ids
                for h in handles
            )
            max_violation = min(max_violation, h_raw_min)

            # Apply filter (or not)
            if filter_enabled:
                q_out = enforce_safe_action(q_raw, kin, handles)
            else:
                q_out = q_raw

            was_filtered = filter_enabled and not np.allclose(q_raw, q_out, atol=1e-8)
            if was_filtered:
                n_filtered += 1

            kin.compute_kinematics_and_jacobians(kin.expand(q_out))
            h_safe_min = min(
                h(kin.get_frame_position(fid))
                for fid in kin.critical_frame_ids
                for h in handles
            )
            min_h_safe = min(min_h_safe, h_safe_min)

            if i % 20 == 0 or was_filtered:
                mark = " ← FILTERED" if was_filtered else ""
                print(f"{i:>5}  {h_raw_min:>+10.4f}  {h_safe_min:>+10.4f}  "
                      f"{str(was_filtered):>8}{mark}", flush=True)

            if viz is not None and viz.is_running():
                # Handle hot-reload / filter toggle while playing
                if reload_event.is_set():
                    reload_event.clear()
                    _do_reload()
                if filter_event.is_set():
                    filter_event.clear()
                    filter_enabled = not filter_enabled
                    print(f"[toggle] Safety filter: {'ON' if filter_enabled else 'OFF'}", flush=True)

                # Use logical DOF if viz expects it
                if len(q_out) == viz.dof:
                    viz.set_qpos(list(q_out))
                else:
                    # Policy dim may be 14 (joints only, no gripper) while viz
                    # expects 14 too — pass as-is; mismatch will raise in viz.
                    viz.set_qpos(list(q_out[: viz.dof]))
                time.sleep(cfg.dt)

        print("-" * 47)
        passed = min_h_safe >= -1e-4
        print(f"\nResults (run {run_index + 1})")
        print(f"  Steps            : {n_steps}")
        print(f"  Steps filtered   : {n_filtered}")
        print(f"  Worst raw h      : {max_violation:+.4f} m  "
              f"({'in box' if max_violation >= 0 else 'VIOLATION'})")
        print(f"  Min safe h       : {min_h_safe:+.4f} m")
        print(f"  Filter passed    : {passed}")
        if not filter_enabled and max_violation < 0:
            print(f"\n  Tip: the policy exceeds the constraint by "
                  f"{abs(max_violation):.4f} m — consider widening the box "
                  f"or enabling --filter.")

        run_index += 1

        if viz is None or not viz.is_running():
            break

        # Wait for SPACE before resetting to home pose
        if not _wait_for_space("reset + replay"):
            break
        if viz is not None:
            viz.set_qpos(list(q_home))

    if viz is not None:
        viz.close()


# ── CLI entry point ───────────────────────────────────────────────────────────

@draccus.wrap()
def main(cfg: SimSafetyConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
