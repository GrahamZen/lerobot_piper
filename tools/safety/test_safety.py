"""
Unit test: apply a dangerous trajectory through the safety filter and
visualise the result in MuJoCo.

Architecture
------------
  Data / control layer  →  safety_filter.py   (Pinocchio + QP)
  Rendering / validation → mujoco_viz.py       (MuJoCo passive viewer)

The test generates a sweep trajectory that intentionally drives one or
more critical links outside the configured 3-D box constraint, then verifies
that enforce_safe_action() keeps every link inside the box throughout.

The box constraint is loaded from a JSON config file via draccus
(default: tools/safety/constraint_config.json) with fields:
    x_min, x_max, y_min, y_max, z_min, z_max   (metres, world frame)

Usage
-----
    # default config and URDF
    python tools/safety/test_safety.py

    # custom constraint config (draccus reads the JSON)
    python tools/safety/test_safety.py --config tools/safety/constraint_config.json

    # custom URDF
    python tools/safety/test_safety.py \\
        --urdf assets/piper_urdf/piper_description/urdf/piper_description.urdf

    # override a single constraint bound on the command line
    python tools/safety/test_safety.py --constraint.z_min 0.25

    # run without opening a viewer (CI / headless)
    python tools/safety/test_safety.py --no_viz
"""

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import draccus
import numpy as np

# ── project imports ──────────────────────────────────────────────────────────
from lerobot.safety import (
    RobotKinematicsInfo,
    enforce_safe_action,
    load_constraint_config,
    make_box_handles,
)
from mujoco_viz import PiperMuJoCoViz

# ── default paths ────────────────────────────────────────────────────────────
_REPO_ROOT   = Path(__file__).resolve().parents[2]
DEFAULT_URDF = str(
    _REPO_ROOT / "assets/piper_dual_description/urdf/piper_dual_description.urdf"
)
DEFAULT_CONFIG = str(_REPO_ROOT / "tools/safety/constraint_config.json")


# ── CLI config dataclass ──────────────────────────────────────────────────────

@dataclass
class TestConfig:
    """
    Configuration for the safety-filter unit test.

    All constraint parameters are read exclusively from the JSON file
    specified by `config`.  Edit that file to change bounds or links.

    Fields
    ------
    urdf:
        Path to the robot URDF (single or dual arm).
    config:
        Path to the JSON box-constraint config file (BoxConstraintConfig).
    steps:
        Number of trajectory steps.
    dt:
        Seconds between visualiser frames.
    no_viz:
        Disable MuJoCo viewer (headless / CI mode).
    """
    urdf: str = DEFAULT_URDF
    config: str = DEFAULT_CONFIG
    steps: int = 200
    dt: float = 0.03
    no_viz: bool = False


# ── trajectory generation ────────────────────────────────────────────────────

def generate_dangerous_trajectory(
    q_start: np.ndarray,
    logical_dof: int,
    n_steps: int = 200,
) -> list[np.ndarray]:
    """
    Two-phase trajectory in **logical DOF** space.

    Works for both single arm (logical_dof=7) and dual arm (logical_dof=14).

    Phase 1 — free descent:
        shoulder (j2) and elbow (j3) sweep to extreme "downward" values.
    Phase 2 — constrained slide:
        The policy keeps pressing against the floor while the base joint(s)
        rotate.  The safety filter blocks the descent and lets the arm slide.
    """
    n1 = n_steps // 3
    n2 = n_steps - n1

    per_arm = logical_dof // 2 if logical_dof > 7 else logical_dof
    n_arms  = logical_dof // per_arm

    q_down  = q_start.copy()
    q_slide = q_start.copy()

    for arm in range(n_arms):
        base = arm * per_arm
        q_down[base + 1]  = 2.8
        q_slide[base + 1] = 2.8
        q_down[base + 2]  = -2.5
        q_slide[base + 2] = -2.1
        q_slide[base + 0] = 1.8 if arm == 0 else -1.8

    phase1 = [
        (1.0 - t) * q_start + t * q_down
        for t in np.linspace(0, 1, n1)
    ]
    phase2 = [
        (1.0 - t) * q_down + t * q_slide
        for t in np.linspace(0, 1, n2)
    ]
    return phase1 + phase2


# ── main test ────────────────────────────────────────────────────────────────

def run_test(cfg: TestConfig) -> dict:
    """
    Run the safety-filter unit test.

    Returns a result dict with keys:
        n_steps        — total trajectory steps
        n_filtered     — steps where the filter modified q
        max_violation  — worst (most negative) min-h over all raw steps
        min_h_safe     — smallest min-h over all filtered steps
        passed         — True if min_h_safe >= -1e-4 (numerically safe)
    """
    box = load_constraint_config(cfg.config)

    print(f"\n{'='*60}")
    print(f"Safety Filter Unit Test")
    print(f"  URDF       : {Path(cfg.urdf).name}")
    print(f"  Config     : {Path(cfg.config).name}")
    print(f"  Box        : x=[{box.x_min}, {box.x_max}]  "
          f"y=[{box.y_min}, {box.y_max}]  z=[{box.z_min}, {box.z_max}]  (m)")
    print(f"  Steps      : {cfg.steps}")
    print(f"{'='*60}\n")

    # ── 1. Pinocchio backend ──────────────────────────────────────────────────
    print("[1/3] Loading Pinocchio model...")
    kin = RobotKinematicsInfo(
        cfg.urdf,
        critical_link_indices=box.critical_links if box.critical_links else None,
    )
    print(f"      nq={kin.nq}  logical_dof={kin.logical_dof}  "
          f"links checked: {kin.critical_link_names}")

    handles = make_box_handles(box)

    q_start    = np.zeros(kin.logical_dof)
    trajectory = generate_dangerous_trajectory(q_start, kin.logical_dof, cfg.steps)

    # ── 2. MuJoCo frontend ───────────────────────────────────────────────────
    viz = None
    play_event   = threading.Event()
    reload_event = threading.Event()

    # handles is a mutable list so the reload callback can swap its contents
    handles = make_box_handles(box)

    if not cfg.no_viz:
        print("[2/3] Loading MuJoCo visualiser...")
        box_dict = {
            "x_min": box.x_min, "x_max": box.x_max,
            "y_min": box.y_min, "y_max": box.y_max,
            "z_min": box.z_min, "z_max": box.z_max,
        }
        viz = PiperMuJoCoViz(cfg.urdf, constraint_box=box_dict)

        SPACE = 32
        R_KEY = 82   # GLFW key code for 'R'

        def _on_key(keycode: int) -> None:
            if keycode == SPACE and not play_event.is_set():
                play_event.set()
            elif keycode == R_KEY:
                reload_event.set()

        viz.start(key_callback=_on_key)
        print(f"      logical DOF={viz.dof}  physical DOF={viz.dof_physical}")
        print("      R = hot-reload constraint JSON + reset robot")
    else:
        print("[2/3] Visualiser disabled (--no_viz).")

    def _do_reload() -> None:
        """Reload constraint JSON, update handles and box geom in place."""
        nonlocal handles
        try:
            new_box = load_constraint_config(cfg.config)
        except Exception as e:
            print(f"[reload] ERROR reading config: {e}", flush=True)
            return
        handles[:] = make_box_handles(new_box)
        if viz is not None:
            new_box_dict = {
                "x_min": new_box.x_min, "x_max": new_box.x_max,
                "y_min": new_box.y_min, "y_max": new_box.y_max,
                "z_min": new_box.z_min, "z_max": new_box.z_max,
            }
            viz.update_constraint_box(new_box_dict)
            viz.set_qpos(list(q_start))
        print(f"[reload] Constraint updated: "
              f"x=[{new_box.x_min},{new_box.x_max}] "
              f"y=[{new_box.y_min},{new_box.y_max}] "
              f"z=[{new_box.z_min},{new_box.z_max}]", flush=True)

    # ── 3. Replay loop ────────────────────────────────────────────────────────
    print("[3/3] Running trajectory...\n")

    result = {}
    run_index = 0

    while True:
        # Wait for Space (or run immediately in headless mode).
        # Poll so R-key reloads can be processed while waiting.
        if viz is not None:
            if not viz.is_running():
                break
            label = "start" if run_index == 0 else "replay"
            print(f"  >>> SPACE = {label}  |  R = reload JSON + reset <<<",
                  flush=True)
            while not play_event.is_set():
                if not viz.is_running():
                    break
                if reload_event.is_set():
                    reload_event.clear()
                    _do_reload()
                time.sleep(0.05)
            play_event.clear()
            if not viz.is_running():
                break
            print(f"\n[run {run_index + 1}] Starting trajectory.", flush=True)

        n_filtered    = 0
        max_violation = 0.0
        min_h_safe    = np.inf

        print(f"{'Step':>5}  {'h_raw_min':>10}  {'h_safe_min':>10}  {'filtered':>8}")
        print("-" * 45)

        for i, q_raw in enumerate(trajectory):
            kin.compute_kinematics_and_jacobians(kin.expand(q_raw))
            h_raw_min = min(
                h(kin.get_frame_position(fid))
                for fid in kin.critical_frame_ids
                for h in handles
            )
            max_violation = min(max_violation, h_raw_min)

            q_safe = enforce_safe_action(q_raw, kin, handles)

            was_filtered = not np.allclose(q_raw, q_safe, atol=1e-8)
            if was_filtered:
                n_filtered += 1

            kin.compute_kinematics_and_jacobians(kin.expand(q_safe))
            h_safe_min = min(
                h(kin.get_frame_position(fid))
                for fid in kin.critical_frame_ids
                for h in handles
            )
            min_h_safe = min(min_h_safe, h_safe_min)

            if i % 20 == 0 or was_filtered:
                mark = " ← FILTERED" if was_filtered else ""
                print(f"{i:>5}  {h_raw_min:>+10.4f}  {h_safe_min:>+10.4f}  "
                      f"{str(was_filtered):>8}{mark}")

            if viz is not None and viz.is_running():
                viz.set_qpos(list(q_safe))
                time.sleep(cfg.dt)

        print("-" * 45)

        # ── per-run report ────────────────────────────────────────────────────
        passed = min_h_safe >= -1e-4
        result = dict(
            n_steps=cfg.steps,
            n_filtered=n_filtered,
            max_violation=max_violation,
            min_h_safe=min_h_safe,
            passed=passed,
        )
        print(f"\nResults (run {run_index + 1}):")
        print(f"  Steps filtered      : {n_filtered} / {cfg.steps}")
        print(f"  Worst raw violation : h = {max_violation:+.4f} m")
        print(f"  Min safe h          : h = {min_h_safe:+.4f} m")
        print(f"  Test PASSED         : {passed}")
        if not passed:
            print(f"\n  !! FAIL: safety filter allowed h = {min_h_safe:.4f} m < 0")

        run_index += 1

        if viz is None or not viz.is_running():
            break   # headless: single run

        # Wait for Space before resetting to home pose
        print(f"  >>> SPACE = reset + replay  |  R = reload JSON + reset <<<",
              flush=True)
        while not play_event.is_set():
            if not viz.is_running():
                break
            if reload_event.is_set():
                reload_event.clear()
                _do_reload()
            time.sleep(0.05)
        play_event.clear()
        if not viz.is_running():
            break
        viz.set_qpos(list(q_start))

    if viz is not None:
        viz.close()

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_link_table(urdf_path: str) -> None:
    """Print the index ↔ link-name table for the given URDF."""
    import xml.etree.ElementTree as ET
    try:
        links = [
            el.get("name")
            for el in ET.parse(urdf_path).getroot().iter("link")
        ]
    except Exception:
        return
    print(f"\ncritical_links index reference  ({Path(urdf_path).name})")
    print("─" * 42)
    for i, name in enumerate(links):
        print(f"  {i:>3}  {name}")
    print("─" * 42)
    print('Set "critical_links": [<idx>, ...]  in constraint_config.json.')
    print("Empty list → auto-detect (link4 / link6 / gripper_base).\n")


@draccus.wrap()
def main(cfg: TestConfig) -> None:
    result = run_test(cfg)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    import sys
    if "-h" in sys.argv or "--help" in sys.argv:
        _print_link_table(DEFAULT_URDF)
    main()
