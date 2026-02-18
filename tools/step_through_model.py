"""
Interactive step-through script for ACT model inference on a real robot.

Each press of the right arrow key triggers one cycle:
  1) Gather the robot's current observation
  2) Predict a full action chunk from the ACT model
  3) Execute all actions in the chunk on the robot
  4) Wait for the next right arrow key press

Usage:
  uv run python tools/step_through_model.py \
      --pretrained_path outputs/train/my_model/checkpoints/last/pretrained_model \
      --repo_id lerobot_pick_cups \
      --task "Pick up the cups"

Press RIGHT ARROW to step, ESC to quit.
"""

import argparse
import json
import time
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import make_robot_from_config
from lerobot.robots.piper_dual import PIPERDualConfig
from lerobot.utils.constants import OBS_STR
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging

# Defaults
DEFAULT_PRETRAINED_MODEL_PATH = Path(
    "/home/droplab/workspace/lerobot_piper/outputs/train/lerobot_simple_cups/checkpoints/last/pretrained_model"
)
DEFAULT_DATASET_ROOT = "~/.cache/huggingface/lerobot/local/"  # Uses HF_LEROBOT_HOME default
DEFAULT_DATASET_ID = "local/lerobot_simple_cups"
DEFAULT_TASK = "Pick up the cups"


def predict_action_chunk(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
) -> torch.Tensor:
    """
    Predict a full action chunk from the ACT model.

    Similar to predict_action() in control_utils.py, but calls
    predict_action_chunk() instead of select_action() to get the
    entire chunk of actions at once.
    """
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type)
        if device.type == "cuda" and use_amp
        else torch.inference_mode(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        observation = preprocessor(observation)

        # Get the full action chunk
        action_chunk = policy.predict_action_chunk(observation)

        # Postprocess the full chunk
        action_chunk = postprocessor(action_chunk)
        print(action_chunk)

    return action_chunk


def log_cameras_to_rerun(obs_processed: dict, step: int):
    """
    Log camera images from the processed observation to Rerun.
    """
    rr.set_time("step", sequence=step)
    for key, value in obs_processed.items():
        if not isinstance(value, np.ndarray) or value.ndim != 3:
            continue
        img = value
        # Convert CHW -> HWC if needed
        if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))
        # Normalize float images to 0-255
        if img.dtype in (np.float32, np.float64):
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        rr.log(f"cameras/{key}", rr.Image(img))


def execute_action_chunk(
    action_chunk: torch.Tensor,
    robot,
    dataset_features: dict,
    robot_action_processor: RobotProcessorPipeline,
    obs: dict,
    fps: int,
):
    """
    Execute every action in the chunk sequentially on the robot at the given FPS.

    Args:
        action_chunk: (1, chunk_size, action_dim) tensor of actions.
        robot: Connected robot instance.
        dataset_features: Dataset features dict for make_robot_action.
        robot_action_processor: Processor pipeline for robot actions.
        obs: Current raw observation (for the processor pipeline).
        fps: Target frames per second for execution.
    """
    # action_chunk shape: (batch=1, chunk_size, action_dim)
    chunk = action_chunk.squeeze(0)  # (chunk_size, action_dim)
    chunk_size = chunk.shape[0]

    print(f"  Executing {chunk_size} actions in chunk at {fps} FPS...")

    for step_idx in range(chunk_size):
        start_t = time.perf_counter()

        # Extract single action: (action_dim,)
        single_action = chunk[step_idx].unsqueeze(0)  # (1, action_dim) — add batch dim back

        # Convert tensor to named robot action dict
        robot_action = make_robot_action(single_action, dataset_features)

        # Process action through the robot action processor
        action_to_send = robot_action_processor((robot_action, obs))

        # Send to robot
        robot.send_action(action_to_send)

        # Maintain target FPS timing
        elapsed = time.perf_counter() - start_t
        sleep_time = max(1.0 / fps - elapsed, 0.0)
        precise_sleep(sleep_time)

        if (step_idx + 1) % 10 == 0 or step_idx == chunk_size - 1:
            print(f"    Step {step_idx + 1}/{chunk_size} done")


def main():
    parser = argparse.ArgumentParser(
        description="Step through ACT model inference interactively with right arrow key."
    )
    parser.add_argument(
        "--pretrained_path",
        type=Path,
        default=DEFAULT_PRETRAINED_MODEL_PATH,
        help="Path to pretrained ACT model checkpoint",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_DATASET_ID,
        help="Dataset repository ID (for features/stats)",
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root directory (default: HF_LEROBOT_HOME)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="Task description string for the policy",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS for action execution (default: 30)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (default: cuda)",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="Number of action steps to execute per chunk (default: full chunk size)",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help='JSON string of camera configs, e.g. \'{"left":{"type":"opencv","index_or_path":"/dev/video14","width":640,"height":480,"fps":60,"rotation":0}}\'',
    )
    parser.add_argument(
        "--ema",
        type=str,
        default="false",
        help="Apply EMA smoothing to the action chunk (default: false)",
    )
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=0.3,
        help="EMA smoothing factor alpha in (0,1]. Smaller = smoother (default: 0.3)",
    )

    args = parser.parse_args()
    args.ema = args.ema.lower() in ("true", "1", "yes")

    init_logging()

    # ── 1. Load Policy ──────────────────────────────────────────
    print(f"Loading ACT policy from {args.pretrained_path}...")
    policy = ACTPolicy.from_pretrained(args.pretrained_path)
    policy.eval()
    device = get_safe_torch_device(args.device)
    policy.to(device)
    n_action_steps = args.n_action_steps if args.n_action_steps is not None else policy.config.chunk_size
    print(
        f"  Policy loaded on {device}. Chunk size: {policy.config.chunk_size}, n_action_steps: {n_action_steps}"
    )

    # ── 2. Load Dataset (read-only, for features & stats) ───────
    dataset = LeRobotDataset(repo_id="local/lerobot_pick_cups_aug")
    print(f"  Dataset loaded. Stats available: {list(dataset.meta.stats.keys())[:5]}...")

    # ── 3. Create Processors ────────────────────────────────────
    print("Creating pre/post-processors...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.pretrained_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
        },
    )

    _, robot_action_processor, robot_observation_processor = make_default_processors()

    # ── 4. Connect Robot ────────────────────────────────────────
    print("Creating and connecting robot (PIPERDual)...")
    robot_cfg = PIPERDualConfig()

    # Override cameras if --cameras is provided
    if args.cameras is not None:
        cameras_dict = json.loads(args.cameras)
        robot_cfg.cameras = {}
        for cam_name, cam_params in cameras_dict.items():
            # Remove 'type' key (not an OpenCVCameraConfig field)
            cam_params = {k: v for k, v in cam_params.items() if k != "type"}
            robot_cfg.cameras[cam_name] = OpenCVCameraConfig(**cam_params)
            print(f"  Camera '{cam_name}': {cam_params}")

    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    print("  Robot connected.")

    # ── 5. Build dataset features for frame construction ────────
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    # ── 6. Initialize Rerun ──────────────────────────────────────
    print("Initializing Rerun viewer...")
    rr.init("step_through_model", spawn=True)

    # ── 7. Interactive Loop ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  INTERACTIVE STEP-THROUGH MODE")
    print("  Press RIGHT ARROW to step (observe → predict → execute)")
    print("  Press ESC to quit")
    print("=" * 60 + "\n")

    # Reset the policy for a fresh episode
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    step_count = 0
    running = True

    # Use pynput for non-blocking key detection with a blocking wait pattern
    from pynput import keyboard

    # Shared state for key events
    key_event = {"step": False, "quit": False}

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                key_event["step"] = True
            elif key == keyboard.Key.esc:
                key_event["quit"] = True
        except Exception:
            # Squelch errors from pynput listener
            pass  # nosec B110

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        while running:
            # Wait for a key event
            print(f"[Step {step_count}] Waiting for RIGHT ARROW to step (ESC to quit)...")

            while not key_event["step"] and not key_event["quit"]:
                time.sleep(0.05)

            if key_event["quit"]:
                print("\nESC pressed. Exiting...")
                break

            # Reset the step flag
            key_event["step"] = False
            step_count += 1

            print(f"\n── Step {step_count} ─────────────────────────────")

            # 1) Gather observation
            print("  Gathering observation...")
            raw_obs = robot.get_observation()
            obs_processed = robot_observation_processor(raw_obs)

            # Build observation frame (dict of numpy arrays keyed by dataset feature names)
            observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
            print(f"  Observation keys: {list(observation_frame.keys())}")

            # Log observation images to Rerun
            log_cameras_to_rerun(obs_processed, step_count)

            # 2) Predict action chunk
            print("  Predicting action chunk...")
            action_chunk = predict_action_chunk(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=args.task,
                robot_type=robot.robot_type,
            )
            # Optionally apply EMA smoothing
            if args.ema:
                # Per-joint mask: True = apply EMA, False = keep raw
                # Indices: [L1-L6, L_grip, R1-R6, R_grip]
                ema_mask = [
                    False,  # 0  left joint 1
                    True,  # 1  left joint 2
                    True,  # 2  left joint 3
                    False,  # 3  left joint 4
                    False,  # 4  left joint 5
                    False,  # 5  left joint 6
                    False,  # 6  left gripper
                    False,  # 7  right joint 1
                    True,  # 8  right joint 2
                    True,  # 9  right joint 3
                    False,  # 10 right joint 4
                    False,  # 11 right joint 5
                    False,  # 12 right joint 6
                    False,  # 13 right gripper
                ]
                ema_mask_t = torch.tensor(ema_mask, dtype=torch.bool, device=action_chunk.device)

                chunk_2d = action_chunk.squeeze(0)  # (chunk_size, action_dim)
                smoothed = chunk_2d.clone()
                alpha = args.ema_alpha
                for i in range(1, chunk_2d.shape[0]):
                    smoothed[i, ema_mask_t] = (
                        alpha * chunk_2d[i, ema_mask_t] + (1 - alpha) * smoothed[i - 1, ema_mask_t]
                    )
                action_chunk = smoothed.unsqueeze(0)
                print(f"  EMA smoothing applied (alpha={alpha}), mask={ema_mask}")

            chunk_shape = action_chunk.shape
            print(f"  Action chunk shape: {chunk_shape}")

            # Print first and last action in chunk for reference
            chunk_np = action_chunk.squeeze(0).cpu().numpy()
            print(f"  First action:  {np.array2string(chunk_np[0], precision=3, suppress_small=True)}")
            print(f"  Last action:   {np.array2string(chunk_np[-1], precision=3, suppress_small=True)}")

            # 3) Execute action chunk on robot (sliced to n_action_steps)
            execute_action_chunk(
                action_chunk=action_chunk[:, :n_action_steps, :],
                robot=robot,
                dataset_features=dataset_features,
                robot_action_processor=robot_action_processor,
                obs=raw_obs,
                fps=args.fps,
            )

            print(f"  ✓ Step {step_count} complete.\n")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt. Shutting down...")
    finally:
        print("Disconnecting robot...")
        listener.stop()
        if robot.is_connected:
            robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
