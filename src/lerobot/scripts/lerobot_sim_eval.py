#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate a policy in simulation with optional failure recovery and dataset recording.

Two complementary modes can be enabled independently:

Phase 1 — collect successful episodes as a LeRobotDataset (same logic as lerobot-record):
    --dataset.repo_id=user/sim_successes
    --dataset.single_task="Pick and place pudding box"
    --dataset.root=data/  # optional; defaults to HF_LEROBOT_HOME/repo_id

Phase 2 — failure recovery via FailurePostprocessor:
    Activated automatically when failure_handling.json is present in the pretrained model
    directory. Set enable_failure_handling=true in that JSON to turn on checkpoint recovery;
    logging-only mode is used otherwise.

Usage example:
```
lerobot-sim-eval \\
    --policy.path=outputs/train/libero_pudding/checkpoints/last/pretrained_model \\
    --env.type=libero \\
    --env.task=libero_object \\
    --env.task_ids="[8]" \\
    --eval.batch_size=2 \\
    --eval.n_episodes=50 \\
    --policy.use_amp=false \\
    --policy.device=cuda \\
    --dataset.repo_id=eval/libero_successes \\
    --dataset.single_task="Pick and place pudding box"
```
"""

import concurrent.futures as cf
import contextlib
import json
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any, TypedDict

import einops
import gymnasium as gym
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.sim_eval import SimEvalConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    close_envs,
    preprocess_observation,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_PREFIX, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _build_features_from_policy_cfg(policy_cfg) -> dict:
    """Build a LeRobotDataset feature spec from the policy's input/output feature config.

    This avoids a probe rollout: the policy config already declares the exact key names
    and shapes of every observation and action tensor.
    """
    features: dict[str, dict] = {}
    for key, feat in policy_cfg.input_features.items():
        if feat.type == FeatureType.VISUAL:
            features[key] = {
                "dtype": "video",
                "shape": tuple(feat.shape),
                "names": ["channels", "height", "width"],
            }
        else:
            features[key] = {"dtype": "float32", "shape": tuple(feat.shape), "names": None}
    for key, feat in policy_cfg.output_features.items():
        features[key] = {"dtype": "float32", "shape": tuple(feat.shape), "names": None}
    return features


def _save_successful_episodes(
    dataset: LeRobotDataset,
    rollout_data: dict,
    done_indices: Tensor,
    batch_successes: Tensor,
    task: str,
) -> int:
    """Write successful episodes from a rollout batch into the dataset.

    Each successful episode is committed with dataset.save_episode().
    Returns the number of episodes saved.
    """
    saved = 0
    obs_data = rollout_data[OBS_STR]  # {key: (batch, seq+1, *)}
    actions = rollout_data[ACTION]  # (batch, seq, action_dim)

    for ep_ix in range(actions.shape[0]):
        if not batch_successes[ep_ix].item():
            continue
        n_steps = done_indices[ep_ix].item() + 1
        for t in range(n_steps):
            frame: dict[str, Any] = {}
            for key, val in obs_data.items():
                frame[key] = val[ep_ix, t].cpu()
            frame[ACTION] = actions[ep_ix, t].cpu()
            frame["task"] = task
            dataset.add_frame(frame)
        dataset.save_episode()
        saved += 1

    return saved


# ---------------------------------------------------------------------------
# Core rollout (modified from lerobot_eval.py to handle failure recovery)
# ---------------------------------------------------------------------------


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    events: dict | None = None,
    dataset: LeRobotDataset | None = None,
    dataset_task: str = "",
    save_only_success: bool = True,
    rr_step_offset: int = 0,
) -> dict:
    """Batched policy rollout with failure recovery support.

    Identical to lerobot_eval.rollout() with one addition: after each
    policy.select_action() call, _failure_postprocessor.recovery_pending_wait
    is cleared immediately — in simulation there is no physical robot to wait for.
    fp.reset() is called at the end of each rollout (episode boundary).
    """
    assert isinstance(policy, nn.Module)

    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    # Per-env incremental buffers for immediate dataset saving.
    # Saving happens as soon as each env finishes so the user doesn't wait for
    # the slowest env in the batch before seeing episodes written to disk.
    if dataset is not None:
        ep_obs: list[list[dict]] = [[] for _ in range(env.num_envs)]
        ep_actions: list[list[np.ndarray]] = [[] for _ in range(env.num_envs)]
        ep_any_success: list[bool] = [False] * env.num_envs
    ep_saved_count: int = 0

    step = 0
    done = np.array([False] * env.num_envs)
    prev_done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),
        leave=False,
    )
    check_env_attributes_and_types(env)
    _fp = getattr(policy, "_failure_postprocessor", None)

    # Checkpoint-frame buffer: maps step -> first camera image (HWC uint8/float).
    # Capped at 400 frames to bound memory usage.
    _ck_img_buffer: dict[int, np.ndarray] = {}
    _prev_best_slot_t: int = -1
    # Track which failure_detection/* scalar keys have been logged so we can
    # log NaN for them during a recovery rewind.
    _logged_fd_keys: set[str] = set()

    while not np.all(done) and step < max_steps:
        observation = preprocess_observation(observation)
        observation = add_envs_task(env, observation)
        observation = env_preprocessor(observation)

        # Capture env-0 for Rerun after env_preprocessor (before GPU normalisation).
        if display_data:
            obs0_rerun: dict[str, np.ndarray] = {
                k: v[0].cpu().numpy() for k, v in observation.items() if isinstance(v, torch.Tensor)
            }
            # Buffer first camera image for checkpoint-frame display.
            for _ck_v in obs0_rerun.values():
                if isinstance(_ck_v, np.ndarray) and _ck_v.ndim == 3:
                    _ck_img_buffer[step] = _ck_v
                    if len(_ck_img_buffer) > 400:
                        del _ck_img_buffer[min(_ck_img_buffer)]
                    break

        # Buffer per-env observations for incremental dataset saving.
        if dataset is not None:
            for i in range(env.num_envs):
                if not prev_done[i]:
                    ep_obs[i].append(
                        {k: v[i].cpu() for k, v in observation.items() if isinstance(v, torch.Tensor)}
                    )

        # Save after env_preprocessor (state keys merged, images renamed) but before
        # preprocessor (which normalises and moves tensors to GPU).
        if return_observations:
            all_observations.append(
                {
                    k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in observation.items()
                    if isinstance(v, torch.Tensor)
                }
            )

        observation = preprocessor(observation)

        # Snapshot the current sim state before inference so the strategy can
        # checkpoint it alongside the delta action.  Only supported for
        # SyncVectorEnv wrapping LiberoEnv (which exposes get_abs_state).
        if _fp is not None and isinstance(env, gym.vector.SyncVectorEnv):
            try:
                abs_states = env.call("get_abs_state")
                _fp.set_pending_abs_state(abs_states)
            except Exception:  # nosec B110
                pass  # non-libero envs silently skip

        with torch.inference_mode():
            action = policy.select_action(observation)

        # Failure recovery: in simulation, clear the "wait for robot" flag immediately.
        # When abs states were checkpointed, restore the sim state directly instead of
        # replaying the (invalid for delta-control) checkpoint action.
        if _fp is not None and _fp.recovery_pending_wait:
            _fp.recovery_pending_wait = False
            logging.info(f"[SimEval] step {step}: failure recovery triggered — clearing wait flag")
            # Rerun: erase the invalid portion of the curves by logging NaN for every
            # step after the checkpoint up to (but not including) the current step.
            if display_data and _logged_fd_keys:
                _best_t = _fp.strategy.best_slot_timestep
                if _best_t >= 0:
                    for _gs in range(rr_step_offset + _best_t + 1, rr_step_offset + step):
                        rr.set_time("step", sequence=_gs)
                        for _k in _logged_fd_keys:
                            rr.log(_k, rr.Scalars(float("nan")))
            if _fp.config.enable_logging:
                _fp.log_recovery_frame()
                if _fp.config.flush_metrics_every_step:
                    _fp.flush()

            # Restore sim state from the checkpoint slot (libero / delta-action envs).
            recovery_abs_states = _fp.strategy.select_recovery_abs_state()
            if recovery_abs_states is not None and isinstance(env, gym.vector.SyncVectorEnv):
                for i, e in enumerate(env.envs):
                    if i < len(recovery_abs_states) and recovery_abs_states[i] is not None:
                        try:
                            e.set_abs_state(recovery_abs_states[i])
                        except Exception as exc:
                            logging.warning(f"[SimEval] Recovery: failed to set abs state for env {i}: {exc}")
                logging.info(
                    f"[SimEval] Recovery: restored abs sim state for {len(env.envs)} env(s) "
                    f"at checkpoint step {_fp.strategy.best_slot_timestep}"
                )
                # Use a zero-delta action so env.step() doesn't move the robot
                # away from the just-restored checkpoint position.
                action = torch.zeros_like(action)

        action = postprocessor(action)
        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition[ACTION]

        action_numpy: np.ndarray = action.to("cpu").numpy()
        assert action_numpy.ndim == 2

        # Rerun: log env-0 observation, action, and failure-detection metrics.
        if display_data:
            rr.set_time("step", sequence=rr_step_offset + step)
            action0 = {f"dim_{i}": float(v) for i, v in enumerate(action_numpy[0])}
            log_rerun_data(
                observation=obs0_rerun,
                action=action0,
                compress_images=display_compressed_images,
            )
            if _fp is not None:
                if step == 0:
                    # Build image views dynamically from the actual observation keys.
                    _img_views = []
                    for _k, _v in obs0_rerun.items():
                        if isinstance(_v, np.ndarray) and _v.ndim == 3:
                            _entity = _k if _k.startswith(OBS_PREFIX) else f"{OBS_STR}.{_k}"
                            _img_views.append(rrb.Spatial2DView(name=_entity.split(".")[-1], origin=_entity))
                    _img_views.append(
                        rrb.Spatial2DView(
                            name="checkpoint_frame", origin="failure_detection/checkpoint_frame"
                        )
                    )
                    rr.send_blueprint(
                        rrb.Blueprint(
                            rrb.Horizontal(
                                rrb.Grid(*_img_views, grid_columns=2),
                                rrb.Tabs(
                                    rrb.TimeSeriesView(name="TD", origin="failure_detection/td"),
                                    rrb.TimeSeriesView(
                                        name="Checkpoint Step", origin="failure_detection/checkpoint_step"
                                    ),
                                    rrb.TimeSeriesView(
                                        name="Attention Entropy", origin="failure_detection/attention_entropy"
                                    ),
                                    rrb.TimeSeriesView(
                                        name="Similarity", origin="failure_detection/similarity"
                                    ),
                                ),
                            ),
                            auto_views=False,
                        )
                    )
                # Log detector state — entity paths under failure_detection/td/.
                rr.log("failure_detection/td/td_raw", rr.Scalars(_fp.detector.td_raw))
                rr.log("failure_detection/td/td_smoothed", rr.Scalars(_fp.detector.td_smoothed))
                rr.log("failure_detection/td/is_failing", rr.Scalars(float(_fp.detector.is_failing())))
                rr.log(
                    "failure_detection/td/td_threshold", rr.Scalars(_fp.detector._config.failure_threshold)
                )
                _logged_fd_keys.update(
                    {
                        "failure_detection/td/td_raw",
                        "failure_detection/td/td_smoothed",
                        "failure_detection/td/is_failing",
                        "failure_detection/td/td_threshold",
                    }
                )
                # Log extra numeric fields from last_row, routed by key:
                #   attention_entropy → failure_detection/attention_entropy/
                #   *_similarity     → failure_detection/similarity/
                #   everything else  → failure_detection/td/
                if _fp.recorder.last_row is not None:
                    skip = {"td_raw", "td_smoothed", "episode", "global_step", "timestamp"}
                    for k, v in _fp.recorder.last_row.items():
                        if k in skip:
                            continue
                        with contextlib.suppress(TypeError, ValueError):
                            if k == "attention_entropy":
                                path = f"failure_detection/attention_entropy/{k}"
                            elif k.endswith("_similarity"):
                                path = f"failure_detection/similarity/{k}"
                            elif k == "best_slot_timestep":
                                path = f"failure_detection/checkpoint_step/{k}"
                            else:
                                path = f"failure_detection/td/{k}"
                            rr.log(path, rr.Scalars(float(v)))
                            _logged_fd_keys.add(path)
                # Log checkpoint frame whenever the best slot changes.
                _best_t = _fp.strategy.best_slot_timestep
                if _best_t != _prev_best_slot_t and _best_t >= 0 and _best_t in _ck_img_buffer:
                    _ck_arr = _ck_img_buffer[_best_t]
                    if (
                        _ck_arr.ndim == 3
                        and _ck_arr.shape[0] in (1, 3, 4)
                        and _ck_arr.shape[-1] not in (1, 3, 4)
                    ):
                        _ck_arr = np.transpose(_ck_arr, (1, 2, 0))
                    rr.log("failure_detection/checkpoint_frame", rr.Image(_ck_arr))
                    _prev_best_slot_t = _best_t

        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)

        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError(
                    "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                    "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
                )
            successes = final_info["is_success"].tolist()
        else:
            successes = [False] * env.num_envs

        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        # Buffer per-env actions and save immediately when each env finishes.
        if dataset is not None:
            for i in range(env.num_envs):
                if not prev_done[i]:
                    ep_actions[i].append(action_numpy[i])
                    if successes[i]:
                        ep_any_success[i] = True

            new_done = done & ~prev_done
            for i in range(env.num_envs):
                if not new_done[i]:
                    continue
                if not save_only_success or ep_any_success[i]:
                    for obs_frame, act in zip(ep_obs[i], ep_actions[i], strict=True):
                        frame = {**obs_frame, ACTION: torch.from_numpy(act), "task": dataset_task}
                        dataset.add_frame(frame)
                    dataset.save_episode()
                    ep_saved_count += 1
                    status = "success" if ep_any_success[i] else "failure"
                    logging.info(f"[SimEval] env {i} step {step}: episode saved immediately ({status})")
                ep_obs[i] = []
                ep_actions[i] = []
                ep_any_success[i] = False

        prev_done = done.copy()

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

        if events is not None and events.get("exit_early"):
            break

    if return_observations:
        observation = preprocess_observation(observation)
        observation = add_envs_task(env, observation)
        observation = env_preprocessor(observation)
        all_observations.append(
            {
                k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in observation.items()
                if isinstance(v, torch.Tensor)
            }
        )

    # Reset failure postprocessor at episode boundary.
    # On rerecord: discard buffered rows (before flush) and reset state without advancing
    # the episode counter, so the retried episode is recorded under the same episode index.
    if _fp is not None:
        if events is not None and events.get("rerecord_episode"):
            _fp.discard_current_episode()
            _fp.detector.reset()
            _fp.strategy.reset()
            _fp.perturbation.reset()
            _fp._step = 0
        else:
            if save_only_success:
                ep_success = torch.stack(all_successes, dim=1).any(dim=1).tolist()
                if not any(ep_success):
                    _fp.discard_current_episode()
            _fp.reset()

    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
        "episodes_saved": ep_saved_count,
    }
    if return_observations:
        stacked_observations = {}
        for key, val in all_observations[0].items():
            if not isinstance(val, torch.Tensor):
                continue
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


# ---------------------------------------------------------------------------
# eval_policy: batched evaluation + optional dataset recording
# ---------------------------------------------------------------------------


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    dataset: LeRobotDataset | None = None,
    dataset_task: str = "Sim eval task",
    display_data: bool = False,
    display_compressed_images: bool = False,
    events: dict | None = None,
    save_only_success: bool = True,
) -> dict:
    """Evaluate a policy for n_episodes, recording episodes to dataset if provided."""
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if not isinstance(policy, PreTrainedPolicy):
        exc = ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )
        try:
            from peft import PeftModel

            if not isinstance(policy, PeftModel):
                raise exc
        except ImportError:
            raise exc from None

    start = time.time()
    policy.eval()

    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []
    n_episodes_rendered = 0
    total_episodes_saved = 0

    def render_frame(env: gym.vector.VectorEnv):  # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    _rr_step_offset: int = 0
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs),
                start_seed + ((batch_ix + 1) * env.num_envs),
            )

        # Retry loop: left-arrow resets and replays the same batch with the same seeds.
        while True:
            if events is not None:
                events["exit_early"] = False
                events["rerecord_episode"] = False

            rollout_data = rollout(
                env=env,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                seeds=list(seeds) if seeds else None,
                return_observations=False,
                render_callback=render_frame if max_episodes_rendered > 0 else None,
                display_data=display_data,
                display_compressed_images=display_compressed_images,
                events=events,
                dataset=dataset,
                dataset_task=dataset_task,
                save_only_success=save_only_success,
                rr_step_offset=_rr_step_offset,
            )

            if events is not None and events.get("rerecord_episode"):
                # Subtract any episodes that were saved during this interrupted rollout
                # so the total count stays accurate, then retry the entire batch.
                total_episodes_saved -= rollout_data.get("episodes_saved", 0)
                logging.info(f"[SimEval] batch {batch_ix}: rerecord requested — retrying with same seeds")
                continue
            break

        # Advance the rerun global step offset so the next batch's curves don't overlap.
        _rr_step_offset += rollout_data["done"].shape[1]

        if events is not None and events.get("stop_recording"):
            logging.info("[SimEval] Stop requested — ending evaluation early.")
            break

        n_steps = rollout_data["done"].shape[1]
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()

        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        # Episodes are saved immediately inside rollout() as each env finishes.
        total_episodes_saved += rollout_data.get("episodes_saved", 0)

        # Video rendering
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break
                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    for thread in threads:
        thread.join()

    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
            "episodes_recorded": total_episodes_saved,
        },
    }
    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


# ---------------------------------------------------------------------------
# Multi-task eval (same structure as lerobot_eval.py)
# ---------------------------------------------------------------------------


class TaskMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]


ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths")


def eval_one(
    env: gym.vector.VectorEnv,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
    dataset: LeRobotDataset | None,
    dataset_task: str,
    display_data: bool = False,
    display_compressed_images: bool = False,
    events: dict | None = None,
    save_only_success: bool = True,
) -> TaskMetrics:
    task_result = eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        dataset=dataset,
        dataset_task=dataset_task,
        display_data=display_data,
        display_compressed_images=display_compressed_images,
        events=events,
        save_only_success=save_only_success,
    )
    per_episode = task_result["per_episode"]
    return TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
    )


def run_one(
    task_group: str,
    task_id: int,
    env,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
    dataset: LeRobotDataset | None,
    dataset_task: str,
    display_data: bool = False,
    display_compressed_images: bool = False,
    events: dict | None = None,
    save_only_success: bool = True,
):
    task_videos_dir = None
    if videos_dir is not None:
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)

    metrics = eval_one(
        env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        dataset=dataset,
        dataset_task=dataset_task,
        display_data=display_data,
        display_compressed_images=display_compressed_images,
        events=events,
        save_only_success=save_only_success,
    )
    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])
    return task_group, task_id, metrics


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
    dataset: LeRobotDataset | None = None,
    dataset_task: str = "Sim eval task",
    display_data: bool = False,
    display_compressed_images: bool = False,
    events: dict | None = None,
    save_only_success: bool = True,
) -> dict:
    start_t = time.time()
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    per_task_infos: list[dict] = []

    def _accumulate_to(group: str, metrics: dict):
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))
        _append("max_rewards", metrics.get("max_rewards"))
        _append("successes", metrics.get("successes"))
        paths = metrics.get("video_paths", [])
        if paths:
            group_acc[group]["video_paths"].extend(paths)
            overall["video_paths"].extend(paths)

    task_runner = partial(
        run_one,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        dataset=dataset,
        dataset_task=dataset_task,
        display_data=display_data,
        display_compressed_images=display_compressed_images,
        events=events,
        save_only_success=save_only_success,
    )

    if max_parallel_tasks <= 1:
        for task_group, task_id, env in tasks:
            tg, tid, metrics = task_runner(task_group, task_id, env)
            _accumulate_to(tg, metrics)
            per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
    else:
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            fut2meta = {}
            for task_group, task_id, env in tasks:
                fut = executor.submit(task_runner, task_group, task_id, env)
                fut2meta[fut] = (task_group, task_id)
            for fut in cf.as_completed(fut2meta):
                tg, tid, metrics = fut.result()
                _accumulate_to(tg, metrics)
                per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})

    def _agg_from_list(xs):
        if not xs:
            return float("nan")
        return float(np.nanmean(np.array(xs, dtype=float)))

    groups_aggregated = {}
    for group, acc in group_acc.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }

    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
    }

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@parser.wrap()
def sim_eval_main(cfg: SimEvalConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="sim_eval", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    # --- Keyboard listener ---
    listener, events = init_keyboard_listener()
    print(
        "Keyboard controls:\n"
        "  ← left arrow : rerecord current batch (same seeds)\n"
        "  → right arrow: skip current batch early\n"
        "  Esc          : stop evaluation\n"
    )

    # --- Make environments ---
    logging.info("Making environment.")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    # --- Make policy ---
    logging.info("Making policy.")
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )
    policy.eval()

    # --- Pre/post processors ---
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    # --- Dataset (Phase 1) ---
    # Mirrors lerobot-record: root=None → HF_LEROBOT_HOME/repo_id, root=path → path directly.
    dataset: LeRobotDataset | None = None
    if cfg.dataset is not None:
        logging.info("[SimEval] Phase 1: recording successful episodes as LeRobotDataset.")
        logging.info(
            f"  repo_id={cfg.dataset.repo_id}  root={cfg.dataset.root or '(HF_LEROBOT_HOME/repo_id)'}  "
            f"task={cfg.dataset.single_task}"
        )

        first_env = next(iter(next(iter(envs.values())).values()))
        fps = first_env.unwrapped.metadata.get("render_fps", cfg.dataset.fps)

        features = _build_features_from_policy_cfg(cfg.policy)
        logging.info(f"[SimEval] Dataset features: {list(features.keys())}")

        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            fps,
            root=cfg.dataset.root,
            use_videos=any(f["dtype"] == "video" for f in features.values()),
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            features=features,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
        )
        logging.info(f"[SimEval] Dataset root: {dataset.root}")

        # Save sim-eval config — mirrors lerobot-record's record_config.json so that
        # downstream tools (visualisers, failure-handling scripts) can find it.
        try:
            meta_dir = Path(dataset.root) / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            config_path = meta_dir / "record_config.json"
            with open(config_path, "w") as f:
                json.dump(asdict(cfg), f, indent=4, default=str)
            logging.info(f"[SimEval] Saved sim-eval config to {config_path}")
        except Exception as e:
            logging.warning(f"[SimEval] Failed to save sim-eval config: {e}")

    # When a dataset is being recorded, consolidate all eval outputs under
    # {dataset.root}/eval/ so everything lives in one place.
    # Fall back to cfg.output_dir when running without dataset recording.
    eval_output_dir = Path(dataset.root) / "eval" if dataset is not None else Path(cfg.output_dir)

    # --- Failure postprocessor (Phase 2) ---
    # Initialized after dataset so we can mirror lerobot-record and write
    # failure_metrics.jsonl into dataset.root (rather than cfg.output_dir).
    fp_output_dir = dataset.root if dataset is not None else cfg.output_dir
    if hasattr(policy, "init_failure_postprocessor"):
        pretrained_path = getattr(cfg.policy, "pretrained_path", None)
        failure_handling_json_path = (
            str(Path(pretrained_path) / "failure_handling.json") if pretrained_path else None
        )
        policy.init_failure_postprocessor(
            output_dir=fp_output_dir,
            failure_handling_json_path=failure_handling_json_path,
        )
        logging.info(f"[SimEval] Failure postprocessor initialized (output_dir={fp_output_dir}).")

    # --- Evaluate ---
    video_enc_ctx = VideoEncodingManager(dataset) if dataset is not None else nullcontext()
    with (
        video_enc_ctx,
        torch.no_grad(),
        torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
    ):
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=10,
            videos_dir=eval_output_dir / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
            dataset=dataset,
            dataset_task=cfg.dataset.single_task if cfg.dataset is not None else "",
            display_data=cfg.display_data,
            display_compressed_images=display_compressed_images,
            events=events,
            save_only_success=cfg.save_only_success,
        )
        print("Overall Aggregated Metrics:")
        print(info["overall"])
        for task_group, task_group_info in info.get("per_group", {}).items():
            print(f"\nAggregated Metrics for {task_group}:")
            print(task_group_info)

    # --- Finalize ---
    if listener is not None:
        listener.stop()

    if hasattr(policy, "finalize"):
        policy.finalize()

    if dataset is not None:
        logging.info(f"[SimEval] Finalizing dataset — {dataset.num_episodes} episodes recorded.")
        dataset.finalize()

    close_envs(envs)

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    eval_info_path = eval_output_dir / "eval_info.json"
    with open(eval_info_path, "w") as f:
        json.dump(info, f, indent=2)
    logging.info(f"[SimEval] Saved eval info to {eval_info_path}")

    # Save per-episode success labels (same format as episode_labels.json used by comparison tools)
    successes = info.get("overall", {}).get("successes", [])
    if successes:
        labels = {str(i): int(v) for i, v in enumerate(successes)}
        labels_path = eval_output_dir / "episode_labels.json"
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=4)
        logging.info(f"[SimEval] Saved episode labels to {labels_path}")

    logging.info("End of sim eval")


def main():
    init_logging()
    register_third_party_plugins()
    sim_eval_main()


if __name__ == "__main__":
    main()
