#!/usr/bin/env python3
"""Benchmark per-stage timing of FailurePostprocessor._process_live.

Uses lerobot_pick_and_place_fm (act_fm, checkpoint 100000) with:
  - synthetic checkpoint_features.npz (encoder_out_mean, dim=512)
  - attention_entropy plugin enabled
  - no actual robot / dataset required

Each sample forces a full forward pass (queue cleared before each call) so
hooks fire and all stages have real work to do.

Usage:
    cd /mnt/Data/workspace/robot/lerobot_piper
    python tools/benchmark_failure_handling.py
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_DIR = ROOT / "outputs/train/lerobot_pick_and_place_fm/checkpoints/100000/pretrained_model"
N_SAMPLES = 10
WARMUP = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENC_DIM = 512  # dim_model from config.json
N_TEMPLATES = 5

STAGES = [
    "policy_forward",  # predict_action_chunk (model inference + hooks fire)
    "compute TIDE",  # causal Gaussian smoothing (numpy)
    "cos-sim computation",  # cosine-sim vs template (GPU)
    "plugin_metrics",  # attention entropy (GPU)
    "update_slot_bookkeeping",  # checkpoint slot bookkeeping
    "recorder_log",  # dict serialisation + buffer append
]


# ─── Helpers ──────────────────────────────────────────────────────────────────


def sync():
    """Synchronize CUDA so perf_counter measures wall time accurately."""
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def make_batch():
    """Return a batched observation dict on DEVICE — the format expected by
    select_action (after the to_batch_processor + device_processor pipeline).
    Shapes: state (1,28), images (1,3,480,640) per camera."""
    return {
        "observation.state": torch.randn(1, 28, dtype=torch.float32, device=DEVICE),
        "observation.images.left": torch.rand(1, 3, 480, 640, dtype=torch.float32, device=DEVICE),
        "observation.images.right": torch.rand(1, 3, 480, 640, dtype=torch.float32, device=DEVICE),
        "observation.images.middle": torch.rand(1, 3, 480, 640, dtype=torch.float32, device=DEVICE),
    }


def make_template_npz(path: Path, dim: int = ENC_DIM, n: int = N_TEMPLATES):
    """Write a synthetic encoder_out_mean template to an npz file."""
    vecs = np.random.randn(n, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    np.savez(path, encoder_out_mean=vecs)


def make_failure_handling_json(path: Path):
    cfg = {
        "enable_failure_handling": False,
        "enable_logging": True,
        "flush_metrics_every_step": False,
        "detector": {
            "failure_threshold": 0.21,
            "td_smoothing_sigma": 4.0,
            "td_window_size": 31,
            "td_rho": 1.0,
        },
        "strategy": {
            "name": "checkpoint",
            "feature_type": "encoder_out",
            "template_mode": "mean",
            "peak_timestep_threshold": 30,
        },
        "plugins": [{"name": "attention_entropy", "enabled": True}],
    }
    path.write_text(json.dumps(cfg, indent=2))


# ─── Instrumented _process_live ───────────────────────────────────────────────


def make_timed_process_live(postproc, policy, timings: dict):
    """Return an instrumented replacement for postproc._process_live."""

    def _timed(batch, intended_action, new_actions_chunk):
        step = postproc._step

        if step == 0:
            postproc.strategy.on_episode_start(intended_action)

        # ── Stage: compute_raw_td ──────────────────────────────────────────
        sync()
        t0 = time.perf_counter()
        if new_actions_chunk is not None:
            raw_td = postproc.detector.compute_raw_td(new_actions_chunk, policy)
        else:
            raw_td = 0.0
        sync()

        # ── Stage: compute TIDE (Gaussian smoothing — CPU/numpy) ────────
        smoothed_td = postproc.detector.update(raw_td)
        timings["compute TIDE"].append(time.perf_counter() - t0)

        # ── Stage: strategy_metrics (cosine-sim on GPU) ────────────────────
        sync()
        t0 = time.perf_counter()
        strategy_metrics = postproc.strategy.compute_step_metrics()
        sync()
        timings["cos-sim computation"].append(time.perf_counter() - t0)

        # ── Stage: plugin_metrics (attention entropy on GPU) ───────────────
        sync()
        t0 = time.perf_counter()
        plugin_metrics: dict = {}
        for p in postproc.plugins:
            plugin_metrics.update(p.compute())
        sync()
        timings["plugin_metrics"].append(time.perf_counter() - t0)

        all_metrics = {
            "td_raw": raw_td,
            "td_smoothed": smoothed_td,
            **strategy_metrics,
            **plugin_metrics,
        }

        # ── Stage: strategy_update (slot bookkeeping) ──────────────────────
        sync()
        t0 = time.perf_counter()
        postproc.strategy.update(step, intended_action, all_metrics)
        all_metrics["best_slot_timestep"] = postproc.strategy.best_slot_timestep
        postproc.strategy.reset_step_state()
        for p in postproc.plugins:
            p.reset_step_state()
        sync()
        timings["update_slot_bookkeeping"].append(time.perf_counter() - t0)

        # ── Stage: recorder_log ────────────────────────────────────────────
        t0 = time.perf_counter()
        if postproc.config.enable_logging:
            postproc.recorder.log(all_metrics)
        timings["recorder_log"].append(time.perf_counter() - t0)

        postproc._step += 1

        if postproc.detector.is_failing():
            postproc.perturbation.on_failure_detected()
            if postproc.config.enable_failure_handling and postproc.strategy.can_recover():
                return postproc._do_recovery(batch, intended_action)

        return postproc.perturbation.apply(intended_action)

    return _timed


# ─── Main ─────────────────────────────────────────────────────────────────────


def run(n_samples: int = N_SAMPLES, warmup: int = WARMUP):
    from lerobot.policies.act.modeling_act_fm import ACTFMPolicy
    from lerobot.policies.failure_postprocessor import FailurePostprocessor

    print(f"Device: {DEVICE}")
    print(f"Loading model from {MODEL_DIR} ...")
    policy = ACTFMPolicy.from_pretrained(MODEL_DIR)
    policy = policy.to(DEVICE)
    policy.eval()
    print("Model loaded.\n")

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        make_template_npz(tmpdir / "checkpoint_features.npz")
        make_failure_handling_json(tmpdir / "failure_handling.json")

        postproc = FailurePostprocessor(
            policy=policy,
            output_dir=None,  # no file I/O
            failure_handling_json_path=str(tmpdir / "failure_handling.json"),
            enable_logging=True,
        )
        policy._failure_postprocessor = postproc

        timings: dict[str, list[float]] = {s: [] for s in STAGES}

        # Patch postproc.process to use the instrumented live path
        timed_live = make_timed_process_live(postproc, policy, timings)

        def patched_process(batch, intended_action, new_actions_chunk, metrics_override=None):
            return timed_live(batch, intended_action, new_actions_chunk)

        postproc.process = patched_process

        # ── Instrument predict_action_chunk for forward-pass timing ─────────
        _orig_predict = policy.predict_action_chunk
        _fwd_buf = []  # mutable container so inner closure can write

        def timed_predict(batch_arg, **kw):
            sync()
            t0 = time.perf_counter()
            result = _orig_predict(batch_arg, **kw)
            sync()
            _fwd_buf.append(time.perf_counter() - t0)
            return result

        policy.predict_action_chunk = timed_predict

        def run_one():
            """One timed inference call (queue forced empty → forward pass runs)."""
            _fwd_buf.clear()
            # Reset policy (clears action queue) and postprocessor step counter
            policy.reset()
            postproc._step = 0
            postproc.detector.reset()

            batch = make_batch()
            with torch.no_grad():
                policy.select_action(batch)

            if _fwd_buf:
                timings["policy_forward"].append(_fwd_buf[0])

        # ── Warm-up ──────────────────────────────────────────────────────────
        print(f"Warm-up ({warmup} passes) ...")
        for _ in range(warmup):
            run_one()

        # Reset collected timings after warm-up
        timings = {s: [] for s in STAGES}
        timed_live_new = make_timed_process_live(postproc, policy, timings)

        def patched_process2(batch, intended_action, new_actions_chunk, metrics_override=None):
            return timed_live_new(batch, intended_action, new_actions_chunk)

        postproc.process = patched_process2

        _fwd_buf.clear()
        fwd_buf_ref = _fwd_buf  # same list

        def timed_predict2(batch_arg, **kw):
            sync()
            t0 = time.perf_counter()
            result = _orig_predict(batch_arg, **kw)
            sync()
            fwd_buf_ref.append(time.perf_counter() - t0)
            return result

        policy.predict_action_chunk = timed_predict2

        def run_one2():
            fwd_buf_ref.clear()
            policy.reset()
            postproc._step = 0
            postproc.detector.reset()
            batch = make_batch()
            with torch.no_grad():
                policy.select_action(batch)
            if fwd_buf_ref:
                timings["policy_forward"].append(fwd_buf_ref[0])

        # ── Measurement ──────────────────────────────────────────────────────
        print(f"Measuring {n_samples} samples ...")
        for i in range(n_samples):
            run_one2()
            print(f"  sample {i + 1}/{n_samples}")

    # ── Report ───────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print(f"  Failure Handling Pipeline Timing  ({n_samples} samples, {DEVICE})")
    print("=" * 62)
    print(f"  {'Stage':<25}  {'Mean (ms)':>10}  {'Std (ms)':>10}  {'n':>4}")
    print("  " + "-" * 56)
    for stage in STAGES:
        vals = timings[stage]
        if not vals:
            print(f"  {stage:<25}  {'—':>10}  {'—':>10}  {'0':>4}")
            continue
        arr = np.array(vals) * 1000.0  # s → ms
        print(f"  {stage:<25}  {arr.mean():>10.3f}  {arr.std():>10.3f}  {len(arr):>4}")
    print("=" * 62)

    overhead_stages = ["compute TIDE", "cos-sim computation", "update_slot_bookkeeping"]
    overhead_stages = [s for s in overhead_stages if timings[s]]
    # Per-sample total across overhead stages (aligned by index)
    n_min = min(len(timings[s]) for s in overhead_stages)
    per_sample_total = sum(np.array(timings[s][:n_min]) for s in overhead_stages) * 1000.0
    print(
        f"  {'Total (excl. policy_forward)':<25}  {per_sample_total.mean():>10.3f}  {per_sample_total.std():>10.3f}"
    )
    print("=" * 62)


if __name__ == "__main__":
    run()
