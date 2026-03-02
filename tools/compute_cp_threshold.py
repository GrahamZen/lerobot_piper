#!/usr/bin/env python
"""Compute Conformal Prediction threshold using Mahalanobis Distance Fusion.

It extracts 2D features: [temporal_disagreement, attention_local_variance]
from successful episodes in failure_metrics.jsonl.
Computes the mean vector (mu) and inverse covariance matrix (inv_cov).
Calculates Mahalanobis distances as calibration scores, computes:
    q_level = ceil((n + 1) * (1 - alpha)) / n
    cp_threshold = quantile(distances, min(q_level, 1.0))

Then writes `mu`, `inv_cov`, and `cp_threshold` to:
  <pretrained_path>/failure_handling.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CP threshold from 2D Mahalanobis distance.")
    parser.add_argument("--repo_id", type=str, required=True, help="Repo id, e.g. eval/eval_failure_metrics")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Miscoverage level alpha. Default 0.01 (confidence 99%%).",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="Window size for computing Attention Local Variance.",
    )
    parser.add_argument(
        "--cache_root",
        type=Path,
        default=Path("~/.cache/huggingface/lerobot").expanduser(),
        help="Lerobot cache root. Default: ~/.cache/huggingface/lerobot",
    )
    return parser.parse_args()


def load_and_extract_2d_features(metrics_path: Path, window_size: int) -> np.ndarray:
    """Read metrics, compute per-episode local variance, and return (N, 2) feature array."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"failure_metrics.jsonl not found: {metrics_path}")

    # Group rows by episode to avoid calculating variance across episode boundaries
    episodes_data = defaultdict(list)
    with metrics_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {metrics_path} at line {line_no}") from exc

            # Default episode to 0 if not present
            ep = row.get("episode", 0)
            episodes_data[ep].append(row)

    features_list = []

    for _ep, rows in episodes_data.items():
        # Ensure rows are sorted by step
        rows = sorted(rows, key=lambda r: r.get("step", 0))

        entropies = np.array([float(r.get("attention_entropy", 0.0)) for r in rows])
        tds = np.array([float(r.get("temporal_disagreement", 0.0)) for r in rows])

        # Calculate Local Variance with sliding window
        variances = np.zeros_like(entropies)
        for i in range(len(entropies)):
            start_idx = max(0, i - window_size)
            window = entropies[start_idx : i + 1]
            if len(window) > 1:
                variances[i] = np.var(window)
            else:
                variances[i] = 0.0

        # Stack TD and Variance as 2D features
        for i in range(len(rows)):
            features_list.append([tds[i], variances[i]])

    if not features_list:
        raise ValueError(f"No valid data found in {metrics_path}")

    return np.array(features_list, dtype=np.float64)


def compute_mahalanobis_and_cp(
    features: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """Compute distribution stats and Conformal Prediction threshold."""
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n = len(features)

    # 1. Compute Mean
    mu = np.mean(features, axis=0)

    # 2. Compute Covariance Matrix and its Inverse
    cov = np.cov(features, rowvar=False)
    # Add a tiny epsilon to the diagonal to prevent singular matrix errors
    # (e.g., if variance is perfectly constant in a dummy dataset)
    cov += np.eye(2) * 1e-8
    inv_cov = np.linalg.inv(cov)

    # 3. Compute Mahalanobis Distance for all calibration points
    diff = features - mu
    # D^2 = (X - mu)^T * Sigma^-1 * (X - mu)
    left = np.dot(diff, inv_cov)
    mahalanobis_sq = np.sum(left * diff, axis=1)
    distances = np.sqrt(np.abs(mahalanobis_sq))

    # 4. Conformal Prediction Threshold
    q_level = float(np.ceil((n + 1) * (1 - alpha)) / n)
    q_level = min(q_level, 1.0)
    cp_threshold = float(np.quantile(distances, q_level))

    return mu, inv_cov, cp_threshold, q_level, n


def read_record_config(record_config_path: Path) -> dict:
    if not record_config_path.exists():
        raise FileNotFoundError(f"record_config.json not found: {record_config_path}")
    else:
        print(f"✅ Found record_config.json at: {record_config_path}")
    with record_config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_pretrained_path(record_config: dict) -> Path:
    policy_cfg = record_config.get("policy", {})
    pretrained_path = policy_cfg.get("pretrained_path") or record_config.get("pretrained_path")
    if not pretrained_path:
        raise KeyError("pretrained_path not found in record_config.json")
    return Path(pretrained_path)


def write_fusion_config(
    failure_handling_path: Path,
    mu: np.ndarray,
    inv_cov: np.ndarray,
    cp_threshold: float,
    window_size: int,
) -> None:
    if not failure_handling_path.exists():
        raise FileNotFoundError(f"failure_handling.json not found at {failure_handling_path}")

    with failure_handling_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if "metrics" not in config:
        config["metrics"] = {}

    if "fusion_mahalanobis" not in config["metrics"]:
        config["metrics"]["fusion_mahalanobis"] = {}

    # Update with the computed multi-dimensional metrics
    config["metrics"]["fusion_mahalanobis"].update(
        {
            "enabled": True,
            "window_size": window_size,
            "cp_threshold": cp_threshold,
            "mu": mu.tolist(),
            "inv_cov": inv_cov.tolist(),
        }
    )

    with failure_handling_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
        file.write("\n")


def main() -> None:
    args = parse_args()

    repo_dir = args.cache_root / args.repo_id
    metrics_path = repo_dir / "failure_metrics.jsonl"
    record_config_path = repo_dir / "meta" / "record_config.json"

    # 1. Extract and process 2D features
    print(f"Loading data from {metrics_path}...")
    features = load_and_extract_2d_features(metrics_path, args.window_size)

    # 2. Compute Distribution and CP Threshold
    mu, inv_cov, cp_threshold, q_level, n = compute_mahalanobis_and_cp(features, args.alpha)

    print("\n--- Mahalanobis Calibration Results ---")
    print(f"Samples (n): {n}")
    print(f"Alpha: {args.alpha}, Q-Level: {q_level:.5f}")
    print(f"Mean (mu): [{mu[0]:.5f}, {mu[1]:.5f}]")
    print(f"CP Threshold: {cp_threshold:.5f}")

    if not record_config_path.exists():
        print(f"❌ record_config.json not found at {record_config_path}. Cannot write config.")
        return

    record_config = read_record_config(record_config_path)
    pretrained_path = resolve_pretrained_path(record_config)
    failure_handling_path = pretrained_path / "failure_handling.json"

    # 3. Write to JSON
    write_fusion_config(
        failure_handling_path=failure_handling_path,
        mu=mu,
        inv_cov=inv_cov,
        cp_threshold=cp_threshold,
        window_size=args.window_size,
    )
    print(f"\n✅ Successfully written fusion parameters to: {failure_handling_path}")


if __name__ == "__main__":
    main()
