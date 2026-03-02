#!/usr/bin/env python
"""Compute Conformal Prediction threshold from failure_metrics.jsonl.

It extracts `temporal_disagreement` as calibration scores, computes:
    q_level = ceil((n + 1) * (1 - alpha)) / n
    cp_threshold = quantile(scores, min(q_level, 1.0))

Then writes `cp_threshold` to:
  <pretrained_path>/failure_handling.json
where `pretrained_path` is taken from record_config.json
(prefers policy.pretrained_path, then top-level pretrained_path).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and save CP threshold from temporal_disagreement.")
    parser.add_argument("--repo_id", type=str, required=True, help="Repo id, e.g. eval/eval_failure_metrics")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Miscoverage level alpha. Default 0.05 (confidence 95%%).",
    )
    parser.add_argument(
        "--cache_root",
        type=Path,
        default=Path("~/.cache/huggingface/lerobot").expanduser(),
        help="Lerobot cache root. Default: ~/.cache/huggingface/lerobot",
    )
    return parser.parse_args()


def load_temporal_disagreement(metrics_path: Path) -> np.ndarray:
    if not metrics_path.exists():
        raise FileNotFoundError(f"failure_metrics.jsonl not found: {metrics_path}")

    scores = []
    with metrics_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {metrics_path} at line {line_no}") from exc

            value = row.get("temporal_disagreement")
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"temporal_disagreement must be numeric, got {type(value).__name__} "
                    f"at line {line_no} in {metrics_path}"
                )
            scores.append(float(value))

    if not scores:
        raise ValueError(f"No valid temporal_disagreement values found in {metrics_path}")

    return np.array(scores, dtype=np.float64)


def compute_cp_threshold(calibration_scores: np.ndarray, alpha: float) -> tuple[float, float, int]:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n = len(calibration_scores)
    q_level = float(np.ceil((n + 1) * (1 - alpha)) / n)
    q_level = min(q_level, 1.0)
    cp_threshold = float(np.quantile(calibration_scores, q_level))
    return cp_threshold, q_level, n


def read_record_config(record_config_path: Path) -> dict:
    if not record_config_path.exists():
        raise FileNotFoundError(f"record_config.json not found: {record_config_path}")
    with record_config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_pretrained_path(record_config: dict) -> Path:
    policy_cfg = record_config.get("policy", {})
    pretrained_path = policy_cfg.get("pretrained_path") or record_config.get("pretrained_path")
    if not pretrained_path:
        raise KeyError("pretrained_path not found in record_config.json")
    return Path(pretrained_path)


def write_cp_threshold(
    failure_handling_path: Path,
    cp_threshold: float,
    record_config_path: Path,
    pretrained_path: Path,
) -> None:
    if not failure_handling_path.exists():
        raise FileNotFoundError(
            "failure_handling.json not found at path from record_config.json\n"
            f"record_config: {record_config_path}\n"
            f"pretrained_path(from json): {pretrained_path}\n"
            f"expected_file: {failure_handling_path}"
        )

    with failure_handling_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    config["metrics"]["temporal_disagreement"]["cp_threshold"] = cp_threshold

    with failure_handling_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
        file.write("\n")


def main() -> None:
    args = parse_args()

    repo_dir = args.cache_root / args.repo_id
    metrics_path = repo_dir / "failure_metrics.jsonl"
    record_config_path = repo_dir / "meta" / "record_config.json"

    calibration_scores = load_temporal_disagreement(metrics_path)
    cp_threshold, q_level, n = compute_cp_threshold(calibration_scores, args.alpha)

    print(f"repo_id: {args.repo_id}")
    print(f"samples (n): {n}")
    print(f"alpha: {args.alpha}")
    print(f"q_level: {q_level}")
    print(f"cp_threshold: {cp_threshold}")
    if not record_config_path.exists():
        print(
            f"❌ record_config.json not found at {record_config_path}. Cannot write CP threshold without it."
        )
        return
    record_config = read_record_config(record_config_path)
    pretrained_path = resolve_pretrained_path(record_config)
    failure_handling_path = pretrained_path / "failure_handling.json"
    write_cp_threshold(
        failure_handling_path=failure_handling_path,
        cp_threshold=cp_threshold,
        record_config_path=record_config_path,
        pretrained_path=pretrained_path,
    )
    print(f"written_to: {failure_handling_path}")


if __name__ == "__main__":
    main()
