"""
Check and fix near-zero std issues in a pretrained model's normalizer.

When a feature (e.g. a gripper joint) never moves across all demonstrations,
its std in the normalizer safetensors will be essentially zero. During inference,
any sensor noise on that feature gets divided by this near-zero std (or by the
eps=1e-8 fallback), amplifying noise by up to 1e8x and causing catastrophic
shaking behavior.

This script detects affected dimensions, reports which features they correspond
to, and optionally fixes them by setting a safe minimum std floor.

Usage:
    uv run python tools/fix_normalizer_std.py <pretrained_model_path> [--fix] [--floor 0.01]

Arguments:
    pretrained_model_path   Path to the pretrained_model directory
                            (contains config.json + *.safetensors)
    --fix                   Apply the fix in-place (backs up originals first)
    --floor FLOAT           Minimum std floor to apply (default: 0.01)
    --eps FLOAT             Normalization eps used by lerobot (default: 1e-8)

Examples:
    # Check only
    uv run python tools/fix_normalizer_std.py outputs/train/my_model/checkpoints/last/pretrained_model

    # Check and fix
    uv run python tools/fix_normalizer_std.py outputs/train/my_model/checkpoints/last/pretrained_model --fix
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

NORMALIZER_FILE = "policy_preprocessor_step_3_normalizer_processor.safetensors"
# Only observation inputs matter — action is output (unnormalized by multiply, not divide)
INPUT_KEYS = ["observation.state"]


def load_feature_names(model_dir: Path) -> list[str] | None:
    """Try to load feature names from config.json."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        cfg = json.load(f)
    state_feature = cfg.get("input_features", {}).get("observation.state", {})
    names = state_feature.get("names")
    return names


def check_and_fix(model_dir: Path, fix: bool, floor: float, eps: float):
    sf_path = model_dir / NORMALIZER_FILE
    if not sf_path.exists():
        print(f"ERROR: normalizer file not found: {sf_path}")
        return

    feature_names = load_feature_names(model_dir)

    tensors = {}
    with safe_open(sf_path, framework="pt") as f:
        for k in f.keys():  # noqa: SIM118
            tensors[k] = f.get_tensor(k).clone()

    problems = []
    for key in INPUT_KEYS:
        std_key = f"{key}.std"
        if std_key not in tensors:
            continue
        std = tensors[std_key]
        for i, v in enumerate(std):
            v = v.item()
            if v < eps:
                # std < eps means eps-clamping kicks in, amplification = 1/eps
                dim_name = feature_names[i] if feature_names and i < len(feature_names) else f"dim {i}"
                problems.append((key, i, dim_name, v))

    if not problems:
        print("OK: no near-zero std issues found.")
        return

    print(f"Found {len(problems)} problematic dimension(s) (std < eps={eps:.0e}):\n")
    print(f"  {'feature key':<30} {'dim':>4}  {'name':<30}  {'current std':>14}  {'effective amp':>14}")
    print(f"  {'-' * 30}  {'-' * 4}  {'-' * 30}  {'-' * 14}  {'-' * 14}")
    for key, i, name, v in problems:
        amp = 1.0 / max(v, eps)
        print(f"  {key:<30}  {i:>4}  {name:<30}  {v:>14.3e}  {amp:>14.3e}x")

    if not fix:
        print(f"\nRun with --fix to apply a std floor of {floor} to affected dimensions.")
        return

    # Backup
    bak_path = sf_path.with_suffix(".safetensors.bak")
    if bak_path.exists():
        print(f"\nBackup already exists: {bak_path.name} (skipping backup)")
    else:
        shutil.copy2(sf_path, bak_path)
        print(f"\nBacked up to: {bak_path.name}")

    # Apply floor
    print(f"\nApplying std floor = {floor} to affected dimensions:")
    for key, i, name, old_v in problems:
        std_key = f"{key}.std"
        tensors[std_key][i] = torch.tensor(floor, dtype=tensors[std_key].dtype)
        print(f"  {key}.std[{i}] ({name}): {old_v:.3e} -> {floor:.3e}")

    save_file(tensors, sf_path)
    print(f"\nSaved: {sf_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Check and fix near-zero std in model normalizer.")
    parser.add_argument("model_dir", type=Path, help="Path to pretrained_model directory")
    parser.add_argument("--fix", action="store_true", help="Apply the fix in-place")
    parser.add_argument("--floor", type=float, default=0.01, help="Minimum std floor (default: 0.01)")
    parser.add_argument("--eps", type=float, default=1e-8, help="Normalization eps (default: 1e-8)")
    args = parser.parse_args()

    if not args.model_dir.exists():
        print(f"ERROR: path does not exist: {args.model_dir}")
        return

    check_and_fix(args.model_dir, fix=args.fix, floor=args.floor, eps=args.eps)


if __name__ == "__main__":
    main()
