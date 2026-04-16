"""Evaluate failure-detection methods using functional conformal prediction.

Reads pre-computed per-step anomaly scores (produced by
``compute_comparison.py``) together with a per-episode success/failure label
file, then applies the FAIL-Detect evaluation framework (RSS 2025) to compute:

    TPR, TNR, Accuracy, Weighted Accuracy, Detection Time (± SE)

for every method.  Results are saved as a pandas DataFrame pickle and printed
as a table.

Label file format
-----------------
A JSON file with **integer episode indices as string keys** and boolean/int
success flags as values::

    {
        "0": 1,
        "1": 0,
        "2": 1,
        ...
    }

1 = success, 0 = failure.  By default the script looks for
``<scores_dir>/episode_labels.json``.  Pass ``--labels`` to override.

Score files
-----------
Each method must have a ``<method>_scores.jsonl`` file under ``<scores_dir>``.
Each line::

    {"global_step": 123, "episode": 2, "step_in_episode": 5, "score": 0.043}

Usage
-----
::

    python tools/comparison/evaluate_comparison.py \\
        --repo_id eval/eval_pick_and_place_act \\
        --cal_repo_id eval/eval_pick_and_place_calibrate

Calibration split
-----------------
The first ``num_train + num_cal`` **successful** episodes are used to fit the
conformal band (``num_train`` for the band centre, ``num_cal`` for the
quantile).  All remaining episodes (both success and failure) form the test
set used to compute detection metrics.

Method-specific evaluation
--------------------------
* **All methods**: constant scalar threshold — conformal quantile over
  per-episode maximum calibration scores (``_detect_max_threshold``).
"""

from __future__ import annotations

import argparse
import json
import pickle  # nosec B403
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_episode_labels(path: Path) -> dict[int, int]:
    """Load ``{episode_idx: success_flag}`` from a JSON file.

    Keys may be strings (JSON requirement); they are cast to int.
    Values are cast to int (0 or 1).
    """
    with path.open() as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def load_scores_by_episode(scores_path: Path) -> dict[int, list[float]]:
    """Load a ``*_scores.jsonl`` file and group scores by episode.

    Returns:
        ``{episode_idx: [score_t0, score_t1, ...]}`` sorted by
        ``step_in_episode``.
    """
    if not scores_path.exists():
        return {}
    per_episode: dict[int, list[tuple[int, float]]] = {}
    with scores_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                ep = int(row["episode"])
                step = int(row["step_in_episode"])
                score = float(row["score"])
                per_episode.setdefault(ep, []).append((step, score))
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
    return {ep: [s for _, s in sorted(steps)] for ep, steps in per_episode.items()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute TPR, TNR, Accuracy, Weighted Accuracy.

    Args:
        y_true: 1-D int array; 1 = failure, 0 = success.
        y_pred: 1-D int array; 1 = predicted failure, 0 = predicted success.

    Returns:
        Dict with keys: TPR, TNR, Accuracy, Accuracy_weighted.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tpr + tnr) / 2.0
    # Weighted: weight by class frequency in y_true
    failure_rate = float(y_true.sum()) / len(y_true)
    success_rate = 1.0 - failure_rate
    accuracy_weighted = success_rate * tpr + failure_rate * tnr
    return {
        "TPR": tpr,
        "TNR": tnr,
        "Accuracy": accuracy,
        "Accuracy_weighted": accuracy_weighted,
    }


# ---------------------------------------------------------------------------
# Scalar max-score detection
# ---------------------------------------------------------------------------


def _episode_score(seq: list[float], tail_steps: int) -> float:
    """Aggregate a per-step score sequence into a single episode score.

    ``tail_steps=0``  → per-episode **max** (original behaviour).
    ``tail_steps>0``  → mean of the **last K steps**.  Failures tend to end in
    a persistently anomalous state (robot stuck/dropped object), so the tail
    mean separates them from successes whose anomalies are transient
    mid-trajectory peaks that return to normal by the end.
    """
    if not seq:
        return 0.0
    if tail_steps > 0:
        return float(np.mean(seq[-tail_steps:]))
    return float(max(seq))


def _detect_max_threshold(
    train_scores: list[list[float]],
    test_scores: list[list[float]],
    test_labels: list[int],
    alpha: float,
    threshold: float | None = None,
    tail_steps: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Scalar threshold detection using per-episode CP formula.

    When ``tail_steps=0`` (default): calibration score = per-episode **max**,
    and failure is flagged at the first timestep where ``score > threshold``.

    When ``tail_steps>0``: calibration score = mean of last ``tail_steps``
    steps, and a test episode is flagged if its tail mean exceeds the
    threshold.  Failure episodes typically end in a persistently anomalous
    state, whereas successful episodes return to normal by the end — so the
    tail mean gives much better separation than the max for RND / logpZO.

    threshold = CP quantile over per-episode calibration scores:
        n = number of calibration episodes
        q_level = min(ceil((n+1)*(1-alpha)) / n, 1.0)
        threshold = quantile(episode_scores, q_level)

    Args:
        threshold: If provided, skip computation and use this value directly.
        tail_steps: Number of terminal steps to average for episode score.
                    0 = use max (original behaviour).
    """
    if threshold is None:
        cal_scores = [_episode_score(seq, tail_steps) for seq in train_scores if seq]
        if cal_scores:
            n = len(cal_scores)
            q_level = min(float(np.ceil((n + 1) * (1.0 - alpha)) / n), 1.0)
            threshold = float(np.quantile(cal_scores, q_level))
        else:
            threshold = 0.0

    y_true: list[int] = []
    y_pred: list[int] = []
    first_steps: list[int] = []

    for seq, label in zip(test_scores, test_labels, strict=False):
        predicted = 0
        if tail_steps > 0:
            # Episode-level classification: tail mean vs threshold
            if _episode_score(seq, tail_steps) > threshold:
                predicted = 1
                if label == 0:  # true positive
                    first_steps.append(max(0, len(seq) - tail_steps))
        else:
            # Step-level detection: first step exceeding threshold
            for t, score in enumerate(seq):
                if score > threshold:
                    predicted = 1
                    if label == 0:  # true positive
                        first_steps.append(t)
                    break
        y_true.append(1 - label)
        y_pred.append(predicted)

    return np.array(y_true), np.array(y_pred), first_steps


# Methods whose detection threshold is calibrated on per-episode MIN scores
# (i.e., where a *low* per-step score is anomalous and scores are NOT
# already negated by the detector).  Currently unused — ActionEntropyDetector
# negates in update() so the standard max-threshold path applies.
_INVERTED_METHODS: frozenset[str] = frozenset()


def _detect_min_threshold(
    train_scores: list[list[float]],
    test_scores: list[list[float]],
    test_labels: list[int],
    alpha: float,
    threshold: float | None = None,
    tail_steps: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Scalar threshold detection for inverted-score methods.

    Scores are stored **negated** so that failure → high (less-negative) score.
    Calibration uses per-episode **min** (most negative = most normal) to
    mirror how ``ActionEntropyDetector.calibration_score()`` works.

    threshold = CP quantile over per-episode **min** calibration scores:
        n = number of calibration episodes
        q_level = min(ceil((n+1)*(1-alpha)) / n, 1.0)
        threshold = quantile(episode_min_scores, q_level)

    Failure is flagged at the first timestep where ``score > threshold``.

    Args:
        threshold: If provided, skip computation and use this value directly.
        tail_steps: Unused for min-threshold methods; kept for API parity.
    """
    _ = tail_steps
    if threshold is None:
        min_scores = [min(seq) for seq in train_scores if seq]
        if min_scores:
            n = len(min_scores)
            q_level = min(float(np.ceil((n + 1) * (1.0 - alpha)) / n), 1.0)
            threshold = float(np.quantile(min_scores, q_level))
        else:
            threshold = 0.0

    y_true: list[int] = []
    y_pred: list[int] = []
    first_steps: list[int] = []

    for seq, label in zip(test_scores, test_labels, strict=False):
        predicted = 0
        for t, score in enumerate(seq):
            if score > threshold:
                predicted = 1
                if label == 0:  # true positive
                    first_steps.append(t)
                break
        y_true.append(1 - label)
        y_pred.append(predicted)

    return np.array(y_true), np.array(y_pred), first_steps


# ---------------------------------------------------------------------------
# Per-method evaluation
# ---------------------------------------------------------------------------


def evaluate_method(
    method: str,
    all_scores: dict[int, list[float]],
    all_labels: dict[int, int],
    num_train: int,
    num_cal: int,
    alpha: float,
    debug: bool = False,
    cal_scores_by_episode: dict[int, list[float]] | None = None,
    saved_threshold: float | None = None,
    tail_steps: int = 0,
) -> dict[str, float] | None:
    """Evaluate one method.

    Returns a dict with keys:
        TPR, TNR, Accuracy, Accuracy_weighted,
        Detection_time, Detection_time_SE

    or ``None`` if there is insufficient data.

    Split logic — two modes
    -----------------------
    **Separate calibration dataset** (``cal_scores_by_episode`` provided):
        * All episodes in ``cal_scores_by_episode`` are used for train+cal.
        * First ``num_train`` cal episodes → band centre fit.
        * Next ``num_cal`` cal episodes → conformal quantile.
        * ALL episodes in ``all_scores`` that have labels → test set.
        This is the correct mode when you have a dedicated calibration dataset.

    **Single dataset** (``cal_scores_by_episode`` is None):
        * Successful episodes in ``all_scores`` are split internally.
        * First ``num_train`` → train, next ``num_cal`` → cal, remainder → test.
        * All failure episodes go into the test set.
    """
    if cal_scores_by_episode is not None:
        # --- Two-dataset mode: calibrate → train+cal, eval → test ---
        cal_seqs = list(cal_scores_by_episode.values())
        n_cal_eps = len(cal_seqs)
        if n_cal_eps < num_train + num_cal:
            _nt = n_cal_eps // 2
            _nc = n_cal_eps - _nt
            print(
                f"[{method}] Only {n_cal_eps} cal episodes; "
                f"using {_nt} train + {_nc} cal (requested {num_train}+{num_cal})."
            )
        else:
            _nt, _nc = num_train, num_cal

        train_scores = cal_seqs[:_nt]
        cal_scores = cal_seqs[_nt : _nt + _nc]

        common_eps = sorted(set(all_scores) & set(all_labels))
        if not common_eps:
            print(f"[{method}] No episodes with both scores and labels — skipping.")
            return None

        test_eps = common_eps
        test_scores = [all_scores[ep] for ep in test_eps]
        test_labels = [all_labels[ep] for ep in test_eps]

    else:
        # --- Single-dataset mode: split success episodes internally ---
        common_eps = sorted(set(all_scores) & set(all_labels))
        if not common_eps:
            print(f"[{method}] No episodes with both scores and labels — skipping.")
            return None

        success_eps = [ep for ep in common_eps if all_labels[ep] == 1]
        failure_eps = [ep for ep in common_eps if all_labels[ep] == 0]

        n_success = len(success_eps)
        n_needed = num_train + num_cal
        if n_success < n_needed + 1:
            print(
                f"[{method}] Need {n_needed + 1} successful episodes for calibration "
                f"+ at least 1 test, but only {n_success} available — skipping.\n"
                f"         Tip: pass --cal_repo_id pointing to a dedicated "
                f"calibration dataset to avoid this limit."
            )
            return None

        _nt, _nc = num_train, num_cal
        train_scores = [all_scores[ep] for ep in success_eps[:_nt]]
        cal_scores = [all_scores[ep] for ep in success_eps[_nt : _nt + _nc]]

        test_eps = success_eps[_nt + _nc :] + failure_eps
        test_scores = [all_scores[ep] for ep in test_eps]
        test_labels = [all_labels[ep] for ep in test_eps]

    n_test_fail = sum(1 for lbl in test_labels if lbl == 0)
    n_test_succ = sum(1 for lbl in test_labels if lbl == 1)
    print(
        f"[{method}] train={len(train_scores)} cal={len(cal_scores)} "
        f"test={len(test_eps)} (success={n_test_succ}, failure={n_test_fail})"
    )

    if n_test_fail == 0:
        print(f"[{method}] No failure episodes in test set — TPR undefined, skipping.")
        return None

    # Use the pre-computed threshold from {method}_threshold.json when available;
    # fall back to computing from the combined train+cal calibration pool otherwise.
    # When tail_steps>0, keep the saved max-based threshold — it is always ≥ any tail-mean
    # of a success episode, so it correctly covers the tail-mean distribution of successes
    # without being inflated by transient mid-trajectory anomaly peaks.
    _detect_fn = _detect_min_threshold if method in _INVERTED_METHODS else _detect_max_threshold
    _tail = 0 if method in _INVERTED_METHODS else tail_steps
    y_true, y_pred, first_steps = _detect_fn(
        train_scores + cal_scores,
        test_scores,
        test_labels,
        alpha=alpha,
        threshold=saved_threshold,
        tail_steps=_tail,
    )

    metrics = compute_metrics(y_true, y_pred)
    metrics["Detection_time"] = float(np.mean(first_steps)) if first_steps else float("nan")
    metrics["Detection_time_SE"] = (
        float(np.std(first_steps) / np.sqrt(len(first_steps))) if len(first_steps) > 1 else 0.0
    )

    if debug:
        _print_episode_debug(method, test_eps, y_true, y_pred, test_scores)

    return metrics


def _print_episode_debug(
    method: str,
    test_eps: list[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_scores: list[list[float]],
) -> None:
    """Print per-episode prediction table for debugging misclassifications."""
    _outcome = {
        (1, 1): "TP",
        (0, 0): "TN",
        (1, 0): "FN",  # failure missed
        (0, 1): "FP",  # success flagged as failure
    }
    header = f"{'ep':>5}  {'true':>8}  {'pred':>8}  {'result':>6}  {'max_score':>12}  {'n_steps':>7}"
    print(f"\n[{method}] Per-episode debug:")
    print(header)
    print("-" * len(header))
    for ep, yt, yp, scores in zip(test_eps, y_true, y_pred, test_scores, strict=False):
        true_label = "failure" if yt == 1 else "success"
        pred_label = "failure" if yp == 1 else "success"
        outcome = _outcome.get((int(yt), int(yp)), "??")
        max_s = max(scores) if scores else float("nan")
        marker = " ←" if yt != yp else ""
        print(
            f"{ep:>5}  {true_label:>8}  {pred_label:>8}  {outcome:>6}  {max_s:>12.4f}  {len(scores):>7}{marker}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_dataset_root(repo_id: str, root: str | None) -> Path:
    """Return the local root of a LeRobot dataset."""
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root)
    return Path(ds.root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate failure-detection methods with functional CP calibration."
    )
    parser.add_argument(
        "--repo_id",
        required=True,
        help="Dataset repo_id of the eval dataset "
        "(e.g. eval/eval_pick_and_place_act).  "
        "Scores are read from <dataset_root>/comparison/, "
        "labels from <dataset_root>/comparison/episode_labels.json, "
        "results written to <dataset_root>/comparison/.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Dataset root (defaults to ~/.cache/huggingface/lerobot).",
    )
    parser.add_argument(
        "--cal_repo_id",
        default=None,
        help="repo_id of the calibration dataset "
        "(e.g. eval/eval_pick_and_place_calibrate).  "
        "When provided, all episodes in <cal_root>/comparison/ are used "
        "for train+cal and all --repo_id episodes become the test set.",
    )
    parser.add_argument(
        "--cal_root",
        default=None,
        help="Root for the calibration dataset (defaults to --root).",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Override path to episode_labels.json (default: <dataset_root>/comparison/episode_labels.json).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to evaluate.  Auto-detected from *_scores.jsonl files if omitted.",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=100,
        help="Successful episodes used to fit the band centre (default: 100).",
    )
    parser.add_argument(
        "--num_cal",
        type=int,
        default=200,
        help="Successful episodes used for the conformal quantile (default: 200).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.025,
        help="Significance level for CP band (default: 0.025).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to write results.pkl and results.csv.  Defaults to <dataset_root>/comparison/.",
    )
    parser.add_argument(
        "--tail_steps",
        type=int,
        default=0,
        help="If >0, use mean of last N steps as episode score instead of max. "
        "Useful for methods (RND, logpZO) where failures persist until episode end "
        "but successes have only transient mid-trajectory anomalies (default: 0 = max).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-episode prediction table (episode index, true/pred label, "
        "max score, TP/TN/FP/FN).  Misclassified episodes are marked with ←.",
    )
    args = parser.parse_args()

    # --- Resolve dataset roots → comparison directories ---
    dataset_root = _resolve_dataset_root(args.repo_id, args.root)
    scores_dir = dataset_root / "comparison"
    labels_path = Path(args.labels) if args.labels else scores_dir / "episode_labels.json"
    output_dir = Path(args.output_dir) if args.output_dir else scores_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Dataset root : {dataset_root}")
    print(f"[INFO] Scores dir   : {scores_dir}")

    # --- Labels ---
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Episode labels file not found: {labels_path}\n"
            "Create a JSON file mapping episode index → 1 (success) / 0 (failure)."
        )
    all_labels = load_episode_labels(labels_path)
    n_success = sum(v == 1 for v in all_labels.values())
    n_fail = sum(v == 0 for v in all_labels.values())
    print(f"[INFO] Labels loaded: {len(all_labels)} episodes (success={n_success}, failure={n_fail})")

    # --- Discover methods ---
    methods = args.methods or sorted(p.stem.replace("_scores", "") for p in scores_dir.glob("*_scores.jsonl"))
    if not methods:
        raise RuntimeError(f"No *_scores.jsonl files found in {scores_dir}.")
    print(f"[INFO] Methods: {methods}")

    # --- Load eval scores ---
    scores_by_method: dict[str, dict[int, list[float]]] = {}
    for method in methods:
        path = scores_dir / f"{method}_scores.jsonl"
        if not path.exists():
            print(f"[WARN] {path} not found — skipping {method}.")
            continue
        scores_by_method[method] = load_scores_by_episode(path)
        n_eps = len(scores_by_method[method])
        print(f"[INFO] {method}: {n_eps} episodes loaded from {path.name}")

    if not scores_by_method:
        raise RuntimeError("No score files could be loaded.")

    # --- Load calibration scores (separate dataset) ---
    cal_scores_by_method: dict[str, dict[int, list[float]]] | None = None
    if args.cal_repo_id:
        cal_root = _resolve_dataset_root(args.cal_repo_id, args.cal_root or args.root)
        cal_dir = cal_root / "comparison"
        print(f"[INFO] Cal scores dir: {cal_dir}")
        cal_scores_by_method = {}
        for method in scores_by_method:
            cal_path = cal_dir / f"{method}_scores.jsonl"
            if not cal_path.exists():
                print(
                    f"[WARN] Cal scores not found: {cal_path} — will fall back to single-dataset split for {method}."
                )
                continue
            cal_scores_by_method[method] = load_scores_by_episode(cal_path)
            print(f"[INFO] {method}: {len(cal_scores_by_method[method])} cal episodes from {cal_path.name}")

    # --- Load saved scalar thresholds ---
    # {method}_threshold.json is written by run_calibration in compute_comparison.py.
    # It may have been produced with different smoothing/quantile parameters than
    # the calibration score JSONL, so we prefer it when available.
    saved_thresholds: dict[str, float] = {}
    for method in scores_by_method:
        threshold_path = scores_dir / f"{method}_threshold.json"
        if threshold_path.exists():
            try:
                with threshold_path.open() as _f:
                    threshold_data = json.load(_f)
                tval = threshold_data.get("threshold")
                if isinstance(tval, (int, float)):
                    saved_thresholds[method] = float(tval)
                    print(f"[INFO] {method}: loaded saved threshold={tval:.6f} from {threshold_path.name}")
            except Exception as e:
                print(f"[WARN] Could not load {threshold_path}: {e}")

    # --- Evaluate ---
    metric_names = [
        "TPR",
        "TNR",
        "Accuracy",
        "Accuracy_weighted",
        "Detection_time",
        "Detection_time_SE",
    ]
    records: dict[str, dict[str, float]] = {}

    for method, scores in scores_by_method.items():
        cal_ep_scores = (cal_scores_by_method or {}).get(method)
        result = evaluate_method(
            method=method,
            all_scores=scores,
            all_labels=all_labels,
            num_train=args.num_train,
            num_cal=args.num_cal,
            alpha=args.alpha,
            debug=args.debug,
            cal_scores_by_episode=cal_ep_scores,
            saved_threshold=saved_thresholds.get(method),
            tail_steps=args.tail_steps,
        )
        if result is not None:
            records[method] = result

    if not records:
        print("[ERROR] No methods produced valid results.")
        return

    # --- Build DataFrame ---
    df = pd.DataFrame(records, index=metric_names).T  # rows=methods, cols=metrics
    df.index.name = "Method"

    print("\n" + "=" * 60)
    print(df.round(4).to_string())
    print("=" * 60)

    # --- Save ---
    pkl_path = output_dir / "results.pkl"
    csv_path = output_dir / "results.csv"
    with pkl_path.open("wb") as f:
        pickle.dump(df, f)
    df.to_csv(csv_path)
    print(f"\n[INFO] Saved → {pkl_path}")
    print(f"[INFO] Saved → {csv_path}")


if __name__ == "__main__":
    main()
