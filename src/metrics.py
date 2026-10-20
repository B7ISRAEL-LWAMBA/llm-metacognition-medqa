"""Calibration and metacognition metrics.

All functions take `correct` (0/1 or bool array) and `confidence` (0-1 array).
Parse-error rows must be dropped before calling them.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RAW_DIR = Path(__file__).resolve().parent.parent / "results" / "raw"
N_BINS = 10


def accuracy(correct, confidence=None) -> float:
    return float(np.mean(correct))


def ece(correct, confidence, n_bins: int = N_BINS) -> float:
    """Expected calibration error: weighted mean |accuracy - confidence| over
    equal-width bins. Confidence 1.0 falls in the top bin."""
    correct, confidence = np.asarray(correct, float), np.asarray(confidence, float)
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def brier(correct, confidence) -> float:
    return float(np.mean((np.asarray(confidence, float) - np.asarray(correct, float)) ** 2))


def auroc(correct, confidence) -> float:
    """Type-2 AUROC (metacognitive sensitivity): the probability that a random
    right answer got higher confidence than a random wrong one. NaN if every
    answer was right, or every answer was wrong."""
    correct = np.asarray(correct, int)
    if correct.min() == correct.max():
        return float("nan")
    return float(roc_auc_score(correct, confidence))


def reliability_bins(correct, confidence, n_bins: int = N_BINS) -> pd.DataFrame:
    """Per-bin mean confidence, accuracy and count, for the reliability diagram."""
    correct, confidence = np.asarray(correct, float), np.asarray(confidence, float)
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            rows.append({"bin": b, "confidence": confidence[mask].mean(),
                         "accuracy": correct[mask].mean(), "n": int(mask.sum())})
    return pd.DataFrame(rows)


def risk_coverage(correct, confidence) -> pd.DataFrame:
    """Accuracy on the answers kept when everything below a confidence
    threshold is deferred. One point per distinct confidence value, so tied
    confidences are kept or deferred together (important for condition B,
    which only has 5 possible values)."""
    correct, confidence = np.asarray(correct, float), np.asarray(confidence, float)
    rows = []
    for t in np.unique(confidence)[::-1]:
        kept = confidence >= t
        rows.append({"threshold": t, "coverage": kept.mean(), "accuracy": correct[kept].mean()})
    return pd.DataFrame(rows)


def bootstrap_ci(metric, correct, confidence, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval, resampling questions with replacement."""
    correct, confidence = np.asarray(correct), np.asarray(confidence)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(correct), len(correct))
        stats.append(metric(correct[idx], confidence[idx]))
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap_diff(metric, a: pd.DataFrame, b: pd.DataFrame, n: int = 1000, seed: int = 0) -> dict:
    """Difference metric(a) - metric(b) on the questions both sets answered,
    with a 95% interval from resampling the same questions for both. If the
    interval excludes 0, the difference is unlikely to be noise."""
    merged = a.merge(b, on="question_id", suffixes=("_a", "_b"))
    ca, fa = merged["is_correct_a"].to_numpy(int), merged["confidence_a"].to_numpy(float)
    cb, fb = merged["is_correct_b"].to_numpy(int), merged["confidence_b"].to_numpy(float)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(merged), len(merged))
        diffs.append(metric(ca[idx], fa[idx]) - metric(cb[idx], fb[idx]))
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {"diff": metric(ca, fa) - metric(cb, fb), "ci_low": float(lo), "ci_high": float(hi),
            "n_questions": len(merged)}


def load_results(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """All raw JSONL files in one DataFrame (parse errors included)."""
    rows = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        # Group by the configured model name from the file name, since the API
        # may report a dated snapshot name in the `model` field.
        model_key = path.stem.split("__")[0]
        with open(path, encoding="utf-8") as f:
            rows.extend({**json.loads(line), "model_key": model_key} for line in f if line.strip())
    return pd.DataFrame(rows)


def results_table(df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """One row per model and condition with all headline metrics."""
    rows = []
    for (model, cond), g in df.groupby(["model_key", "condition"]):
        ok = g[~g["parse_error"].astype(bool)]
        c, f = ok["is_correct"].to_numpy(int), ok["confidence"].to_numpy(float)
        acc_lo, acc_hi = bootstrap_ci(accuracy, c, f, n_boot)
        auc_lo, auc_hi = bootstrap_ci(auroc, c, f, n_boot)
        row = {
            "model": model, "condition": cond, "n": len(ok), "parse_errors": len(g) - len(ok),
            "accuracy": accuracy(c), "acc_ci": f"[{acc_lo:.3f}, {acc_hi:.3f}]",
            "ece": ece(c, f), "brier": brier(c, f),
            "auroc": auroc(c, f), "auroc_ci": f"[{auc_lo:.3f}, {auc_hi:.3f}]",
            "mean_confidence": f.mean(),
        }
        if cond == "C":
            initial_right = ok["initial_answer"] == ok["correct_answer"]
            row["critic_fixed"] = int((~initial_right & ok["is_correct"].astype(bool)).sum())
            row["critic_broke"] = int((initial_right & ~ok["is_correct"].astype(bool)).sum())
        rows.append(row)
    return pd.DataFrame(rows)
