"""
benchmark/metrics.py — Evaluation metrics for confidence estimation.

Metrics:
  - AUROC: discrimination quality (separating correct from incorrect)
  - ECE: calibration (confidence matches empirical accuracy)
  - Brier Score: combined calibration + discrimination
  - Risk-Coverage Curve: error rate as a function of coverage (fraction kept)
  - Accuracy@K%: accuracy when keeping top-K% most confident predictions
  - Spearman ρ: rank correlation between confidence and correctness
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MetricSummary:
    """All metrics for a single (method, dataset) pair."""
    method: str
    dataset: str
    n_examples: int
    auroc: float
    ece: float
    brier: float
    accuracy_at_90: float
    accuracy_at_80: float
    accuracy_at_70: float
    accuracy_at_50: float
    risk_coverage_auc: float
    spearman_rho: float


def compute_auroc(confidences: list[float], is_correct: list[bool]) -> float:
    """
    AUROC via the trapezoidal rule over the ROC curve.

    Returns float in [0, 1]. 0.5 = random, 1.0 = perfect.
    """
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore
        return float(roc_auc_score(is_correct, confidences))
    except ImportError:
        pass
    except ValueError:
        # Only one class present
        return 0.5

    # Fallback: manual AUROC via sorted pairs
    pairs = sorted(zip(confidences, is_correct), reverse=True)
    n_pos = sum(is_correct)
    n_neg = len(is_correct) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = fp = 0
    prev_tp = prev_fp = 0
    auc = 0.0
    prev_conf = None
    for conf, label in pairs:
        if conf != prev_conf and prev_conf is not None:
            auc += _trapezoid(fp / n_neg, tp / n_pos, prev_fp / n_neg, prev_tp / n_pos)
            prev_fp, prev_tp = fp, tp
        if label:
            tp += 1
        else:
            fp += 1
        prev_conf = conf
    auc += _trapezoid(fp / n_neg, tp / n_pos, prev_fp / n_neg, prev_tp / n_pos)
    return float(auc)


def _trapezoid(x1: float, y1: float, x2: float, y2: float) -> float:
    return abs(x2 - x1) * (y1 + y2) / 2


def compute_ece(
    confidences: list[float],
    is_correct: list[bool],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error: mean |confidence − accuracy| across equal-width bins.

    Lower is better. < 0.05 is well-calibrated.
    """
    n = len(confidences)
    if n == 0:
        return 0.0

    bin_size = 1.0 / n_bins
    ece = 0.0
    for b in range(n_bins):
        lo = b * bin_size
        hi = lo + bin_size
        indices = [i for i, c in enumerate(confidences) if lo <= c < hi]
        if not indices:
            continue
        bin_conf = sum(confidences[i] for i in indices) / len(indices)
        bin_acc = sum(1 for i in indices if is_correct[i]) / len(indices)
        ece += (len(indices) / n) * abs(bin_conf - bin_acc)

    return float(ece)


def compute_brier(confidences: list[float], is_correct: list[bool]) -> float:
    """
    Brier Score: mean squared error between confidence and binary outcome.

    Lower is better. 0.25 = uninformative baseline (always predict 0.5).
    """
    try:
        from sklearn.metrics import brier_score_loss  # type: ignore
        return float(brier_score_loss(is_correct, confidences))
    except ImportError:
        pass

    n = len(confidences)
    if n == 0:
        return 0.0
    return sum((c - int(y)) ** 2 for c, y in zip(confidences, is_correct)) / n


def compute_risk_coverage(
    confidences: list[float],
    is_correct: list[bool],
    n_points: int = 20,
) -> dict:
    """
    Risk-Coverage curve data.

    At each coverage level (fraction of examples kept by confidence threshold),
    compute the error rate (1 − accuracy) of the kept subset.

    Returns:
        dict with keys:
          'coverage': list[float] — fraction of examples kept (descending threshold)
          'risk': list[float] — error rate at each coverage level
          'auc': float — area under the risk-coverage curve (lower is better)
    """
    n = len(confidences)
    if n == 0:
        return {"coverage": [], "risk": [], "auc": 1.0}

    # Sort by descending confidence
    order = sorted(range(n), key=lambda i: confidences[i], reverse=True)

    coverages: list[float] = []
    risks: list[float] = []

    step = max(1, n // n_points)
    for k in range(step, n + 1, step):
        kept = order[:k]
        coverage = k / n
        error_rate = 1.0 - sum(is_correct[i] for i in kept) / k
        coverages.append(coverage)
        risks.append(error_rate)

    # AUC via trapezoidal rule
    auc = 0.0
    for i in range(1, len(coverages)):
        auc += (coverages[i] - coverages[i - 1]) * (risks[i] + risks[i - 1]) / 2

    return {"coverage": coverages, "risk": risks, "auc": float(auc)}


def compute_accuracy_at_k(
    confidences: list[float],
    is_correct: list[bool],
    k: float = 0.9,
) -> float:
    """
    Accuracy when keeping only the top-K fraction of most confident predictions.

    Args:
        k: Fraction to keep, e.g. 0.9 = keep top 90%.

    Returns:
        Accuracy in [0, 1] on the kept subset.
    """
    n = len(confidences)
    if n == 0:
        return 0.0
    n_keep = max(1, int(n * k))
    order = sorted(range(n), key=lambda i: confidences[i], reverse=True)
    kept = order[:n_keep]
    return sum(is_correct[i] for i in kept) / n_keep


def compute_spearman(confidences: list[float], is_correct: list[bool]) -> float:
    """
    Spearman rank correlation between confidence and binary correctness.

    Returns float in [−1, 1]. 1.0 = perfect rank agreement.
    """
    try:
        from scipy.stats import spearmanr  # type: ignore
        rho, _ = spearmanr(confidences, [int(y) for y in is_correct])
        return float(rho) if not math.isnan(rho) else 0.0
    except ImportError:
        pass

    # Fallback: manual Spearman via rank differences
    n = len(confidences)
    if n < 2:
        return 0.0

    def _ranks(values):
        order = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        for rank, idx in enumerate(order):
            ranks[idx] = rank + 1
        return ranks

    rc = _ranks(confidences)
    ry = _ranks([float(y) for y in is_correct])
    d_sq = sum((rc[i] - ry[i]) ** 2 for i in range(n))
    return float(1.0 - 6 * d_sq / (n * (n ** 2 - 1)))


def compute_all_metrics(
    method: str,
    dataset: str,
    confidences: list[float],
    is_correct: list[bool],
) -> MetricSummary:
    """Compute all metrics for a (method, dataset) pair and return a summary."""
    rc = compute_risk_coverage(confidences, is_correct)
    return MetricSummary(
        method=method,
        dataset=dataset,
        n_examples=len(confidences),
        auroc=compute_auroc(confidences, is_correct),
        ece=compute_ece(confidences, is_correct),
        brier=compute_brier(confidences, is_correct),
        accuracy_at_90=compute_accuracy_at_k(confidences, is_correct, 0.90),
        accuracy_at_80=compute_accuracy_at_k(confidences, is_correct, 0.80),
        accuracy_at_70=compute_accuracy_at_k(confidences, is_correct, 0.70),
        accuracy_at_50=compute_accuracy_at_k(confidences, is_correct, 0.50),
        risk_coverage_auc=rc["auc"],
        spearman_rho=compute_spearman(confidences, is_correct),
    )
