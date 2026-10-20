"""Measuring how well risk scores rank, how honest their probabilities are, and for whom.

Precision-recall leads because fewer than one inspection in five is at risk.
Precision and recall at a fixed capacity translate scores into the decision
Ofsted actually faces: which share of schools to look at first. Bootstrap
intervals show how much of any difference between models is noise, and the
fairness audit reports error rates by deprivation quintile.
"""

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def flag_top(scores: np.ndarray, share: float) -> np.ndarray:
    """Flag the highest-scoring share of rows, as a scheduler with fixed capacity would.

    Args:
        scores: Risk scores.
        share: Proportion of rows that can be flagged.

    Returns:
        A boolean array marking the flagged rows. Ties at the cut-off are
        broken by position so exactly the requested number is flagged.
    """
    count = int(round(share * len(scores)))
    flagged = np.zeros(len(scores), dtype=bool)
    flagged[np.argsort(-scores, kind="stable")[:count]] = True
    return flagged


def summarise(labels: np.ndarray, scores: np.ndarray, capacity_shares: Sequence[float]) -> Dict[str, float]:
    """Compute the headline metrics for one model on one set of rows.

    Args:
        labels: True labels, 1 for at risk.
        scores: Predicted probabilities.
        capacity_shares: Shares of schools that could be flagged.

    Returns:
        PR-AUC, ROC-AUC, Brier score, mean predicted and observed rates, and
        precision and recall at each capacity.
    """
    result = {
        "pr_auc": average_precision_score(labels, scores),
        "roc_auc": roc_auc_score(labels, scores),
        "brier": brier_score_loss(labels, scores),
        "mean_predicted": float(np.mean(scores)),
        "observed_rate": float(np.mean(labels)),
    }
    for share in capacity_shares:
        flagged = flag_top(scores, share)
        percent = int(round(share * 100))
        result[f"precision_at_{percent}"] = float(labels[flagged].mean())
        result[f"recall_at_{percent}"] = float(labels[flagged].sum() / max(labels.sum(), 1))
    return result


def bootstrap_intervals(
    labels: np.ndarray, scores: Dict[str, np.ndarray], baseline: str, resamples: int, seed: int
) -> pd.DataFrame:
    """Bootstrap 95% intervals for PR-AUC and for each model's gain over a baseline.

    The same resampled rows are used for every model, so the difference
    between two models is measured on identical data each time.

    Args:
        labels: True labels.
        scores: Predicted probabilities by model name.
        baseline: Model whose PR-AUC the others are compared against.
        resamples: Number of bootstrap samples.
        seed: Random seed.

    Returns:
        One row per model with the PR-AUC interval, the mean gain over the
        baseline, its interval, and the share of samples with a positive gain.
    """
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in scores}
    rows = len(labels)
    for _ in range(resamples):
        sample = rng.integers(0, rows, rows)
        if labels[sample].min() == labels[sample].max():
            continue
        for name, values in scores.items():
            draws[name].append(average_precision_score(labels[sample], values[sample]))
    base = np.array(draws[baseline])
    summary = []
    for name, values in draws.items():
        values = np.array(values)
        gain = values - base
        summary.append(
            {
                "model": name,
                "pr_auc_low": np.percentile(values, 2.5),
                "pr_auc_high": np.percentile(values, 97.5),
                "gain_vs_baseline": gain.mean(),
                "gain_low": np.percentile(gain, 2.5),
                "gain_high": np.percentile(gain, 97.5),
                "share_gain_positive": float((gain > 0).mean()),
            }
        )
    return pd.DataFrame(summary)


def calibration_table(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Compare predicted and observed rates within equal-sized score bins.

    Args:
        labels: True labels.
        scores: Predicted probabilities.
        bins: Number of quantile bins.

    Returns:
        One row per bin with its size, mean predicted probability and
        observed at-risk rate.
    """
    frame = pd.DataFrame({"label": labels, "score": scores})
    frame["bin"] = pd.qcut(frame["score"].rank(method="first"), bins, labels=False)
    return (
        frame.groupby("bin")
        .agg(rows=("label", "size"), mean_predicted=("score", "mean"), observed_rate=("label", "mean"))
        .reset_index()
    )


def fairness_table(labels: np.ndarray, scores: np.ndarray, groups: pd.Series, share: float) -> pd.DataFrame:
    """Report flag rates and error rates by group at a fixed capacity.

    Args:
        labels: True labels.
        scores: Predicted probabilities.
        groups: Group of each row, such as deprivation quintile.
        share: Share of all rows flagged, applied across groups together.

    Returns:
        One row per group with its size, base rate, flag rate, recall
        (true positive rate), false positive rate and precision.
    """
    frame = pd.DataFrame(
        {"group": groups.to_numpy(), "label": labels, "flagged": flag_top(scores, share)}
    )
    rows = []
    for group, part in frame.groupby("group", dropna=False):
        positives, negatives = part["label"] == 1, part["label"] == 0
        rows.append(
            {
                "group": group,
                "rows": len(part),
                "base_rate": part["label"].mean(),
                "flag_rate": part["flagged"].mean(),
                "recall": part.loc[positives, "flagged"].mean() if positives.any() else np.nan,
                "false_positive_rate": part.loc[negatives, "flagged"].mean() if negatives.any() else np.nan,
                "precision": part.loc[part["flagged"], "label"].mean() if part["flagged"].any() else np.nan,
            }
        )
    return pd.DataFrame(rows)
