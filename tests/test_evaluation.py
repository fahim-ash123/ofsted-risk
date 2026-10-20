"""Tests for capacity flagging, metrics, bootstrap intervals, calibration and fairness."""

import numpy as np
import pandas as pd

from ofsted_risk.evaluation import bootstrap_intervals, calibration_table, fairness_table, flag_top, summarise


def test_flag_top_flags_exactly_the_requested_share():
    """A 20% capacity flags exactly a fifth of rows, highest scores first."""
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.05])

    flagged = flag_top(scores, 0.2)

    assert flagged.sum() == 2
    assert flagged[[0, 2]].all()


def test_summarise_reports_precision_and_recall_at_capacity():
    """Precision and recall at capacity match a hand count."""
    labels = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.05])

    result = summarise(labels, scores, (0.2,))

    assert result["precision_at_20"] == 1.0
    assert abs(result["recall_at_20"] - 2 / 3) < 1e-9
    assert abs(result["observed_rate"] - 0.3) < 1e-9


def test_bootstrap_gain_of_the_baseline_is_zero():
    """The baseline's gain over itself is zero, and a better model gains."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 400)
    scores = {"weak": rng.random(400), "strong": labels * 0.5 + rng.random(400) * 0.5}

    table = bootstrap_intervals(labels, scores, "weak", resamples=100, seed=1).set_index("model")

    assert table.loc["weak", "gain_vs_baseline"] == 0
    assert table.loc["strong", "gain_low"] > 0
    assert table.loc["strong", "share_gain_positive"] == 1.0


def test_calibration_table_has_equal_bins_covering_every_row():
    """Quantile bins split the rows evenly."""
    rng = np.random.default_rng(0)

    table = calibration_table(rng.integers(0, 2, 100), rng.random(100), bins=10)

    assert table["rows"].tolist() == [10] * 10


def test_fairness_table_matches_a_hand_count():
    """Group recall, false positive rate and precision match counted values."""
    labels = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.1, 0.7, 0.2, 0.1, 0.1])
    groups = pd.Series([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

    table = fairness_table(labels, scores, groups, 0.3).set_index("group")

    assert table.loc[1, "recall"] == 0.5
    assert table.loc[1, "false_positive_rate"] == 1 / 3
    assert table.loc[2, "recall"] == 1.0
    assert table.loc[2, "precision"] == 1.0
