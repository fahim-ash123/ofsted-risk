"""Tests for preparing features and describing text themes."""

import numpy as np
import pandas as pd

from ofsted_risk.features import CATEGORICAL_FEATURES
from ofsted_risk.modelling import TABULAR_NUMERIC, describe_themes, logit, prepare


def test_prepare_makes_missing_categories_and_text_explicit():
    """Missing categories become "missing", missing text becomes empty."""
    row = {column: np.nan for column in TABULAR_NUMERIC}
    row.update({column: np.nan for column in CATEGORICAL_FEATURES})
    row["improvement_text"] = np.nan
    table = pd.DataFrame([row, {**row, "ofsted_phase": "Primary", "improvement_text": "Improve reading."}])

    prepared = prepare(table)

    assert prepared["ofsted_phase"].tolist() == ["missing", "Primary"]
    assert prepared["improvement_text"].tolist() == ["", "Improve reading."]
    assert prepared["census_pct_fsm"].dtype == float


def test_text_length_is_not_a_tabular_feature():
    """Report length describes the text, so it stays out of the context models."""
    assert "text_characters" not in TABULAR_NUMERIC


def test_logit_is_finite_at_the_extremes():
    """Probabilities of exactly 0 or 1 give finite log-odds."""
    assert np.isfinite(logit(np.array([0.0, 0.5, 1.0]))).all()


def test_describe_themes_covers_every_row_once():
    """Theme sizes add up to the number of texts."""
    texts = ["improve reading"] * 10 + ["strengthen leadership"] * 10
    vectors = np.array([[1.0, 0.0]] * 10 + [[0.0, 1.0]] * 10)
    labels = np.array([0] * 10 + [1] * 10)

    _, themes = describe_themes(texts, vectors, labels, count=2, seed=0)

    assert themes["rows"].sum() == 20
    assert sorted(themes["at_risk_rate"].tolist()) == [0.0, 1.0]
