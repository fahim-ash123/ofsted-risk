"""Training the basic, text and fused models on the earlier years.

Every model is tuned by five-fold cross-validation within the training
years, grouped by school so that the same school never sits on both sides
of a fold, and scored by average precision (PR-AUC). The fused model is a
stacked late fusion: a logistic regression learns how to weight the
out-of-fold scores of the best tabular and best text model, as in the
strongest pipelines benchmarked by Shi et al. (2021).
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ofsted_risk import config
from ofsted_risk.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TEXT_FEATURE

# Text length describes the report, not the school, so it belongs with the
# text models and is kept out of the tabular feature set.
TABULAR_NUMERIC = tuple(name for name in NUMERIC_FEATURES if name != "text_characters")


def prepare(table: pd.DataFrame) -> pd.DataFrame:
    """Make categorical and text columns safe for scikit-learn in any pandas version.

    Args:
        table: Modelling table as read from CSV.

    Returns:
        A copy with categoricals and text as plain strings, missing values
        made explicit, and numeric features as floats.
    """
    prepared = table.copy()
    for column in CATEGORICAL_FEATURES:
        prepared[column] = prepared[column].astype(object).where(prepared[column].notna(), "missing").astype(str)
    prepared[TEXT_FEATURE] = prepared[TEXT_FEATURE].astype(object).where(prepared[TEXT_FEATURE].notna(), "").astype(str)
    for column in TABULAR_NUMERIC:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").astype(float)
    return prepared


def logistic_tabular() -> Tuple[Pipeline, dict]:
    """Penalised logistic regression on school context, the transparent baseline.

    Returns:
        The pipeline and its tuning grid.
    """
    columns = ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), list(TABULAR_NUMERIC)),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORICAL_FEATURES)),
        ]
    )
    pipeline = Pipeline([("columns", columns), ("model", LogisticRegression(max_iter=2000))])
    return pipeline, {"model__C": [0.1, 1.0, 10.0]}


def boosted_tabular() -> Tuple[Pipeline, dict]:
    """Histogram gradient boosting on school context, the tabular benchmark.

    Missing numeric values are handled natively, and categoricals are passed
    as ordinal codes flagged as categorical, so no imputation is needed.

    Returns:
        The pipeline and its tuning grid.
    """
    columns = ColumnTransformer(
        [
            ("numeric", "passthrough", list(TABULAR_NUMERIC)),
            ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), list(CATEGORICAL_FEATURES)),
        ]
    )
    categorical_mask = [False] * len(TABULAR_NUMERIC) + [True] * len(CATEGORICAL_FEATURES)
    model = HistGradientBoostingClassifier(
        categorical_features=categorical_mask, max_iter=300, early_stopping=True, random_state=config.RANDOM_STATE
    )
    pipeline = Pipeline([("columns", columns), ("model", model)])
    grid = {
        "model__learning_rate": [0.05, 0.1],
        "model__max_leaf_nodes": [15, 31],
        "model__min_samples_leaf": [20, 50],
    }
    return pipeline, grid


def tfidf_text() -> Tuple[Pipeline, dict]:
    """TF-IDF word and bigram features with logistic regression, the text baseline.

    Returns:
        The pipeline and its tuning grid.
    """
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    return pipeline, {"model__C": [0.3, 1.0, 3.0]}


def embedding_text() -> Tuple[Pipeline, dict]:
    """Logistic regression on frozen sentence embeddings, the advanced text model.

    Returns:
        The pipeline and its tuning grid.
    """
    return Pipeline([("model", LogisticRegression(max_iter=2000))]), {"model__C": [0.1, 1.0, 10.0]}


def tune(pipeline: Pipeline, grid: dict, features, labels: np.ndarray, groups: np.ndarray) -> GridSearchCV:
    """Tune a pipeline by grouped cross-validation on PR-AUC and refit on all training rows.

    Args:
        pipeline: Model pipeline.
        grid: Parameter grid.
        features: Training features in the form the pipeline expects.
        labels: Training labels.
        groups: School identifier of each row.

    Returns:
        The fitted search, whose ``best_estimator_`` is refitted on all rows.
    """
    search = GridSearchCV(
        pipeline, grid, scoring="average_precision", cv=GroupKFold(n_splits=config.CV_FOLDS), n_jobs=1, refit=True
    )
    search.fit(features, labels, groups=groups)
    return search


def logit(probabilities: np.ndarray) -> np.ndarray:
    """Convert probabilities to log-odds, clipped away from 0 and 1.

    Args:
        probabilities: Predicted probabilities.

    Returns:
        Log-odds of the same shape.
    """
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def fit_fusion(
    searches: Dict[str, GridSearchCV], inputs: Dict[str, object], labels: np.ndarray, groups: np.ndarray
) -> LogisticRegression:
    """Learn how to weight the base models from their out-of-fold scores.

    Out-of-fold scores are used so that the weights reflect how each model
    performs on rows it was not trained on, as it will at test time.

    Args:
        searches: Fitted searches for the base models being fused.
        inputs: Training inputs for each base model, keyed like ``searches``.
        labels: Training labels.
        groups: School identifier of each row.

    Returns:
        A logistic regression over the base models' log-odds.
    """
    folds = GroupKFold(n_splits=config.CV_FOLDS)
    columns = [
        logit(
            cross_val_predict(
                search.best_estimator_, inputs[name], labels, groups=groups, cv=folds, method="predict_proba"
            )[:, 1]
        )
        for name, search in searches.items()
    ]
    return LogisticRegression(max_iter=1000).fit(np.column_stack(columns), labels)


def describe_themes(
    texts: List[str], embeddings: np.ndarray, labels: np.ndarray, count: int, seed: int, top_terms: int = 6
) -> Tuple[KMeans, pd.DataFrame]:
    """Group improvement texts into themes and describe each with its distinctive words.

    This follows the BERTopic recipe (Grootendorst, 2022): cluster the
    embeddings, then join each cluster's texts into one document and rank
    words by TF-IDF across clusters.

    Args:
        texts: Training texts.
        embeddings: Their embeddings.
        labels: Their labels, to report the at-risk rate per theme.
        count: Number of themes.
        seed: Random seed.
        top_terms: Number of words used to describe each theme.

    Returns:
        The fitted clustering and one row per theme with its size,
        at-risk rate and describing words.
    """
    clustering = KMeans(n_clusters=count, n_init=10, random_state=seed).fit(embeddings)
    documents = [" ".join(np.array(texts)[clustering.labels_ == theme]) for theme in range(count)]
    vectoriser = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_df=0.8, sublinear_tf=True)
    weights = vectoriser.fit_transform(documents).toarray()
    vocabulary = np.array(vectoriser.get_feature_names_out())
    rows = []
    for theme in range(count):
        members = clustering.labels_ == theme
        rows.append(
            {
                "theme": theme,
                "rows": int(members.sum()),
                "at_risk_rate": float(labels[members].mean()) if members.any() else np.nan,
                "describing_words": ", ".join(vocabulary[np.argsort(-weights[theme])[:top_terms]]),
            }
        )
    return clustering, pd.DataFrame(rows)
