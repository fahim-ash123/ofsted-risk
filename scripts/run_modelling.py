"""Train, compare and audit every model, writing the tables and figures for the report.

Run from the repository root after ``build_model_table.py``:

    python scripts/run_modelling.py --model-table data/processed/model_table.csv --out-dir data/processed/results

Models are trained on 2021/22 and 2022/23 and evaluated once on 2023/24.
Use ``--encoder lsa`` for a quick run without the sentence model.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from ofsted_risk import config, embeddings, evaluation, modelling  # noqa: E402
from ofsted_risk.features import CATEGORICAL_FEATURES, TEXT_FEATURE  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Read command-line options.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-table", type=Path, default=Path("data/processed/model_table.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/results"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--encoder", choices=("sentence", "lsa"), default="sentence")
    parser.add_argument("--bootstrap", type=int, default=config.BOOTSTRAP_RESAMPLES)
    return parser.parse_args()


def describe_dataset(table: pd.DataFrame) -> pd.DataFrame:
    """Summarise every column: type, missing share, range and distinct values.

    Args:
        table: The modelling table.

    Returns:
        One row per column, for the report's dataset description.
    """
    rows = []
    for column in table.columns:
        values = table[column]
        is_numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
        numeric = values.astype(float) if is_numeric else None
        rows.append(
            {
                "column": column,
                "type": "numeric" if is_numeric else ("boolean" if pd.api.types.is_bool_dtype(values) else "text or category"),
                "missing_share": round(float(values.isna().mean()), 4),
                "minimum": round(float(numeric.min()), 2) if is_numeric else None,
                "median": round(float(numeric.median()), 2) if is_numeric else None,
                "maximum": round(float(numeric.max()), 2) if is_numeric else None,
                "distinct_values": int(values.nunique()),
            }
        )
    return pd.DataFrame(rows)


def save_figure(figure, path: Path) -> None:
    """Save a figure at report resolution and close it.

    Args:
        figure: Matplotlib figure.
        path: Output file.
    """
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def exploratory_figures(table: pd.DataFrame, figures: Path) -> None:
    """Draw the exploratory figures used in the report.

    Args:
        table: The modelling table.
        figures: Output folder.
    """
    by_year = table.groupby("year")["label_primary"].agg(["mean", "size"])
    figure, axis = plt.subplots(figsize=(6, 3.5))
    axis.bar(by_year.index, by_year["mean"] * 100, color="#4472c4")
    for position, (rate, size) in enumerate(zip(by_year["mean"], by_year["size"])):
        axis.text(position, rate * 100 + 0.5, f"{rate:.1%}\nn={size:,}", ha="center", fontsize=8)
    axis.set_ylabel("At-risk rate (%)")
    axis.set_title("At-risk rate by target year")
    save_figure(figure, figures / "fig_at_risk_by_year.png")

    quintiles = table.pivot_table(index="idaci_quintile", columns="year", values="label_primary", aggfunc="mean") * 100
    figure, axis = plt.subplots(figsize=(6, 3.5))
    quintiles.plot(kind="bar", ax=axis)
    axis.set_xlabel("IDACI quintile (1 least deprived, 5 most deprived)")
    axis.set_ylabel("At-risk rate (%)")
    axis.set_title("At-risk rate by deprivation quintile and year")
    save_figure(figure, figures / "fig_at_risk_by_quintile.png")

    figure, axis = plt.subplots(figsize=(6, 3.5))
    for label, name in ((0, "Not at risk"), (1, "At risk")):
        axis.hist(table.loc[table["label_primary"] == label, "years_since_previous_inspection"], bins=30, alpha=0.6, density=True, label=name)
    axis.set_xlabel("Years since previous graded inspection")
    axis.set_ylabel("Density")
    axis.legend()
    axis.set_title("Time since the previous graded inspection")
    save_figure(figure, figures / "fig_years_since_previous.png")

    figure, axis = plt.subplots(figsize=(6, 3.5))
    axis.hist(table["text_characters"].clip(upper=3000), bins=40, color="#70ad47")
    axis.set_xlabel("Improvement section length (characters, capped at 3,000 for display)")
    axis.set_ylabel("Inspections")
    axis.set_title("Length of the previous report's improvement section")
    save_figure(figure, figures / "fig_text_length.png")


def main() -> None:
    """Run the full modelling stage and write every output."""
    args = parse_args()
    started = time.time()
    out = args.out_dir
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.model_table, dtype={"urn": str, "report_key": str, "year": str})
    describe_dataset(raw).to_csv(out / "dataset_description.csv", index=False)
    table = modelling.prepare(raw)
    example = table.head(3).copy()
    example[TEXT_FEATURE] = example[TEXT_FEATURE].str.slice(0, 160) + "..."
    example.to_csv(out / "example_rows.csv", index=False)
    exploratory_figures(table, figures)
    print(f"Exploratory outputs written ({time.time() - started:.0f}s)", flush=True)

    train = table["year"].isin(config.TRAIN_YEARS).to_numpy()
    test = (table["year"] == config.TEST_YEAR).to_numpy()
    labels = table["label_primary"].to_numpy(dtype=int)
    groups = table["urn"].to_numpy()

    encoder = embeddings.SentenceEncoder(config.SENTENCE_MODEL) if args.encoder == "sentence" else embeddings.LsaEncoder()
    # Several inspections can share one previous report, so each distinct
    # text is encoded once and the vectors are mapped back to the rows.
    distinct = pd.unique(table[TEXT_FEATURE])
    distinct_vectors = embeddings.cached_encode(encoder, list(distinct), args.cache_dir)
    position = {value: index for index, value in enumerate(distinct)}
    vectors = distinct_vectors[[position[value] for value in table[TEXT_FEATURE]]]
    print(f"Embeddings ready: {vectors.shape} ({time.time() - started:.0f}s)", flush=True)

    tabular = table[list(modelling.TABULAR_NUMERIC) + list(CATEGORICAL_FEATURES)]
    text = table[TEXT_FEATURE]
    inputs = {
        "Logistic regression (context)": (modelling.logistic_tabular(), tabular),
        "Gradient boosting (context)": (modelling.boosted_tabular(), tabular),
        "TF-IDF + logistic regression (text)": (modelling.tfidf_text(), text),
        "Sentence embeddings + logistic regression (text)": (modelling.embedding_text(), vectors),
    }
    searches, tuning = {}, []
    for name, ((pipeline, grid), features) in inputs.items():
        train_features = features[train] if isinstance(features, np.ndarray) else features.loc[train]
        searches[name] = modelling.tune(pipeline, grid, train_features, labels[train], groups[train])
        tuning.append({"model": name, "cv_pr_auc": round(searches[name].best_score_, 4), "best_params": json.dumps(searches[name].best_params_)})
        print(f"Tuned {name}: CV PR-AUC {searches[name].best_score_:.3f} ({time.time() - started:.0f}s)", flush=True)
    pd.DataFrame(tuning).to_csv(out / "tuning.csv", index=False)

    def subset(features, mask):
        """Select rows from a feature frame or array.

        Args:
            features: Feature frame, series or array.
            mask: Boolean row mask.

        Returns:
            The selected rows.
        """
        return features[mask] if isinstance(features, np.ndarray) else features.loc[mask]

    best_tabular = max(("Logistic regression (context)", "Gradient boosting (context)"), key=lambda name: searches[name].best_score_)
    best_text = max(("TF-IDF + logistic regression (text)", "Sentence embeddings + logistic regression (text)"), key=lambda name: searches[name].best_score_)
    fused = {name: searches[name] for name in (best_tabular, best_text)}
    fusion = modelling.fit_fusion(fused, {name: subset(inputs[name][1], train) for name in fused}, labels[train], groups[train])
    weights = {name: round(float(weight), 3) for name, weight in zip(fused, fusion.coef_[0])}
    print(f"Fusion weights {weights} ({time.time() - started:.0f}s)", flush=True)

    scores = {name: search.best_estimator_.predict_proba(subset(inputs[name][1], test))[:, 1] for name, search in searches.items()}
    fusion_name = "Late fusion (context + text)"
    scores[fusion_name] = fusion.predict_proba(np.column_stack([modelling.logit(scores[name]) for name in fused]))[:, 1]

    test_labels = labels[test]
    metrics = pd.DataFrame([{"model": name, **evaluation.summarise(test_labels, values, config.CAPACITY_SHARES)} for name, values in scores.items()])
    intervals = evaluation.bootstrap_intervals(test_labels, scores, best_tabular, args.bootstrap, config.RANDOM_STATE)
    metrics = metrics.merge(intervals, on="model").merge(pd.DataFrame(tuning), on="model", how="left")

    strict = table.loc[test, "label_strict"].to_numpy()
    graded = ~np.isnan(strict)
    metrics["strict_pr_auc"] = [
        average_precision_score(strict[graded].astype(int), values[graded]) for values in scores.values()
    ]
    metrics.round(4).to_csv(out / "model_comparison.csv", index=False)

    for name in (best_tabular, fusion_name):
        evaluation.calibration_table(test_labels, scores[name]).round(4).to_csv(out / f"calibration_{'fusion' if name == fusion_name else 'tabular'}.csv", index=False)
    fairness = evaluation.fairness_table(test_labels, scores[fusion_name], table.loc[test, "idaci_quintile"], config.FAIRNESS_CAPACITY)
    fairness.round(4).to_csv(out / "fairness_by_quintile.csv", index=False)

    importance = permutation_importance(
        searches[best_tabular].best_estimator_, subset(tabular, test), test_labels,
        scoring="average_precision", n_repeats=10, random_state=config.RANDOM_STATE,
    )
    importance_table = pd.DataFrame({"feature": tabular.columns, "mean_drop": importance.importances_mean, "std_drop": importance.importances_std}).sort_values("mean_drop", ascending=False)
    importance_table.round(4).to_csv(out / "permutation_importance.csv", index=False)

    clustering, themes = modelling.describe_themes(text[train].tolist(), vectors[train], labels[train], config.THEME_COUNT, config.RANDOM_STATE)
    test_themes = clustering.predict(vectors[test])
    themes["test_rows"] = [int((test_themes == theme).sum()) for theme in themes["theme"]]
    themes["test_at_risk_rate"] = [test_labels[test_themes == theme].mean() if (test_themes == theme).any() else np.nan for theme in themes["theme"]]
    themes.round(4).to_csv(out / "themes.csv", index=False)

    figure, axis = plt.subplots(figsize=(6, 4.5))
    for name, values in scores.items():
        precision, recall, _ = precision_recall_curve(test_labels, values)
        axis.plot(recall, precision, label=f"{name} ({average_precision_score(test_labels, values):.3f})")
    axis.axhline(test_labels.mean(), color="grey", linestyle="--", label=f"Base rate ({test_labels.mean():.3f})")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-recall on the 2023/24 test year")
    axis.legend(fontsize=6)
    save_figure(figure, figures / "fig_precision_recall.png")

    figure, axis = plt.subplots(figsize=(5, 4.5))
    for name in (best_tabular, fusion_name):
        bins = evaluation.calibration_table(test_labels, scores[name])
        axis.plot(bins["mean_predicted"], bins["observed_rate"], marker="o", label=name)
    axis.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Perfect calibration")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed at-risk rate")
    axis.set_title("Calibration on the 2023/24 test year")
    axis.legend(fontsize=7)
    save_figure(figure, figures / "fig_calibration.png")

    figure, axis = plt.subplots(figsize=(6, 3.5))
    width = 0.25
    positions = np.arange(len(fairness))
    for offset, column in zip((-width, 0, width), ("base_rate", "recall", "false_positive_rate")):
        axis.bar(positions + offset, fairness[column], width, label=column.replace("_", " "))
    axis.set_xticks(positions, [str(int(group)) if pd.notna(group) else "missing" for group in fairness["group"]])
    axis.set_xlabel("IDACI quintile (1 least deprived, 5 most deprived)")
    axis.set_title("Fused model at 20% capacity, by deprivation quintile")
    axis.legend(fontsize=7)
    save_figure(figure, figures / "fig_fairness.png")

    figure, axis = plt.subplots(figsize=(6, 4))
    top = importance_table.head(10).iloc[::-1]
    axis.barh(top["feature"], top["mean_drop"], xerr=top["std_drop"], color="#ed7d31")
    axis.set_xlabel("Drop in PR-AUC when shuffled")
    axis.set_title(f"Permutation importance: {best_tabular}")
    save_figure(figure, figures / "fig_permutation_importance.png")

    summary = {
        "encoder": encoder.model_name,
        "train_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        "test_at_risk_rate": float(test_labels.mean()),
        "best_tabular": best_tabular,
        "best_text": best_text,
        "fusion_weights": dict(zip(fused, [float(weight) for weight in fusion.coef_[0]])),
        "runtime_seconds": round(time.time() - started),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(metrics[["model", "pr_auc", "pr_auc_low", "pr_auc_high", "roc_auc", "brier", "precision_at_20", "recall_at_20", "gain_vs_baseline", "share_gain_positive"]].round(3).to_string(index=False))
    print(f"Done in {time.time() - started:.0f}s. Outputs in {out}")


if __name__ == "__main__":
    main()
