"""Evaluation v2. Writes results/metrics.json, results/report.md and results/casebook.csv."""

import json
import os
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from common import (
    N_FOLDS, RESULTS_DIR, SEED, TEST_SIZE, UNCERTAIN_BAND,
    build_vectorizer, dedupe, fit_model, load_data, normalize, overlap_count, split,
)

LABELS = ["ham", "spam"]


def fmt(x):
    """The one number format used in report.md and the README."""
    return f"{x:.4f}"


def scores(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix": {
            "labels": LABELS,
            "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        },
    }


def run_setting(df):
    """Split, fit, score. Returns the pieces so callers can dig further."""
    X_train, X_test, y_train, y_test = split(df)
    vectorizer, model = fit_model(X_train, y_train)
    pred = model.predict(vectorizer.transform(X_test))
    return X_train, X_test, y_train, y_test, vectorizer, model, pred


def comparison_row(name, df):
    X_train, X_test, y_train, y_test, _, _, pred = run_setting(df)
    s = scores(y_test, pred)
    return {
        "setting": name,
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_test_overlap": overlap_count(X_train, X_test),
        "accuracy": s["accuracy"],
        "spam_precision": s["per_class"]["spam"]["precision"],
        "spam_recall": s["per_class"]["spam"]["recall"],
        "spam_f1": s["per_class"]["spam"]["f1"],
    }


def overlap_breakdown(df):
    """Accuracy on test messages seen in train vs truly unseen ones."""
    X_train, X_test, _, y_test, _, _, pred = run_setting(df)
    seen = X_test.isin(set(X_train)).to_numpy()
    y = y_test.to_numpy()
    return {
        "n_test": int(len(X_test)),
        "n_seen_in_train": int(seen.sum()),
        "accuracy_seen_in_train": float((pred[seen] == y[seen]).mean()),
        "n_unseen": int((~seen).sum()),
        "accuracy_unseen": float((pred[~seen] == y[~seen]).mean()),
    }


def cross_validate(df):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    X, y = df["message"].to_numpy(), df["label"].to_numpy()
    acc, f1 = [], []
    for train_idx, test_idx in skf.split(X, y):
        # The pipeline refits the vectorizer inside each fold, on that fold's training part only.
        pipe = make_pipeline(build_vectorizer(), MultinomialNB())
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        acc.append(accuracy_score(y[test_idx], pred))
        f1.append(f1_score(y[test_idx], pred, pos_label="spam"))
    return {
        "n_folds": N_FOLDS,
        "n_rows": int(len(df)),
        "std_note": "sample standard deviation (ddof=1) across folds",
        "accuracy_per_fold": [float(a) for a in acc],
        "accuracy_mean": float(np.mean(acc)),
        "accuracy_std": float(np.std(acc, ddof=1)),
        "spam_f1_per_fold": [float(f) for f in f1],
        "spam_f1_mean": float(np.mean(f1)),
        "spam_f1_std": float(np.std(f1, ddof=1)),
    }


def build_casebook(X_test, y_test, pred, proba_spam):
    lo, hi = UNCERTAIN_BAND
    frame = pd.DataFrame({
        "id": X_test.index,
        "message": X_test.to_numpy(),
        "true_label": y_test.to_numpy(),
        "predicted_label": pred,
        "spam_probability": np.round(proba_spam, 4),
    })
    wrong = frame["true_label"] != frame["predicted_label"]
    uncertain = frame["spam_probability"].between(lo, hi)
    book = frame[wrong | uncertain].sort_values("spam_probability").copy()
    for col in ["my_label", "reason", "rule"]:
        book[col] = ""
    return book, int(wrong.sum()), int(uncertain.sum()), int((wrong & uncertain).sum())


def run_evaluation(audit=None):
    raw = load_data()
    legacy = load_data(legacy_parsing=True)
    clean = dedupe(raw)

    comparison = [
        comparison_row("v1 original (pandas default parser, no dedupe)", legacy),
        comparison_row("fixed parser, no dedupe", raw),
        comparison_row("v2 (fixed parser, exact duplicates dropped)", clean),
    ]

    X_train, X_test, y_train, y_test, vectorizer, model, pred = run_setting(clean)
    overlap = overlap_count(X_train, X_test)
    assert overlap == 0, "train and test share messages"
    spam_col = list(model.classes_).index("spam")
    proba_spam = model.predict_proba(vectorizer.transform(X_test))[:, spam_col]

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train.to_frame(), y_train)
    baseline_pred = baseline.predict(X_test.to_frame())

    train_norm = set(X_train.map(normalize))
    near_dup_overlap = int(X_test.map(normalize).isin(train_norm).sum())

    book, n_wrong, n_uncertain, n_both = build_casebook(X_test, y_test, pred, proba_spam)

    metrics = {
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "settings": {"seed": SEED, "test_size": TEST_SIZE, "n_folds": N_FOLDS, "uncertain_band": list(UNCERTAIN_BAND)},
        "audit": audit,
        "comparison": comparison,
        "v2_holdout": {
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "train_test_overlap": overlap,
            "near_duplicate_overlap": near_dup_overlap,
            "baseline_always_ham": scores(y_test, baseline_pred),
            "model": scores(y_test, pred),
        },
        "v2_cross_validation": cross_validate(clean),
        "original_split_seen_vs_unseen": overlap_breakdown(legacy),
        "casebook": {
            "rows": int(len(book)),
            "misclassified": n_wrong,
            "uncertain": n_uncertain,
            "misclassified_and_uncertain": n_both,
        },
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    book.to_csv(os.path.join(RESULTS_DIR, "casebook.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write(render_report(metrics))
    return metrics


def comparison_table(m):
    lines = [
        "| Setting | Rows | Train n | Test n | Train/test overlap | Accuracy | Spam precision | Spam recall | Spam F1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in m["comparison"]:
        lines.append(
            f"| {r['setting']} | {r['n_rows']} | {r['n_train']} | {r['n_test']} | {r['train_test_overlap']} "
            f"| {fmt(r['accuracy'])} | {fmt(r['spam_precision'])} | {fmt(r['spam_recall'])} | {fmt(r['spam_f1'])} |"
        )
    return "\n".join(lines)


def holdout_table(m):
    h = m["v2_holdout"]
    cv = m["v2_cross_validation"]
    lines = [
        f"Test n = {h['n_test']}, train n = {h['n_train']}, train/test overlap = {h['train_test_overlap']}.",
        "",
        "| Model | Accuracy | Spam precision | Spam recall | Spam F1 | Ham precision | Ham recall | Ham F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, s in [("Baseline: always predict ham", h["baseline_always_ham"]), ("Naive Bayes (v2)", h["model"])]:
        sp, hm = s["per_class"]["spam"], s["per_class"]["ham"]
        lines.append(
            f"| {name} | {fmt(s['accuracy'])} | {fmt(sp['precision'])} | {fmt(sp['recall'])} | {fmt(sp['f1'])} "
            f"| {fmt(hm['precision'])} | {fmt(hm['recall'])} | {fmt(hm['f1'])} |"
        )
    lines += [
        "",
        f"{cv['n_folds']}-fold stratified cross-validation on all {cv['n_rows']} deduplicated messages: "
        f"accuracy {fmt(cv['accuracy_mean'])} ± {fmt(cv['accuracy_std'])}, "
        f"spam F1 {fmt(cv['spam_f1_mean'])} ± {fmt(cv['spam_f1_std'])} (mean ± sample standard deviation).",
    ]
    return "\n".join(lines)


def render_report(m):
    h = m["v2_holdout"]
    cm = h["model"]["confusion_matrix"]["rows_true_cols_pred"]
    o = m["original_split_seen_vs_unseen"]
    c = m["casebook"]
    cv = m["v2_cross_validation"]
    env = m["environment"]
    return "\n".join([
        "# Evaluation report (v2)",
        "",
        "Generated by `evaluate.py`. Do not edit by hand.",
        "",
        f"Python {env['python']}, scikit-learn {env['scikit_learn']}, pandas {env['pandas']}, numpy {env['numpy']}. "
        f"Seed {m['settings']['seed']}, test size {m['settings']['test_size']}.",
        "",
        "## Before and after",
        "",
        "Same model, same split settings. Only the data handling changes.",
        "",
        comparison_table(m),
        "",
        "## v2 holdout results",
        "",
        holdout_table(m),
        "",
        "Cross-validation folds (accuracy): " + ", ".join(fmt(a) for a in cv["accuracy_per_fold"]),
        "",
        "Confusion matrix on the v2 test set (rows are true labels, columns are predictions):",
        "",
        "| | predicted ham | predicted spam |",
        "|---|---|---|",
        f"| true ham | {cm[0][0]} | {cm[0][1]} |",
        f"| true spam | {cm[1][0]} | {cm[1][1]} |",
        "",
        "## Original split: seen vs unseen test messages",
        "",
        "| Test messages | n | Accuracy |",
        "|---|---|---|",
        f"| Also in train (exact copy) | {o['n_seen_in_train']} | {fmt(o['accuracy_seen_in_train'])} |",
        f"| Not in train | {o['n_unseen']} | {fmt(o['accuracy_unseen'])} |",
        f"| All | {o['n_test']} | {fmt(m['comparison'][0]['accuracy'])} |",
        "",
        "## Near-duplicates across the v2 split",
        "",
        f"v2 test messages whose normalized text (lowercased, punctuation removed, whitespace collapsed) "
        f"also appears in train: {h['near_duplicate_overlap']} of {h['n_test']}. Flagged, not removed.",
        "",
        "## Casebook",
        "",
        f"`results/casebook.csv` has {c['rows']} rows: {c['misclassified']} misclassified, "
        f"{c['uncertain']} with spam probability between {m['settings']['uncertain_band'][0]} and "
        f"{m['settings']['uncertain_band'][1]}, {c['misclassified_and_uncertain']} in both groups.",
        "",
    ])


if __name__ == "__main__":
    print(render_report(run_evaluation()))
