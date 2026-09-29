"""Shared constants, data loading, splitting and model building."""

import csv
import re
import string

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Every random seed and split setting lives here.
SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5
UNCERTAIN_BAND = (0.3, 0.7)

DATA_PATH = "data/sms.tsv"
RESULTS_DIR = "results"


def load_data(path=DATA_PATH, legacy_parsing=False):
    """Read the TSV. The file has no quoting, so quote characters are kept as text.

    legacy_parsing=True uses the pandas default (quote-aware) parser, which is
    what v1 did. It merges some lines and strips some quote marks.
    """
    kwargs = {} if legacy_parsing else {"quoting": csv.QUOTE_NONE}
    return pd.read_csv(path, sep="\t", header=None, names=["label", "message"], **kwargs)


class LabelConflictError(ValueError):
    pass


def label_conflicts(df):
    """Messages that appear with more than one label."""
    n_labels = df.groupby("message")["label"].nunique()
    return sorted(n_labels[n_labels > 1].index)


def dedupe(df):
    """Drop exact duplicate messages, keeping the first copy. Stop on label conflicts."""
    conflicts = label_conflicts(df)
    if conflicts:
        raise LabelConflictError(f"{len(conflicts)} messages have conflicting labels: {conflicts[:5]}")
    return df.drop_duplicates(subset="message", keep="first")


def split(df):
    """Stratified train/test split with the shared seed."""
    return train_test_split(
        df["message"], df["label"], test_size=TEST_SIZE, random_state=SEED, stratify=df["label"]
    )


def overlap_count(X_train, X_test):
    """Number of test rows whose exact text also appears in train."""
    return int(pd.Series(X_test).isin(set(X_train)).sum())


def build_vectorizer():
    return CountVectorizer(stop_words="english", lowercase=True)


def fit_model(X_train, y_train):
    """Fit the vectorizer on training text only, then Naive Bayes on the counts."""
    vectorizer = build_vectorizer()
    model = MultinomialNB()
    model.fit(vectorizer.fit_transform(X_train), y_train)
    return vectorizer, model


_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")


def normalize(text):
    """Lowercase, remove punctuation, collapse whitespace. Used only to flag near-duplicates."""
    return " ".join(_PUNCT.sub("", text.lower()).split())
