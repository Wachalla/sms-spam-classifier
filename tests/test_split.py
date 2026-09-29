import pandas as pd
import pytest

from common import LabelConflictError, dedupe, load_data, overlap_count, split


def test_train_and_test_share_no_messages():
    X_train, X_test, _, _ = split(dedupe(load_data()))
    assert overlap_count(X_train, X_test) == 0
    assert not set(X_train) & set(X_test)


def test_without_dedupe_the_split_does_leak():
    # Guards the test above: it only means something if the raw data would leak.
    X_train, X_test, _, _ = split(load_data())
    assert overlap_count(X_train, X_test) > 0


def test_dedupe_stops_on_label_conflicts():
    df = pd.DataFrame({"label": ["ham", "spam", "ham"], "message": ["hi", "hi", "bye"]})
    with pytest.raises(LabelConflictError):
        dedupe(df)


def test_dedupe_keeps_first_copy():
    df = pd.DataFrame({"label": ["ham", "ham", "spam"], "message": ["hi", "hi", "win"]})
    out = dedupe(df)
    assert list(out.index) == [0, 2]
