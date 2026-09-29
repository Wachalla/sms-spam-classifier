from common import build_vectorizer, dedupe, fit_model, load_data, split


def test_vocabulary_comes_from_training_data_only():
    X_train, X_test, y_train, _ = split(dedupe(load_data()))
    vectorizer, _ = fit_model(X_train, y_train)
    vocab = set(vectorizer.vocabulary_)

    train_tokens = set(build_vectorizer().fit(X_train).vocabulary_)
    test_tokens = set(build_vectorizer().fit(X_test).vocabulary_)
    test_only = test_tokens - train_tokens

    assert vocab == train_tokens
    assert test_only, "expected some words that appear only in test"
    assert not vocab & test_only
