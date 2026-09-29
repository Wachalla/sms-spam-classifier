"""Naive Bayes SMS spam classifier trained on the UCI SMS Spam Collection."""

import joblib
from sklearn.metrics import accuracy_score, classification_report

from common import dedupe, fit_model, load_data, overlap_count, split


def main(save=True):
    df = load_data()
    print(f"Loaded {len(df)} messages ({(df.label == 'spam').sum()} spam)")

    # Drop exact duplicates before splitting, so no message can land on both sides.
    df = dedupe(df)
    print(f"After dropping exact duplicates: {len(df)} messages")

    X_train, X_test, y_train, y_test = split(df)
    overlap = overlap_count(X_train, X_test)
    print(f"Test messages that also appear in train: {overlap}")
    assert overlap == 0, "train and test share messages"

    # Turn text into word counts. The vocabulary comes from training data only.
    vectorizer, model = fit_model(X_train, y_train)
    X_test_vec = vectorizer.transform(X_test)
    predictions = model.predict(X_test_vec)

    train_acc = accuracy_score(y_train, model.predict(vectorizer.transform(X_train)))
    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, digits=4))

    # Try it on new messages
    samples = [
        "Congratulations! You won a free prize, claim now!",
        "Hey, are we still meeting for lunch tomorrow?",
    ]
    for msg, pred in zip(samples, model.predict(vectorizer.transform(samples))):
        print(f"[{pred.upper()}] {msg}")

    if save:
        joblib.dump(model, "model.joblib")
        joblib.dump(vectorizer, "vectorizer.joblib")
        print("\nSaved model.joblib and vectorizer.joblib")


if __name__ == "__main__":
    main()
