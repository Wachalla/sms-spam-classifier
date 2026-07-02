"""Naive Bayes SMS spam classifier trained on the UCI SMS Spam Collection."""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def load_data(path="data/sms.tsv"):
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])
    return df


def main():
    df = load_data()
    print(f"Loaded {len(df)} messages ({(df.label == 'spam').sum()} spam)")

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Turn text into word counts: each message becomes a vector of word frequencies
    vectorizer = CountVectorizer(stop_words="english", lowercase=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    train_acc = accuracy_score(y_train, model.predict(X_train_vec))
    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions))

    # Try it on new messages
    samples = [
        "Congratulations! You won a free prize, claim now!",
        "Hey, are we still meeting for lunch tomorrow?",
    ]
    for msg, pred in zip(samples, model.predict(vectorizer.transform(samples))):
        print(f"[{pred.upper()}] {msg}")


if __name__ == "__main__":
    main()
