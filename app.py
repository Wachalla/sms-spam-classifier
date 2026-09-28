"""Gradio demo for the Naive Bayes SMS spam classifier.

Trains on load from the same data/sms.tsv used in spam_classifier.py
(UCI SMS Spam Collection), using the identical stratified 80/20 split
and CountVectorizer + MultinomialNB pipeline, then serves a simple
web UI for live classification.
"""

import os

import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

DATA_URL = "https://raw.githubusercontent.com/Wachalla/sms-spam-classifier/main/data/sms.tsv"


def load_data(path=DATA_URL):
    return pd.read_csv(path, sep="\t", header=None, names=["label", "message"])


df = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

vectorizer = CountVectorizer(stop_words="english", lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

test_acc = accuracy_score(y_test, model.predict(X_test_vec))


def classify(message):
    if not message or not message.strip():
        return "Enter a message to classify.", {}
    vec = vectorizer.transform([message])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    proba_dict = {c: float(p) for c, p in zip(model.classes_, proba)}
    label = "SPAM" if pred == "spam" else "HAM (not spam)"
    return label, proba_dict


with gr.Blocks(title="SMS Spam Classifier") as demo:
    gr.Markdown(
        f"""
        # SMS Spam Classifier
        Naive Bayes classifier trained on the UCI SMS Spam Collection
        (stratified 80/20 train/test split, bag-of-words features).

        **Held-out test accuracy: {test_acc:.1%}**

        Source: [github.com/Wachalla/sms-spam-classifier](https://github.com/Wachalla/sms-spam-classifier)
        """
    )
    inp = gr.Textbox(label="SMS message", placeholder="Type or paste a text message...", lines=3)
    btn = gr.Button("Classify", variant="primary")
    out_label = gr.Textbox(label="Prediction")
    out_proba = gr.Label(label="Class probabilities")
    btn.click(classify, inputs=inp, outputs=[out_label, out_proba])

    gr.Examples(
        examples=[
            "Congratulations! You won a free prize, claim now!",
            "Hey, are we still meeting for lunch tomorrow?",
            "URGENT! Your account has been suspended, click here to verify",
            "Can you pick up milk on your way home?",
        ],
        inputs=inp,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
