"""Gradio demo for the Naive Bayes SMS spam classifier.

Trains on load with the same v2 pipeline as spam_classifier.py: the UCI SMS
Spam Collection with exact duplicates dropped before the stratified 80/20
split, and CountVectorizer + MultinomialNB fit on training data only. Then it
serves a simple web UI for live classification.
"""

import os

import gradio as gr
from sklearn.metrics import accuracy_score

from common import DATA_PATH, dedupe, fit_model, load_data, overlap_count, split

DATA_URL = "https://raw.githubusercontent.com/Wachalla/sms-spam-classifier/main/data/sms.tsv"


# Use the bundled file when it is there, otherwise fall back to the copy on GitHub.
df = dedupe(load_data(DATA_PATH if os.path.exists(DATA_PATH) else DATA_URL))

X_train, X_test, y_train, y_test = split(df)
assert overlap_count(X_train, X_test) == 0, "train and test share messages"

vectorizer, model = fit_model(X_train, y_train)

test_acc = accuracy_score(y_test, model.predict(vectorizer.transform(X_test)))


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
        (exact duplicates dropped, then a stratified 80/20 train/test split, bag-of-words features).

        **Held-out test accuracy: {test_acc:.1%}** on {len(y_test)} test messages, none of which appear in the training set.

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
