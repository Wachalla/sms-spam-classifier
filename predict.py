"""Classify a message as spam or ham using the saved model."""

import joblib

model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

print("Type a message and press Enter ('quit' to exit).")

while True:
    message = input("> ")
    if message.lower() == "quit":
        break
    vector = vectorizer.transform([message])
    prediction = model.predict(vector)[0]
    print(f"[{prediction.upper()}] {message}")
