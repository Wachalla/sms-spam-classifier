# SMS Spam Classifier

> Originally built in 2024 as a university project; rebuilt and published in 2026
> while properly learning Git.

A Naive Bayes classifier that detects spam SMS messages, trained on the
UCI SMS Spam Collection (5,572 real text messages, 13.4% spam).

I rebuilt this from scratch to make sure I could still explain every line —
not just run it.

## Results

| Metric | Score |
|---|---|
| Test accuracy | 98.5% |
| Spam precision | 0.96 |
| Spam recall | 0.92 |
| Spam F1 | 0.94 |

Train accuracy (99.4%) vs test accuracy (98.5%) — the small gap means the
model generalizes instead of memorizing.

## How it works

1. Each message is converted into a word-count vector with `CountVectorizer`
   (fit on training data only, to avoid data leakage)
2. Multinomial Naive Bayes learns which words show up more in spam
   ("free", "winner", "claim") vs normal messages
3. Evaluated on a stratified 80/20 split so both sets keep the same spam ratio

## Run it

    pip install pandas scikit-learn
    python spam_classifier.py

## What I learned rebuilding this

- Why you never fit a vectorizer on test data (data leakage)
- Why accuracy alone lies on imbalanced data — support, precision and recall
  tell the real story
- Comparing train vs test accuracy is the quickest overfitting check

## Dataset

[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — included as `data/sms.tsv`
