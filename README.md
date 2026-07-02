# SMS Spam Classifier

A Naive Bayes classifier that detects spam SMS messages, trained on the
UCI SMS Spam Collection (5,572 messages, 13.4% spam).

## Results

| Metric | Score |
|---|---|
| Test accuracy | 98.5% |
| Spam precision | 0.96 |
| Spam recall | 0.92 |
| Spam F1 | 0.94 |

Train accuracy (99.4%) vs test accuracy (98.5%) shows minimal overfitting.

## How it works

1. Messages are converted to word-count vectors with `CountVectorizer`
   (fit on training data only, to avoid data leakage)
2. A Multinomial Naive Bayes model learns which words are likelier in spam
3. Evaluation uses a stratified 80/20 train/test split

## Run it

    pip install pandas scikit-learn
    python spam_classifier.py

## Dataset

[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — `data/sms.tsv`
