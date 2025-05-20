import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("dataset/fake_or_real_news.csv")

# Basic cleaning
df = df.dropna(subset=["text", "label"])
df = df.drop_duplicates()

# Encode label
df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text

df["clean_text"] = df["text"].apply(clean_text)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n{name} Evaluation:")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
    print("Accuracy:", accuracy_score(y_test, preds))

    # ROC Curve
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, probs)
        plt.title(f"ROC Curve - {name}")
        plt.show()
