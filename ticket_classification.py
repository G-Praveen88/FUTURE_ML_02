"""
FUTURE_ML_02 - Support Ticket Classification
==============================================
Task: Automatically classify customer support tickets by category
      and assign a priority level (high / medium / low).

Pipeline:
  1. Load & clean data
  2. Text preprocessing (lowercase, remove punctuation/stopwords)
  3. TF-IDF vectorization
  4. Train two classifiers: category model + priority model
  5. Evaluate with accuracy, precision, recall, F1
  6. Build a simple predict() function for new tickets
"""

import webbrowser
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# 1. Load & clean data
# ---------------------------------------------------------
df = pd.read_csv("data/support_tickets.csv")
print("Missing values:\n", df.isnull().sum())
df = df.dropna().reset_index(drop=True)
print(f"\nDataset size after cleaning: {len(df)} tickets ({df['ticket_text'].nunique()} unique)")

# ---------------------------------------------------------
# 2. Text preprocessing
# ---------------------------------------------------------
STOPWORDS = set("""a an the this that these those is are was were be been being
i you he she it we they my your his her its our their to of in on for and or but
with as at by from please help need want can cannot do does did not no""".split())

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)          # remove punctuation/numbers
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

df["clean_text"] = df["ticket_text"].apply(clean_text)
print("\nExample before/after cleaning:")
print("Raw  :", df["ticket_text"].iloc[0])
print("Clean:", df["clean_text"].iloc[0])

# ---------------------------------------------------------
# 3. TF-IDF vectorization
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["clean_text"])

# ---------------------------------------------------------
# 4a. Category classification model
# ---------------------------------------------------------
y_cat = df["category"]
X_train, X_test, y_cat_train, y_cat_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)

cat_model = LogisticRegression(max_iter=1000)
cat_model.fit(X_train, y_cat_train)
y_cat_pred = cat_model.predict(X_test)

print("\n=== CATEGORY CLASSIFICATION RESULTS ===")
print(f"Accuracy: {accuracy_score(y_cat_test, y_cat_pred):.3f}")
print(classification_report(y_cat_test, y_cat_pred))

# ---------------------------------------------------------
# 4b. Priority classification model
# ---------------------------------------------------------
y_pri = df["priority"]
_, _, y_pri_train, y_pri_test = train_test_split(
    X, y_pri, test_size=0.2, random_state=42, stratify=y_pri
)
# Reuse same split indices as category (same random_state/test_size ensures alignment)
X_train_p, X_test_p, y_pri_train, y_pri_test = train_test_split(
    X, y_pri, test_size=0.2, random_state=42, stratify=y_pri
)

pri_model = LogisticRegression(max_iter=1000, class_weight="balanced")
pri_model.fit(X_train_p, y_pri_train)
y_pri_pred = pri_model.predict(X_test_p)

print("\n=== PRIORITY CLASSIFICATION RESULTS ===")
print(f"Accuracy: {accuracy_score(y_pri_test, y_pri_pred):.3f}")
print(classification_report(y_pri_test, y_pri_pred))

# ---------------------------------------------------------
# 5. Confusion matrix visualization
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

labels_cat = sorted(y_cat.unique())
cm_cat = confusion_matrix(y_cat_test, y_cat_pred, labels=labels_cat)
im0 = axes[0].imshow(cm_cat, cmap="Blues")
axes[0].set_xticks(range(len(labels_cat))); axes[0].set_xticklabels(labels_cat, rotation=45)
axes[0].set_yticks(range(len(labels_cat))); axes[0].set_yticklabels(labels_cat)
axes[0].set_title("Category Confusion Matrix")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
for i in range(len(labels_cat)):
    for j in range(len(labels_cat)):
        axes[0].text(j, i, cm_cat[i, j], ha="center", va="center",
                     color="white" if cm_cat[i, j] > cm_cat.max()/2 else "black")

labels_pri = sorted(y_pri.unique())  # works for any set of priority levels, e.g. + "critical"
cm_pri = confusion_matrix(y_pri_test, y_pri_pred, labels=labels_pri)
im1 = axes[1].imshow(cm_pri, cmap="Oranges")
axes[1].set_xticks(range(len(labels_pri))); axes[1].set_xticklabels(labels_pri)
axes[1].set_yticks(range(len(labels_pri))); axes[1].set_yticklabels(labels_pri)
axes[1].set_title("Priority Confusion Matrix")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
for i in range(len(labels_pri)):
    for j in range(len(labels_pri)):
        axes[1].text(j, i, cm_pri[i, j], ha="center", va="center",
                     color="white" if cm_pri[i, j] > cm_pri.max()/2 else "black")

plt.tight_layout()
plt.savefig("classification_output.png", dpi=150)
webbrowser.open_new_tab("file://" + os.path.abspath("classification_output.png"))
print("\nSaved chart -> classification_output.png")

# ---------------------------------------------------------
# 6. Predict function for new incoming tickets
# ---------------------------------------------------------
def classify_ticket(text: str) -> dict:
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    category = cat_model.predict(vec)[0]
    priority = pri_model.predict(vec)[0]
    return {"ticket_text": text, "predicted_category": category, "predicted_priority": priority}

# Demo on new unseen tickets
new_tickets = [
    "The website has been down for an hour, this is critical for our business",
    "Can I get a student discount on the pro plan",
    "I need to update the email linked to my account",
]

print("\n=== DEMO: Classifying New Tickets ===")
demo_results = [classify_ticket(t) for t in new_tickets]
for r in demo_results:
    print(f"- \"{r['ticket_text'][:60]}...\" -> {r['predicted_category']} / {r['predicted_priority']}")

pd.DataFrame(demo_results).to_csv("demo_predictions.csv", index=False)
print("\nSaved demo predictions -> demo_predictions.csv")
