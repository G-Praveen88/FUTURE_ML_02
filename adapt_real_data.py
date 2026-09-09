"""
adapt_real_data.py
====================
Converts the Kaggle "Customer Support Ticket Dataset" (suraj520/customer-support-ticket-dataset)
into the same schema ticket_classification.py already expects:
    ticket_text, category, priority

Download from: https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset
Save the downloaded file as: data/customer_support_tickets.csv  (its original filename)

Run this BEFORE ticket_classification.py:
    python adapt_real_data.py
    python ticket_classification.py
"""
import pandas as pd

RAW_PATH = "data/customer_support_tickets.csv"
OUT_PATH = "data/support_tickets.csv"

raw = pd.read_csv(RAW_PATH)

print("Raw columns:", list(raw.columns))
print("Raw shape:", raw.shape)

# Combine subject + description into one text field for the model to read
raw["ticket_text"] = (
    raw["Ticket Subject"].fillna("") + ". " + raw["Ticket Description"].fillna("")
).str.strip()

df = pd.DataFrame({
    "ticket_text": raw["ticket_text"],
    "category": raw["Ticket Type"],
    "priority": raw["Ticket Priority"].str.lower(),
})

# Basic cleaning
df = df.dropna(subset=["ticket_text", "category", "priority"])
df = df[df["ticket_text"].str.len() > 5]  # drop near-empty text rows
df = df.reset_index(drop=True)

print("\nCategory distribution:\n", df["category"].value_counts())
print("\nPriority distribution:\n", df["priority"].value_counts())

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(df)} tickets -> {OUT_PATH}")
