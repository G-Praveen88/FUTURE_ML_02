"""
Generates a synthetic customer support ticket dataset with categories and priority levels.
Adds phrasing variation so tickets aren't near-duplicates of each other.
"""
import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

templates = {
    "Billing": [
        "I was charged twice for my {plan} subscription this month",
        "My invoice shows an incorrect amount of ${amt}, please fix it",
        "Refund request for order #{oid} that never arrived",
        "Why was I billed after canceling my {plan} plan",
        "Payment failed but ${amt} was deducted from my account",
        "Need a copy of my last {n} invoices for tax purposes",
        "Subscription renewed automatically without my consent",
        "The discount code {code} did not apply to my order total",
        "I'm being charged for a plan I already downgraded from",
        "Can you explain the extra ${amt} fee on my latest bill",
    ],
    "Technical": [
        "The app keeps crashing every time I open the {feature} tab",
        "I cannot log into my account, it says invalid password",
        "Page is loading extremely slowly on {browser}",
        "Getting a 500 error when I try to upload a {filetype} file",
        "The mobile app is stuck on the loading screen since {time}",
        "Two-factor authentication code never arrives via {method}",
        "Website is completely down, cannot access dashboard",
        "Export feature is broken and downloads empty {filetype} files",
        "The {feature} module throws an error every time I click save",
        "Sync between devices stopped working after the last update",
    ],
    "Account": [
        "I want to change the email address on my account to {email}",
        "Please help me delete my account permanently",
        "I forgot my username and need help recovering it",
        "How do I update my shipping address to {city}",
        "My account was suspended for no clear reason",
        "Need to merge two accounts into one under {email}",
        "Unable to change my password from the settings page",
        "Someone else may have accessed my account without permission",
        "My profile picture upload keeps failing",
        "Can you verify my account, the badge is missing",
    ],
    "Product": [
        "Does the {plan} plan include access to premium templates",
        "Can I use this software offline without internet",
        "What is the difference between the {plan} and enterprise plans",
        "Is there a mobile app version available for {os}",
        "How many users can I add under one license",
        "Does the product support integration with {tool}",
        "What file formats are supported for export",
        "Is there a student discount available for the {plan} tier",
        "Can I customize the {feature} dashboard layout",
        "Do you offer a free trial before upgrading to {plan}",
    ],
}

fillers = {
    "plan": ["Basic", "Pro", "Premium", "Starter", "Enterprise"],
    "amt": ["19.99", "49.00", "99.50", "12.00", "150.00"],
    "oid": [f"{n}" for n in range(10023, 10099)],
    "n": ["three", "six", "twelve"],
    "code": ["SAVE10", "WELCOME20", "SUMMER25"],
    "feature": ["reports", "dashboard", "analytics", "billing", "export"],
    "browser": ["Chrome", "Safari", "Firefox", "Edge"],
    "filetype": ["CSV", "PDF", "Excel"],
    "time": ["this morning", "yesterday", "an hour ago"],
    "method": ["SMS", "email", "authenticator app"],
    "email": ["newmail@example.com", "work@example.com", "personal@example.com"],
    "city": ["Chennai", "Mumbai", "Bangalore", "Delhi"],
    "os": ["iOS", "Android"],
    "tool": ["Slack", "Zapier", "Google Drive", "Microsoft Teams"],
}

urgent_words = ["urgent", "immediately", "asap", "critical", "down", "cannot access", "suspended", "unauthorized"]

def fill_template(t: str) -> str:
    for key, options in fillers.items():
        if "{" + key + "}" in t:
            t = t.replace("{" + key + "}", random.choice(options))
    return t

rows = []
for _ in range(600):
    cat = random.choice(list(templates.keys()))
    text = fill_template(random.choice(templates[cat]))

    if random.random() < 0.35:
        text = f"{text} - this is {random.choice(['urgent', 'critical', 'very time sensitive'])}, please help ASAP"

    has_urgent = any(w in text.lower() for w in urgent_words)
    if cat == "Technical" and has_urgent:
        priority = "high"
    elif cat == "Billing" and has_urgent:
        priority = "high"
    elif has_urgent:
        priority = "medium"
    elif cat in ["Technical", "Billing"]:
        priority = np.random.choice(["medium", "low"], p=[0.6, 0.4])
    else:
        priority = np.random.choice(["low", "medium"], p=[0.7, 0.3])

    rows.append({"ticket_text": text, "category": cat, "priority": priority})

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Unique tickets: {df['ticket_text'].nunique()} / {len(df)}")
df.to_csv("/home/claude/FUTURE_ML_02/data/support_tickets.csv", index=False)
print(df.head(10))
print("\nCategory distribution:\n", df["category"].value_counts())
print("\nPriority distribution:\n", df["priority"].value_counts())
