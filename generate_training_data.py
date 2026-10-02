"""
Generates a realistic, well-correlated training set with genuinely
distinct sentences per row (no filler-duplication / leakage).
"""
import pandas as pd

rows = [
    # Billing inquiry
    ("I was charged twice for my order, please refund the extra amount immediately", "Billing inquiry", "High"),
    ("My subscription payment failed but the amount was still deducted from my bank account", "Billing inquiry", "High"),
    ("Why does my invoice show a different amount than what I was quoted", "Billing inquiry", "Medium"),
    ("I need to update the billing address linked to my account", "Billing inquiry", "Low"),
    ("Could you send me a copy of last month's invoice for my records", "Billing inquiry", "Low"),
    ("There is an unexpected extra charge on my latest bill, please explain", "Billing inquiry", "Medium"),
    ("I'd like to switch my payment method to a different card", "Billing inquiry", "Low"),
    ("I was billed for a service I already cancelled, please correct this", "Billing inquiry", "High"),
    ("Can you confirm when my next billing cycle starts", "Billing inquiry", "Low"),
    ("My credit card got charged in a different currency than expected", "Billing inquiry", "Medium"),

    # Technical issue
    ("The app crashes every single time I try to upload a photo", "Technical issue", "Medium"),
    ("Our entire system is completely down and nobody on the team can log in", "Technical issue", "Critical"),
    ("I keep getting an error message whenever I try to sign in to my account", "Technical issue", "High"),
    ("The dashboard takes a long time to load but it does eventually work", "Technical issue", "Low"),
    ("The device stopped turning on at all and nothing I try fixes it", "Technical issue", "Critical"),
    ("Some buttons on the settings page are slightly misaligned visually", "Technical issue", "Low"),
    ("The sync feature between devices has stopped working since yesterday", "Technical issue", "Medium"),
    ("Production servers are down and customers cannot check out, we are losing sales right now", "Technical issue", "Critical"),
    ("I noticed a small typo in the help documentation", "Technical issue", "Low"),
    ("The mobile app freezes randomly during video playback", "Technical issue", "Medium"),

    # Refund request
    ("I would like a refund for the item I returned last week", "Refund request", "Medium"),
    ("My refund has still not arrived even though it has been over ten days", "Refund request", "High"),
    ("What is your policy on refunds if I'm not satisfied with a product", "Refund request", "Low"),
    ("I never received a confirmation email for my refund, please check urgently", "Refund request", "High"),
    ("Can I get a partial refund since only one item in my order was damaged", "Refund request", "Medium"),
    ("I'm just curious how long refunds usually take to process", "Refund request", "Low"),
    ("The refunded amount I received is less than what I actually paid", "Refund request", "High"),
    ("Please process my refund right away, I need the money back urgently for an emergency", "Refund request", "Critical"),

    # Cancellation request
    ("I want to cancel my subscription starting next month, no rush", "Cancellation request", "Low"),
    ("Please cancel my order right now, I no longer need it", "Cancellation request", "High"),
    ("How do I go about cancelling my membership whenever I decide to", "Cancellation request", "Low"),
    ("I tried to cancel my plan but the system won't let me, and I'm being charged again very soon", "Cancellation request", "Critical"),
    ("I'd like to pause my subscription for a couple of months instead of cancelling", "Cancellation request", "Low"),
    ("My cancellation request from last week still hasn't been processed", "Cancellation request", "High"),
    ("Just checking what happens to my data if I cancel my account", "Cancellation request", "Low"),

    # Product inquiry
    ("What are the full specifications of this laptop model", "Product inquiry", "Low"),
    ("Does this product come available in other colors or sizes", "Product inquiry", "Low"),
    ("Is this item compatible with my existing home setup", "Product inquiry", "Low"),
    ("I need to know the warranty details before I decide to purchase", "Product inquiry", "Medium"),
    ("Can you tell me the expected delivery time for this item", "Product inquiry", "Low"),
    ("Is there a bulk discount available if I order more than ten units", "Product inquiry", "Low"),
    ("I'd like to know if this product is suitable for outdoor use", "Product inquiry", "Low"),
    ("What accessories are included in the box with this purchase", "Product inquiry", "Low"),
]

df = pd.DataFrame(rows, columns=["Ticket Description", "Ticket Type", "Ticket Priority"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("support_tickets_demo.csv", index=False)
print(f"Generated {len(df)} rows")
print(df["Ticket Type"].value_counts())
print(df["Ticket Priority"].value_counts())
