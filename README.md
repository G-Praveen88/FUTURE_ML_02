# FUTURE_ML_02 — Support Ticket Classification

**Track:** Machine Learning | **Program:** Future Interns

## 📌 Task
Build a system to automatically classify customer support tickets and assign priority levels.

## 🧠 Skills Applied
Text preprocessing, NLP classification (TF-IDF + Logistic Regression), priority logic, support analytics.

## 🗂️ Project Structure
```
FUTURE_ML_02/
├── data/
│   └── support_tickets.csv       # 600 labeled support tickets (generated)
├── generate_data.py              # Creates the synthetic ticket dataset
├── ticket_classification.py      # Main pipeline: clean -> vectorize -> train -> evaluate -> demo
├── classification_output.png     # Confusion matrices (category + priority)
├── demo_predictions.csv          # Predictions on 3 brand-new example tickets
└── README.md
```

## ⚙️ How It Works
1. **Data cleaning** — checks for missing/duplicate entries.
2. **Text preprocessing** — lowercasing, punctuation/number removal, custom stopword filtering.
3. **Vectorization** — TF-IDF with unigrams + bigrams (captures phrases like "cannot access", not just single words).
4. **Two models trained separately:**
   - **Category classifier** — predicts Billing / Technical / Account / Product (Logistic Regression).
   - **Priority classifier** — predicts high / medium / low, with `class_weight="balanced"` since high-priority tickets are naturally rarer.
5. **Evaluation** — accuracy, precision, recall, F1 per class, plus confusion matrices for both models.
6. **`classify_ticket()` function** — ready to classify any new incoming ticket text, demoed on 3 unseen examples at the end of the script.

## 📊 Results (on this run)
- Category accuracy: ~100% (categories use very distinct vocabulary, so this is expected)
- Priority accuracy: ~63% (priority is inherently harder — it depends on subtle urgency cues in phrasing, not just topic)

## ▶️ How to Run
```bash
pip install pandas numpy scikit-learn matplotlib
python generate_data.py           # creates data/support_tickets.csv
python ticket_classification.py   # trains both models, prints reports, saves outputs
```

## 🔄 Using Real Data (Recommended for Submission)
This repo ships with a synthetic dataset so the pipeline runs out of the box. For your actual submission, swap in a real dataset:

1. Download the [Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset) from Kaggle (free account required).
2. Save the downloaded file as `data/customer_support_tickets.csv`.
3. Run the adapter, then the pipeline as normal:
   ```bash
   python adapt_real_data.py       # combines subject+description into ticket_text, reshapes columns
   python ticket_classification.py
   ```
Note: the real dataset includes a 4th priority level ("Critical") beyond low/medium/high — the pipeline already handles any number of priority classes automatically.

## 📈 Deliverable
A ticket classification system (`classify_ticket()`) that helps support teams auto-route and prioritize incoming tickets, with evaluation visuals in `classification_output.png`.
