"""
FUTURE_ML_02 - Support Ticket Classification

Live terminal demo:
Type your own ticket and get predicted Category + Priority.

Outputs:
1. classification_output.png
   -> Confusion matrices for Category and Priority
   -> Created/updated, but NOT opened automatically

2. demo_predictions.csv
   -> Queries entered in the current session

3. demo_predictions.png
   -> PNG table containing ONLY the current session queries
   -> Automatically opened after typing quit
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# =========================================================
# 1. LOAD DATA
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "support_tickets.csv"
)

CLASSIFICATION_PNG = os.path.join(
    BASE_DIR,
    "classification_output.png"
)

DEMO_CSV = os.path.join(
    BASE_DIR,
    "demo_predictions.csv"
)

DEMO_PNG = os.path.join(
    BASE_DIR,
    "demo_predictions.png"
)


print("=" * 65)
print("       FUTURE_ML_02 - SUPPORT TICKET CLASSIFICATION")
print("=" * 65)


if not os.path.exists(DATA_PATH):

    print("\nERROR: Dataset not found!")
    print("Expected:")
    print(DATA_PATH)

    input("\nPress Enter to exit...")
    raise SystemExit


df = pd.read_csv(DATA_PATH)


required_columns = [
    "Ticket Description",
    "Ticket Type",
    "Ticket Priority"
]


for column in required_columns:

    if column not in df.columns:

        print(
            f"\nERROR: Missing column: {column}"
        )

        input("\nPress Enter to exit...")
        raise SystemExit


# Remove missing values
df = df.dropna(
    subset=required_columns
)


# Remove duplicate tickets
df = df.drop_duplicates(
    subset=["Ticket Description"]
).reset_index(drop=True)


print(
    f"\nTotal tickets: {len(df)}"
)


print("\nTicket Categories:")
print(
    df["Ticket Type"].value_counts()
)


print("\nTicket Priorities:")
print(
    df["Ticket Priority"].value_counts()
)


# =========================================================
# 2. PREPARE DATA
# =========================================================

X_text = df["Ticket Description"]

y_category = df["Ticket Type"]

y_priority = df["Ticket Priority"]


# =========================================================
# 3. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, ycat_train, ycat_test, ypri_train, ypri_test = train_test_split(

    X_text,
    y_category,
    y_priority,

    test_size=0.20,

    random_state=42,

    stratify=y_category
)


print(
    "\nTraining tickets:",
    len(X_train)
)

print(
    "Testing tickets :",
    len(X_test)
)


# =========================================================
# 4. TF-IDF
# =========================================================

print("\nTraining TF-IDF model...")


vectorizer = TfidfVectorizer(

    stop_words="english",

    ngram_range=(1, 2),

    min_df=1
)


X_train_vec = vectorizer.fit_transform(
    X_train
)

X_test_vec = vectorizer.transform(
    X_test
)


# =========================================================
# 5. CATEGORY MODEL
# =========================================================

print("Training category model...")


category_model = LogisticRegression(

    max_iter=2000,

    class_weight="balanced"
)


category_model.fit(

    X_train_vec,

    ycat_train
)


# =========================================================
# 6. PRIORITY MODEL
# =========================================================

print("Training priority model...")


priority_model = LogisticRegression(

    max_iter=2000,

    class_weight="balanced"
)


priority_model.fit(

    X_train_vec,

    ypri_train
)


# =========================================================
# 7. TEST MODEL
# =========================================================

cat_pred = category_model.predict(
    X_test_vec
)

pri_pred = priority_model.predict(
    X_test_vec
)


cat_acc = category_model.score(
    X_test_vec,
    ycat_test
)

pri_acc = priority_model.score(
    X_test_vec,
    ypri_test
)


print("\n" + "=" * 65)
print("                    MODEL PERFORMANCE")
print("=" * 65)


print(
    f"\nCategory accuracy : {cat_acc:.2%}"
)

print(
    f"Priority accuracy : {pri_acc:.2%}"
)


# =========================================================
# 8. CLASSIFICATION REPORT
# =========================================================

print("\n" + "=" * 65)
print("                 CATEGORY REPORT")
print("=" * 65)


print(
    classification_report(
        ycat_test,
        cat_pred,
        zero_division=0
    )
)


print("=" * 65)
print("                 PRIORITY REPORT")
print("=" * 65)


print(
    classification_report(
        ypri_test,
        pri_pred,
        zero_division=0
    )
)


# =========================================================
# 9. CREATE / UPDATE CLASSIFICATION OUTPUT
# =========================================================

print("\nCreating classification_output.png...")


category_labels = sorted(
    category_model.classes_
)


priority_labels = sorted(
    priority_model.classes_
)


cm_category = confusion_matrix(

    ycat_test,

    cat_pred,

    labels=category_labels
)


cm_priority = confusion_matrix(

    ypri_test,

    pri_pred,

    labels=priority_labels
)


fig, axes = plt.subplots(

    1,

    2,

    figsize=(18, 8)
)


# ---------------------------------------------------------
# CATEGORY CONFUSION MATRIX
# ---------------------------------------------------------

sns.heatmap(

    cm_category,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=category_labels,

    yticklabels=category_labels,

    ax=axes[0]
)


axes[0].set_title(
    "Ticket Category Confusion Matrix"
)

axes[0].set_xlabel(
    "Predicted Category"
)

axes[0].set_ylabel(
    "Actual Category"
)

axes[0].tick_params(
    axis="x",
    rotation=45
)


# ---------------------------------------------------------
# PRIORITY CONFUSION MATRIX
# ---------------------------------------------------------

sns.heatmap(

    cm_priority,

    annot=True,

    fmt="d",

    cmap="Oranges",

    xticklabels=priority_labels,

    yticklabels=priority_labels,

    ax=axes[1]
)


axes[1].set_title(
    "Ticket Priority Confusion Matrix"
)

axes[1].set_xlabel(
    "Predicted Priority"
)

axes[1].set_ylabel(
    "Actual Priority"
)

axes[1].tick_params(
    axis="x",

    rotation=45
)


plt.tight_layout()


# Replace old classification_output.png
fig.savefig(

    CLASSIFICATION_PNG,

    dpi=100,

    bbox_inches="tight"
)


plt.close(fig)


print(
    "Updated:",
    CLASSIFICATION_PNG
)


# IMPORTANT:
# classification_output.png is NOT opened.
# It only gets created/updated.


# =========================================================
# 10. LIVE CLASSIFICATION FUNCTION
# =========================================================

def classify(ticket_text):

    X_new = vectorizer.transform(
        [ticket_text]
    )


    category = category_model.predict(
        X_new
    )[0]


    priority = priority_model.predict(
        X_new
    )[0]


    return category, priority


# =========================================================
# 11. CREATE CURRENT SESSION PNG
# =========================================================

def create_demo_png(result_df):

    # Close old matplotlib figures
    plt.close("all")


    # Delete previous demo PNG
    if os.path.exists(DEMO_PNG):

        try:

            os.remove(DEMO_PNG)

        except PermissionError:

            print(
                "\nERROR: Please close the previous "
                "demo_predictions.png and run again."
            )

            return False


    rows = len(result_df)


    fig_height = max(
        5,
        rows * 0.75 + 2
    )


    fig, ax = plt.subplots(

        figsize=(16, fig_height),

        dpi=100
    )


    ax.axis("off")


    table = ax.table(

        cellText=result_df.values,

        colLabels=result_df.columns,

        cellLoc="left",

        loc="center",

        colWidths=[
            0.55,
            0.22,
            0.18
        ]
    )


    table.auto_set_font_size(False)

    table.set_fontsize(10)

    table.scale(
        1,
        2
    )


    ax.set_title(

        "FUTURE_ML_02 - Current Session Predictions",

        fontsize=15,

        fontweight="bold",

        pad=20
    )


    plt.tight_layout()


    # Replace old demo_predictions.png
    fig.savefig(

        DEMO_PNG,

        dpi=100,

        bbox_inches="tight"
    )


    plt.close(fig)


    return True


# =========================================================
# 12. LIVE TERMINAL
# =========================================================

if __name__ == "__main__":

    log = []


    print("\n" + "=" * 65)
    print("             LIVE SUPPORT TICKET CLASSIFIER")
    print("=" * 65)


    print(
        "\nEnter your support queries one by one."
    )


    print(
        "The ML model will predict Category + Priority."
    )


    print(
        "Type 'quit' when finished.\n"
    )


    while True:

        message = input(
            "Enter ticket message: "
        ).strip()


        # -------------------------------------------------
        # QUIT
        # -------------------------------------------------

        if message.lower() in (
            "quit",
            "exit"
        ):

            break


        # -------------------------------------------------
        # EMPTY INPUT
        # -------------------------------------------------

        if not message:

            print(
                "Please enter a ticket message.\n"
            )

            continue


        # -------------------------------------------------
        # ML PREDICTION
        # -------------------------------------------------

        category, priority = classify(
            message
        )


        print("\n------------------------------")

        print(
            "YOUR QUERY:"
        )

        print(
            message
        )

        print()

        print(
            "PREDICTED CATEGORY :",
            category
        )

        print(
            "PREDICTED PRIORITY :",
            priority
        )

        print(
            "------------------------------\n"
        )


        # -------------------------------------------------
        # SAVE CURRENT QUERY
        # -------------------------------------------------

        log.append(

            {
                "Ticket Description": message,

                "Predicted Category": category,

                "Predicted Priority": priority
            }
        )


    # =====================================================
    # 13. SAVE CURRENT SESSION
    # =====================================================

    if log:

        result_df = pd.DataFrame(
            log
        )


        # -------------------------------------------------
        # REPLACE PREVIOUS CSV
        # -------------------------------------------------

        result_df.to_csv(

            DEMO_CSV,

            index=False
        )


        # -------------------------------------------------
        # CREATE CURRENT PNG
        # -------------------------------------------------

        print(
            "\nCreating current demo_predictions.png..."
        )


        success = create_demo_png(
            result_df
        )


        if success:

            print("\n" + "=" * 65)
            print("              CURRENT RUN COMPLETED")
            print("=" * 65)


            print(
                "\nUpdated:",
                DEMO_CSV
            )


            print(
                "Updated:",
                DEMO_PNG
            )


            # -------------------------------------------------
            # OPEN ONLY CURRENT DEMO PNG
            # -------------------------------------------------

            if os.name == "nt":

                try:

                    os.startfile(
                        DEMO_PNG
                    )

                    print(
                        "\nCurrent demo_predictions.png opened."
                    )

                except Exception:

                    print(
                        "\nPNG created successfully."
                    )

                    print(
                        "Open manually:",
                        DEMO_PNG
                    )


            print(
                "\nCurrent session predictions:"
            )

            print(
                result_df.to_string(
                    index=False
                )
            )


            print(
                "\n" + "=" * 65
            )

            print(
                "DONE"
            )

            print(
                "=" * 65
            )


    else:

        print(
            "\nNo ticket predictions were entered."
        )


# =========================================================
# IMPORTANT
# =========================================================
#
# classification_output.png:
#     CREATED/UPDATED
#     NOT OPENED
#
# demo_predictions.png:
#     CREATED/REPLACED
#     CONTAINS ONLY CURRENT RUN QUERIES
#     OPENED AUTOMATICALLY AFTER "quit"
#
# demo_predictions.csv:
#     REPLACED WITH CURRENT RUN QUERIES
#
# =========================================================