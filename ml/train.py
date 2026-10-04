"""
ml/train.py  -  Trains the LinkWise Decision Tree model.

HOW TO RUN (from the main project folder, ml-miniproject/):
    python -m ml.train

WHAT THIS FILE DOES, IN 6 STEPS:
    1. LOAD     - read the 1500 satellite packets from the CSV file.
    2. ENCODE   - turn text values (like "Good" or "Safe") into numbers,
                  because a Decision Tree can only compare numbers.
    3. SPLIT    - keep 80% of rows for learning, hide 20% for testing.
    4. TRAIN    - the tree learns yes/no rules from the 80%.
    5. EVALUATE - check how well those rules work on the hidden 20%.
    6. SAVE     - store the model and its scores in files.

FILES IT CREATES:
    model/model.pkl      -> the trained model (predict.py loads this)
    model/metrics.json   -> accuracy and other scores (shown on the website)
    model/test_data.csv  -> the hidden 20% of rows (Nidhi's plots.py uses this)
"""

import json  # to save the scores as a .json file
import os    # to create the model/ folder if it does not exist

import joblib        # to save the trained model into a file (model.pkl)
import pandas as pd  # to read the CSV and work with the data as a table
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

import config  # Nidhi's shared settings: column names, allowed values, file paths


# ---------------------------------------------------------------------------
# MODEL SETTINGS
# ---------------------------------------------------------------------------

# 20% of the rows are hidden during training and used later as a "surprise exam".
TEST_SIZE = 0.2

# The tree may ask at most 5 questions in a row before giving an answer.
# Why limit it? A very deep tree memorises every single training row
# (including the 6% wrong labels) and then does badly on new data.
# This problem is called OVERFITTING. A small tree learns general rules instead.
MAX_DEPTH = 5

# Every final answer (leaf) must be based on at least 10 training rows.
# This stops the tree from making a special rule for just 1 or 2 odd rows.
MIN_SAMPLES_LEAF = 10


# ---------------------------------------------------------------------------
# ENCODING: turning words into numbers
# ---------------------------------------------------------------------------

def encode(df):
    """
    Turns the raw packet data into numbers the Decision Tree can use.

    This SAME function is also used by predict.py for new packets from the
    website, so the model always sees data in exactly the same format.

    Input : a table with the 6 feature columns (data_type, size_kb, ...)
    Output: a table with only numbers
    """
    df = df.copy()  # work on a copy so the original table is not changed

    # --- link_quality: has a natural order (Poor < Fair < Good) ---
    # So we give numbers that keep that order: Poor=0, Fair=1, Good=2.
    # Now the tree can ask questions like "is link quality better than Poor?"
    link_to_number = {name: i for i, name in enumerate(config.LINK_QUALITIES)}
    df["link_quality"] = df["link_quality"].map(link_to_number)

    # --- sat_mode: only two options ---
    # Normal=0, Safe=1. So "sat_mode = 1" means the satellite is in Safe mode.
    mode_to_number = {name: i for i, name in enumerate(config.SAT_MODES)}
    df["sat_mode"] = df["sat_mode"].map(mode_to_number)

    # --- data_type: 4 options with NO natural order ---
    # We cannot say Fault alert=0, Housekeeping=1 ... because that would wrongly
    # suggest "Housekeeping is bigger than Fault alert".
    # Instead we use ONE-HOT ENCODING: one yes/no column per type.
    #   A Fault alert packet becomes:
    #     data_type_Fault alert=1, data_type_Housekeeping=0,
    #     data_type_SSTV image=0,  data_type_Voice/Data=0
    # Listing all 4 categories makes sure all 4 columns are always created,
    # even when we encode just 1 packet from the website.
    df["data_type"] = pd.Categorical(df["data_type"], categories=config.DATA_TYPES)

    # Safety check: if any value was not in the allowed list, it became empty (NaN).
    # Stop here with a clear error instead of making a wrong prediction.
    if df[["link_quality", "sat_mode", "data_type"]].isna().any().any():
        raise ValueError("Unknown value in data_type, link_quality or sat_mode")

    df = pd.get_dummies(df, columns=["data_type"], dtype=int)

    # size_kb, battery_pct and pass_time_min are already numbers: kept as they are.
    return df


# ---------------------------------------------------------------------------
# MAIN TRAINING STEPS
# ---------------------------------------------------------------------------

def main():
    # ---- STEP 1: LOAD ----
    df = pd.read_csv(config.DATA_PATH)
    print(f"Loaded {len(df)} packets from {config.DATA_PATH}")

    # X = the inputs (what the model looks at)
    # y = the correct answers (what the model must learn to predict)
    X = encode(df[config.FEATURES])
    y = df[config.TARGET]
    print(f"After encoding, the model sees {X.shape[1]} number columns:")
    print("   ", list(X.columns))

    # ---- STEP 3: SPLIT ----
    # stratify=y keeps the same mix of High/Medium/Low in both parts,
    #   so the test is fair.
    # random_state fixes the "randomness", so we get the same split every run.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )
    print(f"\nTraining rows: {len(X_train)}   |   Hidden test rows: {len(X_test)}")

    # ---- STEP 4: TRAIN ----
    # The tree looks at the training rows and repeatedly picks the yes/no question
    # that best separates High, Medium and Low (it measures this using "Gini
    # impurity": how mixed the answers are in each group; lower = purer).
    model = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=config.RANDOM_SEED,
    )
    model.fit(X_train, y_train)  # <- this one line is the actual "learning"

    # ---- STEP 5: EVALUATE ----
    # Accuracy on data it has SEEN vs data it has NEVER seen.
    # If train accuracy is much higher than test accuracy, it memorised (overfitting).
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    y_pred = model.predict(X_test)  # the model's answers for the hidden 300 rows
    test_accuracy = accuracy_score(y_test, y_pred)

    # Precision, recall and F1 for each priority:
    #   precision - when the model says "High", how often is it really High?
    #   recall    - out of all the real High packets, how many did it catch?
    #   f1        - one number that balances precision and recall
    report = classification_report(
        y_test, y_pred, labels=config.PRIORITIES, output_dict=True
    )

    # Confusion matrix: a 3x3 table. Rows = real answer, columns = model's answer.
    # Numbers on the diagonal are correct; everything else is a mistake.
    matrix = confusion_matrix(y_test, y_pred, labels=config.PRIORITIES)

    # Feature importance: how much each column helped the tree decide (adds up to 1).
    importance = sorted(
        zip(X.columns, model.feature_importances_),
        key=lambda pair: pair[1],
        reverse=True,
    )

    # ---- STEP 6: SAVE ----
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

    # Save the model together with the exact column order it was trained on.
    # predict.py needs that order to feed new packets in correctly.
    joblib.dump(
        {"model": model, "feature_names": list(X.columns)},
        config.MODEL_PATH,
    )

    # Save all the scores. float()/int() turn numpy numbers into plain Python
    # numbers so they can be written to JSON.
    metrics = {
        "train_accuracy": round(float(train_accuracy), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "tree_depth": int(model.get_depth()),
        "num_leaves": int(model.get_n_leaves()),
        "labels": config.PRIORITIES,
        "per_class": {
            label: {
                "precision": round(float(report[label]["precision"]), 4),
                "recall": round(float(report[label]["recall"]), 4),
                "f1": round(float(report[label]["f1-score"]), 4),
                "support": int(report[label]["support"]),  # how many test rows have this label
            }
            for label in config.PRIORITIES
        },
        "confusion_matrix": matrix.tolist(),
        "feature_importance": {name: round(float(v), 4) for name, v in importance},
    }
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save the hidden test rows (already encoded, plus the real answer)
    # so Nidhi can draw the confusion matrix and other charts.
    test_df = X_test.copy()
    test_df[config.TARGET] = y_test
    test_df.to_csv(config.TEST_DATA_PATH, index=False)

    # ---- PRINT A SUMMARY ----
    print("\n================ RESULTS ================")
    print(f"Train accuracy : {train_accuracy:.1%}")
    print(f"Test accuracy  : {test_accuracy:.1%}   <- the number that matters")
    print(f"Tree depth     : {model.get_depth()}   |   Leaves (final answers): {model.get_n_leaves()}")

    print("\nConfusion matrix (rows = real, columns = predicted):")
    print("           " + "  ".join(f"{p:>7}" for p in config.PRIORITIES))
    for label, row in zip(config.PRIORITIES, matrix):
        print(f"{label:>9}  " + "  ".join(f"{n:>7}" for n in row))

    print("\nMost important features:")
    for name, value in importance:
        if value > 0:
            print(f"   {name:<28} {value:.3f}")

    print("\nThe rules the tree learned:")
    print(export_text(model, feature_names=list(X.columns)))

    print(f"Saved: {config.MODEL_PATH}, {config.METRICS_PATH}, {config.TEST_DATA_PATH}")


# This makes sure training only runs when you run this file directly,
# and NOT when predict.py imports the encode() function from here.
if __name__ == "__main__":
    main()