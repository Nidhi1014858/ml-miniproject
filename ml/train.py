"""
ml/train.py
Trains the Decision Tree Classifier on satellite telemetry data,
evaluates performance, saves model artifacts, and generates visualization plots.
"""

import sys
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib

# Set non-interactive backend before importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def train_model():
    print("=" * 60)
    print("SatPrior - Model Training Pipeline (Decision Tree Classifier)")
    print("=" * 60)

    # 1. Verify dataset exists
    if not config.DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {config.DATASET_PATH}. Please run data/generate_data.py first."
        )

    # 2. Load dataset
    print(f"\n[1/6] Loading dataset from: {config.DATASET_PATH}")
    df = pd.read_csv(config.DATASET_PATH)

    # 3. Separate features and target
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )

    print(f"[2/6] Data split complete:")
    print(f"      - Training records: {len(X_train)} rows")
    print(f"      - Testing records : {len(X_test)} rows")

    # 5. Build preprocessing and model pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    categories=[
                        config.ALLOWED_DATA_TYPES,
                        config.ALLOWED_URGENCY_LEVELS,
                        config.ALLOWED_LINK_QUALITIES,
                    ],
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                config.CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", config.NUMERICAL_FEATURES),
        ]
    )

    clf = DecisionTreeClassifier(
        max_depth=config.MAX_TREE_DEPTH,
        random_state=config.RANDOM_SEED,
        criterion=config.CRITERION,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    # 6. Fit the pipeline
    print(f"[3/6] Training DecisionTreeClassifier (max_depth={config.MAX_TREE_DEPTH})...")
    pipeline.fit(X_train, y_train)

    # Extract feature names after encoding
    cat_feature_names = pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(config.CATEGORICAL_FEATURES)
    all_feature_names = list(cat_feature_names) + config.NUMERICAL_FEATURES

    # 7. Evaluate model
    print("[4/6] Evaluating model on testing split...")
    y_pred = pipeline.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)

    print(f"\nModel Performance Metrics:")
    print(f"--------------------------------------------------")
    print(f"Training Rows : {len(X_train)}")
    print(f"Testing Rows  : {len(X_test)}")
    print(f"Accuracy      : {accuracy * 100:.2f}%")
    print(f"\nClassification Report:\n{report_text}")
    print(f"Confusion Matrix (labels={list(pipeline.classes_)}):\n{cm}")
    print(f"--------------------------------------------------")

    # 8. Save Model and Metrics
    print("[5/6] Saving model and metrics artifacts...")
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, config.MODEL_PATH)
    print(f"      [OK] Saved model to: {config.MODEL_PATH}")

    metrics_data = {
        "model_name": "DecisionTreeClassifier",
        "accuracy": round(accuracy, 4),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "max_depth": config.MAX_TREE_DEPTH,
        "feature_names": all_feature_names,
        "classes": list(pipeline.classes_),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }

    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"      [OK] Saved metrics to: {config.METRICS_PATH}")

    # 9. Generate and save visualization plots
    print("[6/6] Generating and saving evaluation plots...")

    # Plot 1: Decision Tree Visualization
    plt.figure(figsize=(20, 10), facecolor="#ffffff")
    plot_tree(
        clf,
        feature_names=all_feature_names,
        class_names=[str(c) for c in clf.classes_],
        filled=True,
        rounded=True,
        precision=2,
        fontsize=9,
    )
    plt.title("SatPrior — Decision Tree Hierarchy (SomaiyaSat Telemetry)", fontsize=16, pad=15)
    plt.tight_layout()
    plt.savefig(config.TREE_PLOT_PATH, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"      [OK] Saved decision tree plot: {config.TREE_PLOT_PATH}")

    # Plot 2: Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#ffffff")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True, values_format="d")
    ax.set_title("SatPrior — Test Set Confusion Matrix", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(config.CONFUSION_PLOT_PATH, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"      [OK] Saved confusion matrix plot: {config.CONFUSION_PLOT_PATH}")

    # Plot 3: Feature Importance
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="#ffffff")
    bars = ax.barh(
        range(len(sorted_idx)),
        importances[sorted_idx],
        color="#0284c7",
        edgecolor="#0369a1",
        alpha=0.85,
    )
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([all_feature_names[i] for i in sorted_idx], fontsize=9)
    ax.set_xlabel("Gini Impurity Reduction (Feature Importance)", fontsize=10)
    ax.set_title("SatPrior — Feature Importance Ranking", fontsize=13, pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Add numeric labels to bars
    for bar in bars:
        width = bar.get_width()
        if width > 0.005:
            ax.text(
                width + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                va="center",
                fontsize=8,
                color="#0f172a",
            )

    plt.tight_layout()
    plt.savefig(config.IMPORTANCE_PLOT_PATH, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"      [OK] Saved feature importance plot: {config.IMPORTANCE_PLOT_PATH}")

    print("\n[SUCCESS] Pipeline training and evaluation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
