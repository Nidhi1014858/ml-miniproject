"""
SatPrior - Configuration Module
Autonomous Satellite Data Prioritization Using Decision Tree Classification.

This file provides centralized paths, feature specifications, allowed values,
and hyperparameter settings to maintain consistency across the project.
"""

from pathlib import Path

# ==============================================================================
# Base Paths
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

# Data paths
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "satellite_data.csv"

# Model artifact paths
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Static / plot paths
STATIC_DIR = BASE_DIR / "static"
PLOTS_DIR = STATIC_DIR / "plots"
TREE_PLOT_PATH = PLOTS_DIR / "tree.png"
CONFUSION_PLOT_PATH = PLOTS_DIR / "confusion.png"
IMPORTANCE_PLOT_PATH = PLOTS_DIR / "importance.png"

# ==============================================================================
# Feature & Target Definitions
# ==============================================================================
CATEGORICAL_FEATURES = ["data_type", "urgency", "link_quality"]
NUMERICAL_FEATURES = ["data_size_kb", "battery_level"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

TARGET_COLUMN = "priority"

# Allowed values for categorical features
ALLOWED_DATA_TYPES = ["TT&C", "Housekeeping", "SSTV", "Voice/Data"]
ALLOWED_URGENCY_LEVELS = ["Low", "Medium", "High"]
ALLOWED_LINK_QUALITIES = ["Poor", "Fair", "Good"]

# Target priority classes (ordered from lowest to highest urgency)
PRIORITY_CLASSES = ["Low", "Medium", "High"]

ALLOWED_VALUES = {
    "data_type": ALLOWED_DATA_TYPES,
    "urgency": ALLOWED_URGENCY_LEVELS,
    "link_quality": ALLOWED_LINK_QUALITIES,
    "priority": PRIORITY_CLASSES,
}

# Numerical feature constraints for validation
NUMERICAL_BOUNDS = {
    "data_size_kb": {"min": 1, "max": 10000},
    "battery_level": {"min": 0, "max": 100},
}

# ==============================================================================
# Model Hyperparameters & Training Settings
# ==============================================================================
RANDOM_SEED = 42
TEST_SIZE = 0.20
MAX_TREE_DEPTH = 5
CRITERION = "gini"
