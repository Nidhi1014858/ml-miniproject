# config.py - Shared configuration settings for the LinkWise project

# All columns present in the satellite telemetry dataset
COLUMNS = ["data_type", "size_kb", "battery_pct", "link_quality", "pass_time_min", "sat_mode", "priority"]

# Input feature columns used by machine learning models to make predictions
FEATURES = [col for col in COLUMNS if col != "priority"]

# Target column that the machine learning model aims to predict
TARGET = "priority"

# Allowed satellite transmission packet categories
DATA_TYPES = ["Fault alert", "Housekeeping", "SSTV image", "Voice/Data"]

# Ground station radio frequency link quality levels
LINK_QUALITIES = ["Poor", "Fair", "Good"]

# Operational modes of the satellite (Normal operations vs Safe survival mode)
SAT_MODES = ["Normal", "Safe"]

# Transmission priority classifications assigned to each packet
PRIORITIES = ["High", "Medium", "Low"]

# Fixed random seed to ensure reproducible data generation and model training
RANDOM_SEED = 42

# File path to the full synthetic satellite telemetry dataset (1500 rows)
DATA_PATH = "data/satellite_data.csv"

# File path to a small sample dataset (first 50 rows) for quick testing
SAMPLE_PATH = "data/sample_data.csv"

# File path where the trained Decision Tree model artifact will be saved
MODEL_PATH = "model/model.pkl"

# File path where model evaluation metrics and statistics are stored as JSON
METRICS_PATH = "model/metrics.json"

# File path to the holdout test dataset used for final model evaluation
TEST_DATA_PATH = "model/test_data.csv"

# Directory path where diagnostic visualization plots are saved
PLOTS_DIR = "static/plots"
