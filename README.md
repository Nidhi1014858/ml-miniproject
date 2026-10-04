# LinkWise: Autonomous Satellite Data Prioritization Using Decision Tree Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![ML](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

> **SomaiyaSat ML Lab Mini-Project (Semester 5)**  
> An autonomous machine learning system that classifies incoming satellite telemetry and payload packets into **High**, **Medium**, or **Low** transmission priorities using a **Decision Tree Classifier**.

---

## 1. Project Overview

In small satellite operations such as SomaiyaSat, communication pass windows with ground stations are brief (typically 8–12 minutes per pass), and on-board electrical power and downlink bandwidth are severely constrained. Downlinking bulky non-critical payloads during low-elevation passes or low-battery states risks packet drops, frame corruption, and satellite brownout.

**LinkWise** solves this scheduling challenge through transparent, interpretable machine learning:
- Ingests telemetry packet parameters (**Data Type**, **Urgency**, **Data Size**, **Battery Level**, and **Link Quality**).
- Applies a pruned **Decision Tree Classifier** to classify transmission priority into **High**, **Medium**, or **Low**.
- Traces the exact node traversal through the decision tree to generate human-readable explanations (e.g., *"High urgency flag and compact packet size guided the decision tree to assign High transmission priority"*).
- Displays comprehensive model metrics, confusion matrix, feature importance, and tree visual diagrams on an interactive dashboard.

---

## 2. Key Features

- **Domain-Realistic Telemetry Simulation**: Generates 450 synthetic satellite packets mirroring operational CubeSat constraints (TT&C commands, housekeeping telemetry, SSTV images, and amateur voice/data).
- **Interpretable Decision Tree Classifier**: Uses a `DecisionTreeClassifier` with tuned depth (`max_depth=5`) to balance high accuracy (~78–82%) with human interpretability.
- **Rule-Based Decision Path Explanation**: Traces the actual tree split nodes to explain *why* a priority was assigned, without relying on LLMs or external APIs.
- **Interactive Flask Dashboard**: A responsive, space-themed mission control interface featuring real-time priority badges, loading radar indicators, and random packet synthesis.
- **Embedded Model Insights**: Visualizes the decision tree architecture (`tree.png`), test confusion matrix (`confusion.png`), and Gini feature importance ranking (`importance.png`).
- **Clean Architecture & Centralized Config**: All file paths, feature types, allowed values, and random seeds (`42`) are maintained in `config.py`.

---

## 3. Technology Stack

- **Core Language**: Python 3.10+
- **Web Framework**: Flask
- **Machine Learning**: scikit-learn (`DecisionTreeClassifier`, `ColumnTransformer`, `OneHotEncoder`)
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib (non-interactive Agg backend)
- **Frontend**: Semantic HTML5, Vanilla CSS3 (orbital space theme), Vanilla JavaScript (ES6+ `fetch` API)

---

## 4. Project Directory Structure

```text
linkwise/
├── README.md                 # Project documentation and user guide
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── config.py                 # Central configuration and hyperparameters
│
├── data/
│   ├── generate_data.py      # Synthetic satellite dataset generator
│   └── satellite_data.csv    # Generated telemetry dataset (450 rows)
│
├── ml/
│   ├── train.py              # ML training, evaluation, and plot generation
│   └── predict.py            # Prediction pipeline and decision path explainer
│
├── model/
│   ├── model.pkl             # Serialized trained scikit-learn pipeline
│   └── metrics.json          # Evaluation metrics (accuracy, splits, reports)
│
├── app.py                    # Flask application and REST routes
│
├── templates/
│   ├── index.html            # Main mission control dashboard
│   └── _insights.html        # Reusable model insights and evaluation component
│
└── static/
    ├── css/
    │   └── style.css         # Space-themed responsive dashboard stylesheet
    ├── js/
    │   └── main.js           # Interactive form submission and random packet logic
    └── plots/
        ├── tree.png          # Visual decision tree diagram
        ├── confusion.png     # Test set confusion matrix
        └── importance.png    # Gini feature importance bar chart
```

---

## 5. How the ML Pipeline Works

```text
+-------------------------+
|  satellite_data.csv     |
+-------------------------+
             |
             v
+-------------------------------------------------------+
|  ColumnTransformer (OneHotEncoder + Passthrough)      |
|  - data_type: [TT&C, Housekeeping, SSTV, Voice/Data]  |
|  - urgency: [Low, Medium, High]                       |
|  - link_quality: [Poor, Fair, Good]                   |
|  - Numerical: data_size_kb, battery_level             |
+-------------------------------------------------------+
             |
             v
+-------------------------------------------------------+
|  DecisionTreeClassifier (max_depth=5, seed=42)       |
+-------------------------------------------------------+
             |
             v
+-------------------------------------------------------+
|  1. Evaluation: Accuracy, Precision, Recall, F1       |
|  2. Serialization: model/model.pkl & metrics.json     |
|  3. Plot Exports: tree.png, confusion.png, importance |
+-------------------------------------------------------+
```

1. **Preprocessing**: Categorical features are encoded using `OneHotEncoder` with fixed categories, while numerical features (`data_size_kb`, `battery_level`) pass through directly.
2. **Training**: The pipeline splits data 80/20 with stratification on `priority`, training a tree capped at `max_depth=5` to prevent overfitting.
3. **Inference**: Given a new packet dictionary, `ml/predict.py` executes the scikit-learn pipeline and inspects `clf.decision_path(X)` to identify the primary split conditions that guided the packet to its leaf node.

---

## 6. Installation & Setup

### Step 1: Clone or Open the Workspace
```bash
git clone https://github.com/Nidhi1014858/ml-miniproject.git
cd ml-miniproject
```

### Step 2: Set Up Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Packages
```bash
pip install -r requirements.txt
```

---

## 7. Execution Guide

Run the following three commands in sequence:

### 1. Generate the Telemetry Dataset
```bash
python data/generate_data.py
```
*Output: Synthesizes `data/satellite_data.csv` (450 rows) with realistic priority distributions.*

### 2. Train and Evaluate the Decision Tree
```bash
python ml/train.py
```
*Output: Evaluates model performance, saves `model/model.pkl` and `model/metrics.json`, and exports all three plots to `static/plots/`.*

### 3. Launch the Flask Web Application
```bash
python app.py
```

### 4. Open in Browser
Navigate to:
```text
http://127.0.0.1:5000
```

---

## 8. Example Prediction

### Sample Packet Telemetry:
| Field | Value | Reason |
|---|---|---|
| **Data Type** | `TT&C` | Critical Telemetry & Command packet |
| **Urgency** | `High` | Immediate flight maneuver instruction |
| **Data Size** | `25 KB` | Compact frame |
| **Battery Level** | `85%` | Sufficient satellite power |
| **Link Quality** | `Good` | Clear zenith ground pass |

### Output:
- **Predicted Priority**: <span style="color:#ef4444; font-weight:bold;">HIGH</span>
- **Decision Path Explanation**:
  > *"High urgency flag and compact packet size (25 KB <= 411 KB) guided the decision tree to assign High transmission priority for prompt downlink."*

---

## 9. Future Scope

1. **Dynamic Real-Time Satellite Telemetry Ingestion**: Connect to actual software-defined radio (SDR) or CubeSat ground station feeds via MQTT or WebSockets.
2. **Multi-Constraint Optimization**: Incorporate ground station contact horizon timers (AOS/LOS duration) and orbital eclipse windows into the decision features.
3. **Hardware Deployment**: Compile the trained decision tree rules into C/C++ header arrays for execution on resource-constrained on-board microcontrollers (e.g., STM32 / ARM Cortex-M).
4. **Ensemble Benchmarking**: Compare performance against Random Forests and Gradient Boosted Trees while preserving local decision tree explainability.
