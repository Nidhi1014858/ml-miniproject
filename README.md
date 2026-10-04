# LinkWise

Autonomous Satellite Downlink Packet Prioritization using Decision Tree Classification.

---

## What LinkWise Does

During brief satellite communication passes (0.5 to 12 minutes), bandwidth and battery power are limited. **LinkWise** uses a Decision Tree Classifier to prioritize packets into **High**, **Medium**, or **Low** downlink priority so critical satellite data reaches ground station operators first.

---

## Telemetry Inputs

LinkWise evaluates 6 packet attributes:

1. **`data_type`**: Telemetry category — `Fault alert`, `Housekeeping`, `SSTV image`, or `Voice/Data`.
2. **`size_kb`**: Data volume in kilobytes (1 to 800 KB).
3. **`battery_pct`**: Satellite state-of-charge percentage (10 to 100%).
4. **`link_quality`**: Radio frequency connection state — `Poor`, `Fair`, or `Good`.
5. **`pass_time_min`**: Contact window remaining before the ground station goes out of view (0.5 to 12.0 minutes).
6. **`sat_mode`**: Spacecraft operating state — `Normal` or `Safe`.

---

## How Labels Were Made

The synthetic training dataset (1,500 rows) is generated using an operational mission-scoring rule:

1. **Base Score**: `Fault alert` (75), `Housekeeping` (50), `SSTV image` (40), `Voice/Data` (30).
2. **Safe Mode**: +20 boost for critical health and alerts; -20 penalty for non-essential payloads.
3. **Low Battery**: +10 boost for Housekeeping when battery is below 30%; -15 penalty for SSTV and Voice/Data below 25%.
4. **RF Link**: -10 penalty for Poor link, +5 boost for Good link.
5. **Pass Feasibility**: -25 penalty if transmission cannot finish during the pass window (speeds: 1 KB/s Poor, 3 KB/s Fair, 6 KB/s Good).
6. **Priority Cutoffs**: Score $\ge$ 60 $\rightarrow$ **High**, 35–59 $\rightarrow$ **Medium**, < 35 $\rightarrow$ **Low**.

### Why 6% Label Noise Is Added
Real-world satellite operations involve operator overrides, queue adjustments, and atmospheric interference. Adding 6% random label noise prevents the Decision Tree from memorizing deterministic thresholds and reflects realistic flight conditions with an expected ~90% accuracy.

---

## How to Run

Run these commands in order from the project root:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate simulated satellite dataset (1,500 rows)
python data/generate_data.py

# 3. Train Decision Tree and evaluate metrics
python ml/train.py

# 4. Generate diagnostic visualization charts
python ml/plots.py

# 5. Start the web application
python app.py
```

Then open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## Project Structure

```text
ml-miniproject/
├── README.md               # Project guide and overview
├── config.py               # Shared constants, paths, and allowed values
├── app.py                  # Flask web server and /predict route
├── requirements.txt        # Python package dependencies
├── data/
│   ├── DATASET.md          # Dataset specifications and scoring rules
│   ├── generate_data.py    # Synthetic telemetry data generator
│   ├── satellite_data.csv  # Full dataset (1,500 rows)
│   └── sample_data.csv     # Sample dataset (50 rows)
├── ml/
│   ├── train.py            # Model training and evaluation
│   ├── predict.py          # Prediction logic and decision path explainer
│   └── plots.py            # Tree, confusion matrix, and importance plots
├── model/
│   ├── model.pkl           # Saved Decision Tree model artifact
│   ├── metrics.json        # Test accuracy, depth, and split counts
│   └── test_data.csv       # Holdout test set records
├── templates/
│   ├── index.html          # Main web dashboard interface
│   └── _insights.html      # Model metrics and diagnostic charts section
└── static/
    ├── css/style.css       # Space-themed responsive stylesheet
    ├── js/main.js          # Interactive frontend and fetch API logic
    └── plots/              # Exported evaluation charts
        ├── tree.png
        ├── confusion.png
        └── importance.png
```

---

## Team

- **Nidhi**: Data generation and visual plots
- **Sadhana**: Machine learning model and backend
- **Apoorva**: Web frontend and UI design
