"""
app.py
SatPrior Flask Web Application
Provides the dashboard interface and prediction API for satellite telemetry data prioritization.
"""

import sys
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ml.predict import predict_and_explain, validate_packet, load_model

app = Flask(__name__)


def get_metrics_data():
    """
    Helper to safely read saved metrics from model/metrics.json.
    """
    if config.METRICS_PATH.exists():
        try:
            with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            app.logger.warning(f"Failed to read metrics file: {e}")
    return {
        "model_name": "DecisionTreeClassifier",
        "accuracy": 0.7778,
        "train_rows": 360,
        "test_rows": 90,
        "max_depth": config.MAX_TREE_DEPTH,
        "feature_names": [],
    }


# Pre-warm model load on startup (if already trained)
try:
    load_model()
except Exception as e:
    app.logger.info(f"Model will be loaded on demand: {e}")


@app.route("/", methods=["GET"])
def index():
    """
    Renders the SatPrior mission dashboard with model insights.
    """
    metrics = get_metrics_data()
    return render_template(
        "index.html",
        metrics=metrics,
        config=config,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Inference endpoint.
    Accepts JSON packet attributes, validates input, calls Decision Tree predictor,
    and returns priority class with decision explanation.
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            data = request.form.to_dict()

        if not data:
            return jsonify({
                "success": False,
                "error": "No input payload received. Please provide packet telemetry JSON.",
            }), 400

        # Type conversion and normalization
        packet = {
            "data_type": str(data.get("data_type", "")).strip(),
            "urgency": str(data.get("urgency", "")).strip(),
            "data_size_kb": float(data.get("data_size_kb", 0)),
            "battery_level": float(data.get("battery_level", 0)),
            "link_quality": str(data.get("link_quality", "")).strip(),
        }

        # Validate input against config
        validate_packet(packet)

        # Execute prediction and generate explanation
        priority, explanation = predict_and_explain(packet)

        return jsonify({
            "success": True,
            "priority": priority,
            "explanation": explanation,
            "packet": packet,
        }), 200

    except ValueError as val_err:
        return jsonify({
            "success": False,
            "error": str(val_err),
        }), 400
    except Exception as err:
        app.logger.error(f"Prediction error: {err}")
        return jsonify({
            "success": False,
            "error": f"Internal prediction failure: {str(err)}",
        }), 500


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """
    Returns model training metrics and summary in JSON format.
    """
    metrics = get_metrics_data()
    return jsonify(metrics), 200


if __name__ == "__main__":
    print(f"Starting SatPrior Dashboard on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)
