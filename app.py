"""
app.py  -  The LinkWise web server (built with Flask).

It is the "middle person" between the web page and the ML model:

    Browser  --(packet)-->  app.py  -->  predict_packet() in ml/predict.py
    Browser  <--(answer)--  app.py  <--  priority, confidence, path

HOW TO RUN (from the main project folder, ml-miniproject/):
    python app.py
Then open  http://127.0.0.1:5000  in the browser.

ROUTES (a route = a web address the server answers):
    GET  /         -> sends the web page, filled with the model's scores
    POST /predict  -> receives one packet, returns the model's prediction as JSON
"""

import json
import os

from flask import Flask, jsonify, render_template, request

import config                          # shared settings (allowed values, file paths)
from ml.predict import predict_packet  # our prediction function

# Create the web app. Flask automatically finds the templates/ and static/ folders.
app = Flask(__name__)


def load_metrics():
    """
    Reads the model's scores from model/metrics.json (created by ml/train.py).
    Returns None if the model has not been trained yet.
    """
    if not os.path.exists(config.METRICS_PATH):
        return None

    with open(config.METRICS_PATH) as f:
        metrics = json.load(f)

    # The insights section of the page uses the names "accuracy" and "max_depth".
    # train.py saves them as "test_accuracy" and "tree_depth", so we add copies
    # under the names the page expects.
    metrics["accuracy"] = metrics["test_accuracy"]
    metrics["max_depth"] = metrics["tree_depth"]
    return metrics


# ---------------------------------------------------------------------------
# ROUTE 1: the web page
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    metrics = load_metrics()
    if metrics is None:
        # Friendly message instead of a crash if someone forgot to train the model.
        return "Model not trained yet. Run:  python -m ml.train", 503

    # render_template() fills the HTML page with these values.
    #   metrics -> accuracy, train/test rows, depth (shown in the insights section)
    #   config  -> allowed values, so the page can build its dropdowns from them
    return render_template("index.html", metrics=metrics, config=config)


# ---------------------------------------------------------------------------
# ROUTE 2: the prediction API
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    # Read the packet the browser sent (as JSON).
    # silent=True means: if it is not valid JSON, give None instead of crashing.
    packet = request.get_json(silent=True)

    try:
        result = predict_packet(packet)  # all checking + predicting happens in predict.py

    except ValueError as error:
        # Bad input (missing field, unknown value, negative number...).
        # 400 = "the request was wrong" (the user's mistake, not the server's).
        return jsonify({"success": False, "error": str(error)}), 400

    except Exception as error:
        # Anything unexpected. 500 = "something broke on the server".
        app.logger.error(f"Prediction failed: {error}")
        return jsonify({"success": False, "error": "Prediction failed on the server"}), 500

    # Success: send back priority, confidence, path, probabilities
    # plus the packet itself, so the page can show what was evaluated.
    return jsonify({"success": True, **result, "packet": packet})


# This runs only when you start the server with:  python app.py
if __name__ == "__main__":
    # debug=True: the server restarts by itself when you save a file,
    # and shows detailed errors. Fine for a project demo.
    app.run(debug=True)