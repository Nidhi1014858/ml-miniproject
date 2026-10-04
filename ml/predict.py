"""
ml/predict.py  -  Uses the trained model to predict the priority of ONE packet.

train.py  = studying (runs once, saves the model)
predict.py = answering (runs every time someone clicks "Predict" on the website)

MAIN FUNCTION:  predict_packet(packet)

    Input  (a dictionary, e.g. from the website form):
        {"data_type": "Housekeeping", "size_kb": 40, "battery_pct": 25,
         "link_quality": "Good", "pass_time_min": 6.5, "sat_mode": "Normal"}

    Output (a dictionary that app.py sends back to the website):
        {"priority": "High",
         "confidence": 0.95,
         "path": ["Data type is not Fault alert (it is Housekeeping)", ...],
         "probabilities": {"High": 0.95, "Medium": 0.05, "Low": 0.0}}

HOW TO TEST THIS FILE ON ITS OWN (from the main project folder):
    python -m ml.predict
"""

import joblib        # to load the saved model from model.pkl
import pandas as pd  # to turn the packet into a one-row table

import config                 # shared settings (allowed values, file paths)
from ml.train import encode   # the SAME encoding used in training (very important!)


# ---------------------------------------------------------------------------
# LOADING THE MODEL
# ---------------------------------------------------------------------------

_saved = None  # the model is kept here after loading, so we load the file only once


def load_model():
    """Loads model.pkl the first time it's needed, then reuses it (faster)."""
    global _saved
    if _saved is None:
        # model.pkl contains: {"model": the tree, "feature_names": column order}
        _saved = joblib.load(config.MODEL_PATH)
    return _saved


# The 3 number fields, with a friendly name and a unit for the explanation sentences.
NUMBER_FIELDS = {
    "size_kb": ("Packet size", " KB"),
    "battery_pct": ("Battery", "%"),
    "pass_time_min": ("Pass time left", " min"),
}

# The 3 text fields, with the values they are allowed to have (from config.py).
CHOICE_FIELDS = {
    "data_type": config.DATA_TYPES,
    "link_quality": config.LINK_QUALITIES,
    "sat_mode": config.SAT_MODES,
}


# ---------------------------------------------------------------------------
# STEP 1: CHECK THE INPUT
# ---------------------------------------------------------------------------

def clean_packet(packet):
    """
    Checks that the packet is valid and returns a cleaned copy.
    If anything is wrong, it raises a ValueError with a clear message,
    which app.py sends back to the website as an error.
    """
    if not isinstance(packet, dict):
        raise ValueError("Packet must be a JSON object")

    # Every one of the 6 fields must be present and not empty.
    missing = [f for f in config.FEATURES if packet.get(f) in (None, "")]
    if missing:
        raise ValueError(f"Missing field(s): {', '.join(missing)}")

    clean = {}

    # Text fields must be one of the allowed values (e.g. link must be Poor/Fair/Good).
    for field, allowed in CHOICE_FIELDS.items():
        if packet[field] not in allowed:
            raise ValueError(f"{field} must be one of: {', '.join(allowed)}")
        clean[field] = packet[field]

    # Number fields must really be numbers and must not be negative.
    # (The website may send "35" as text, so float() converts it to the number 35.0.)
    for field in NUMBER_FIELDS:
        try:
            value = float(packet[field])
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be a number")
        if value < 0:
            raise ValueError(f"{field} cannot be negative")
        clean[field] = value

    if clean["battery_pct"] > 100:
        raise ValueError("battery_pct cannot be more than 100")

    return clean


# ---------------------------------------------------------------------------
# STEP 5: EXPLAIN THE PATH (turning the tree's questions into sentences)
# ---------------------------------------------------------------------------

def explain_path(model, X, feature_names, packet):
    """
    Follows this packet down the tree and records every question the tree asked.

    The tree stores each question as:  "is <column> <= <threshold>?"
      - answer YES -> the packet goes to the LEFT branch
      - answer NO  -> the packet goes to the RIGHT branch

    If the tree asks about the same thing more than once (e.g. battery twice),
    the answers are MERGED into one sentence, so the website shows a clean list.
    """
    tree = model.tree_
    facts = {}  # what we learned about each field, in the order the tree asked

    # decision_path() gives the list of boxes (nodes) this packet passed through,
    # from the top of the tree down to the final answer.
    for node in model.decision_path(X).indices:

        # A leaf (final answer box) has no question, so skip it.
        if tree.children_left[node] == -1:
            continue

        column = feature_names[tree.feature[node]]  # which column the question is about
        threshold = tree.threshold[node]            # the number it compares against
        went_right = X.iloc[0][column] > threshold  # True = answer was NO (value is bigger)

        if column.startswith("data_type_"):
            # One-hot column, e.g. "data_type_Fault alert" (1 = yes, 0 = no).
            fact = facts.setdefault("data_type", {"is": None, "is_not": []})
            type_name = column[len("data_type_"):]
            if went_right:
                fact["is"] = type_name            # value 1 -> it IS this type
            else:
                fact["is_not"].append(type_name)  # value 0 -> it is NOT this type

        elif column in ("link_quality", "sat_mode"):
            # Ordered category stored as a number (e.g. Poor=0, Fair=1, Good=2).
            # Keep track of which category numbers are still possible after each question.
            names = CHOICE_FIELDS[column]
            fact = facts.setdefault(column, {"possible": list(range(len(names)))})
            fact["possible"] = [i for i in fact["possible"] if (i > threshold) == went_right]

        else:
            # Plain number column (size, battery, pass time).
            # Remember the lower limit ("more than") and upper limit ("at most").
            # Later questions on the same column are always narrower, so we just overwrite.
            fact = facts.setdefault(column, {"more_than": None, "at_most": None})
            if went_right:
                fact["more_than"] = threshold
            else:
                fact["at_most"] = threshold

    return [to_sentence(field, fact, packet) for field, fact in facts.items()]


def fmt(number):
    """Shows numbers neatly: 40.0 -> '40', 22.5 -> '22.5'."""
    return f"{number:g}"


def to_sentence(field, fact, packet):
    """Turns what we learned about one field into one plain-English sentence."""

    if field == "data_type":
        if fact["is"]:
            return f"Data type is {fact['is']}"
        not_list = " or ".join(fact["is_not"])
        return f"Data type is not {not_list} (it is {packet['data_type']})"

    if field == "sat_mode":
        return f"Satellite is in {packet['sat_mode']} mode"

    if field == "link_quality":
        possible = [config.LINK_QUALITIES[i] for i in fact["possible"]]
        if len(possible) == 1:
            return f"Link quality is {possible[0]}"
        return f"Link quality is {' or '.join(possible)} (it is {packet['link_quality']})"

    # Number fields
    label, unit = NUMBER_FIELDS[field]
    low, high = fact["more_than"], fact["at_most"]
    if low is not None and high is not None:
        limit = f"between {fmt(low)} and {fmt(high)}{unit}"
    elif low is not None:
        limit = f"more than {fmt(low)}{unit}"
    else:
        limit = f"at most {fmt(high)}{unit}"
    return f"{label} is {fmt(packet[field])}{unit} ({limit})"


# ---------------------------------------------------------------------------
# MAIN FUNCTION: used by app.py
# ---------------------------------------------------------------------------

def predict_packet(packet):
    """Takes one packet (a dictionary) and returns its priority, confidence and path."""

    # Step 1: check the input
    clean = clean_packet(packet)

    # Load the trained tree and the column order it was trained on
    saved = load_model()
    model, feature_names = saved["model"], saved["feature_names"]

    # Step 2: turn the packet into a one-row table and encode it exactly like in training.
    # reindex() makes sure the columns are in the SAME order the model was trained on.
    X = encode(pd.DataFrame([clean])[config.FEATURES])
    X = X.reindex(columns=feature_names, fill_value=0)

    # Step 3: ask the model for its answer
    priority = model.predict(X)[0]

    # Step 4: confidence.
    # predict_proba() looks at the final box (leaf) this packet landed in and gives the
    # share of TRAINING packets in that box for each priority.
    # e.g. 95 High + 5 Medium in the box -> High: 0.95, Medium: 0.05, Low: 0.0
    probs = dict(zip(model.classes_, model.predict_proba(X)[0]))
    probabilities = {label: round(float(probs.get(label, 0)), 3) for label in config.PRIORITIES}
    confidence = probabilities[priority]  # how sure it is about the answer it gave

    # Step 5: explain the path in plain sentences
    path = explain_path(model, X, feature_names, clean)

    return {
        "priority": str(priority),
        "confidence": confidence,
        "path": path,
        "probabilities": probabilities,
    }


# ---------------------------------------------------------------------------
# QUICK TEST: runs only when you type  python -m ml.predict
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    examples = [
        {"data_type": "Fault alert", "size_kb": 10, "battery_pct": 70,
         "link_quality": "Poor", "pass_time_min": 3.0, "sat_mode": "Normal"},
        {"data_type": "Housekeeping", "size_kb": 40, "battery_pct": 25,
         "link_quality": "Good", "pass_time_min": 6.5, "sat_mode": "Normal"},
        {"data_type": "SSTV image", "size_kb": 500, "battery_pct": 80,
         "link_quality": "Good", "pass_time_min": 9.0, "sat_mode": "Normal"},
        {"data_type": "Voice/Data", "size_kb": 300, "battery_pct": 40,
         "link_quality": "Fair", "pass_time_min": 4.0, "sat_mode": "Safe"},
    ]

    for packet in examples:
        result = predict_packet(packet)
        print("\nPacket:", packet)
        print(f"  -> Priority: {result['priority']}   (confidence {result['confidence']:.0%})")
        print("  -> All chances:", result["probabilities"])
        print("  -> Why:")
        for step in result["path"]:
            print("       -", step)

    # This one is invalid on purpose, to show the error checking works.
    print("\nTesting a bad packet:")
    try:
        predict_packet({"data_type": "Banana", "size_kb": 10, "battery_pct": 50,
                        "link_quality": "Good", "pass_time_min": 5, "sat_mode": "Normal"})
    except ValueError as error:
        print("  -> Error caught correctly:", error)