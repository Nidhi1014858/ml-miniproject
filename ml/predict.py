"""
ml/predict.py
Inference and rule explanation module for LinkWise.
Provides reusable functions to predict transmission priority of satellite packets
and generate human-readable decision explanations based on the trained Decision Tree.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# Global cache for the loaded model pipeline
_CACHED_PIPELINE = None


def load_model(model_path: Path = config.MODEL_PATH):
    """
    Loads and caches the trained scikit-learn pipeline from disk.
    """
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. Please run ml/train.py first."
            )
        _CACHED_PIPELINE = joblib.load(model_path)
    return _CACHED_PIPELINE


def validate_packet(packet: Dict[str, Any]) -> None:
    """
    Validates packet attributes against configuration rules.
    Raises ValueError with descriptive messages if validation fails.
    """
    missing = [f for f in config.FEATURE_COLUMNS if f not in packet]
    if missing:
        raise ValueError(f"Missing required packet features: {missing}")

    for cat_feature, allowed in config.ALLOWED_VALUES.items():
        if cat_feature in packet and packet[cat_feature] not in allowed:
            raise ValueError(
                f"Invalid value '{packet[cat_feature]}' for '{cat_feature}'. Allowed: {allowed}"
            )

    try:
        data_size = float(packet["data_size_kb"])
        if data_size < config.NUMERICAL_BOUNDS["data_size_kb"]["min"] or data_size > config.NUMERICAL_BOUNDS["data_size_kb"]["max"]:
            raise ValueError(
                f"data_size_kb must be between {config.NUMERICAL_BOUNDS['data_size_kb']['min']} and {config.NUMERICAL_BOUNDS['data_size_kb']['max']} KB."
            )
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid data_size_kb: {e}")

    try:
        battery = float(packet["battery_level"])
        if battery < config.NUMERICAL_BOUNDS["battery_level"]["min"] or battery > config.NUMERICAL_BOUNDS["battery_level"]["max"]:
            raise ValueError(
                f"battery_level must be between {config.NUMERICAL_BOUNDS['battery_level']['min']}% and {config.NUMERICAL_BOUNDS['battery_level']['max']}%."
            )
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid battery_level: {e}")


def predict_packet(packet: Dict[str, Any]) -> str:
    """
    Predicts the satellite transmission priority ('High', 'Medium', 'Low')
    for an input telemetry packet.

    Parameters:
        packet (dict): Dictionary with keys:
            - data_type (str)
            - urgency (str)
            - data_size_kb (float/int)
            - battery_level (float/int)
            - link_quality (str)

    Returns:
        str: Predicted priority class ('High', 'Medium', or 'Low')
    """
    validate_packet(packet)
    pipeline = load_model()

    # Create single-row DataFrame matching training feature columns
    df = pd.DataFrame([{
        "data_type": str(packet["data_type"]),
        "urgency": str(packet["urgency"]),
        "data_size_kb": float(packet["data_size_kb"]),
        "battery_level": float(packet["battery_level"]),
        "link_quality": str(packet["link_quality"]),
    }])[config.FEATURE_COLUMNS]

    prediction = pipeline.predict(df)[0]
    return str(prediction)


def explain_prediction(packet: Dict[str, Any], predicted_priority: str = None) -> str:
    """
    Generates a clear human-readable explanation of why the Decision Tree
    assigned the specific priority, based on the actual path traversed in the tree.

    Parameters:
        packet (dict): Telemetry packet feature dictionary.
        predicted_priority (str, optional): The already-predicted class.

    Returns:
        str: Descriptive human-readable explanation sentence.
    """
    pipeline = load_model()
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    # Preprocess the input packet
    df = pd.DataFrame([{
        "data_type": str(packet["data_type"]),
        "urgency": str(packet["urgency"]),
        "data_size_kb": float(packet["data_size_kb"]),
        "battery_level": float(packet["battery_level"]),
        "link_quality": str(packet["link_quality"]),
    }])[config.FEATURE_COLUMNS]

    if predicted_priority is None:
        predicted_priority = str(pipeline.predict(df)[0])

    X_trans = preprocessor.transform(df)

    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(config.CATEGORICAL_FEATURES)
    all_feature_names = list(cat_feature_names) + config.NUMERICAL_FEATURES

    # Trace decision path in tree
    node_indicator = clf.decision_path(X_trans)
    node_index = node_indicator.indices

    reasons = []
    for node_id in node_index:
        # Stop at leaf nodes
        if clf.tree_.children_left[node_id] == clf.tree_.children_right[node_id]:
            continue

        feat_idx = clf.tree_.feature[node_id]
        thresh = clf.tree_.threshold[node_id]
        feat_name = all_feature_names[feat_idx]
        val = X_trans[0, feat_idx]

        # Translate feature and threshold into domain language
        if "urgency_High" in feat_name:
            if val > thresh:
                reasons.append("high urgency flag")
            else:
                reasons.append("non-urgent status")
        elif "urgency_Low" in feat_name and val > thresh:
            reasons.append("low urgency status")
        elif "urgency_Medium" in feat_name and val > thresh:
            reasons.append("moderate urgency")

        elif "data_type_TT&C" in feat_name and val > thresh:
            reasons.append("critical TT&C command packet type")
        elif "data_type_SSTV" in feat_name and val > thresh:
            reasons.append("heavy SSTV payload image type")
        elif "data_type_Housekeeping" in feat_name and val > thresh:
            reasons.append("routine housekeeping telemetry")
        elif "data_type_Voice/Data" in feat_name and val > thresh:
            reasons.append("payload voice/data stream")

        elif "link_quality_Good" in feat_name:
            if val > thresh:
                reasons.append("good link quality window")
            else:
                reasons.append("suboptimal link quality")
        elif "link_quality_Poor" in feat_name and val > thresh:
            reasons.append("poor ground station link quality")
        elif "link_quality_Fair" in feat_name and val > thresh:
            reasons.append("fair link conditions")

        elif "battery_level" in feat_name:
            if val <= thresh:
                reasons.append(f"low battery level ({val:.0f}% <= {thresh:.0f}%)")
            else:
                reasons.append(f"healthy battery level ({val:.0f}% > {thresh:.0f}%)")

        elif "data_size_kb" in feat_name:
            if val <= thresh:
                reasons.append(f"compact packet size ({val:.0f} KB <= {thresh:.0f} KB)")
            else:
                reasons.append(f"large packet footprint ({val:.0f} KB > {thresh:.0f} KB)")

    # Deduplicate while preserving order
    unique_reasons = []
    for r in reasons:
        if r not in unique_reasons:
            unique_reasons.append(r)

    # Format into a clean, human-friendly explanation sentence
    dt = packet.get("data_type", "")
    urg = packet.get("urgency", "")
    link = packet.get("link_quality", "")
    bat = packet.get("battery_level", "")

    if len(unique_reasons) >= 2:
        reasons_text = f"{unique_reasons[0].capitalize()} and {unique_reasons[1]}"
    elif len(unique_reasons) == 1:
        reasons_text = f"{unique_reasons[0].capitalize()}"
    else:
        reasons_text = f"{urg} urgency and {link.lower()} link quality"

    if predicted_priority == "High":
        return f"{reasons_text} guided the decision tree to assign High transmission priority for prompt downlink."
    elif predicted_priority == "Low":
        return f"{reasons_text} led the decision tree to defer transmission with Low priority to preserve satellite resources."
    else:
        return f"{reasons_text} placed the packet into standard Medium transmission priority in the on-board queue."


def predict_and_explain(packet: Dict[str, Any]) -> Tuple[str, str]:
    """
    Convenience function returning both priority and explanation.
    """
    priority = predict_packet(packet)
    explanation = explain_prediction(packet, priority)
    return priority, explanation


if __name__ == "__main__":
    # Self-test with sample packets
    test_packets = [
        {
            "data_type": "TT&C",
            "urgency": "High",
            "data_size_kb": 25,
            "battery_level": 85,
            "link_quality": "Good",
        },
        {
            "data_type": "SSTV",
            "urgency": "Low",
            "data_size_kb": 1850,
            "battery_level": 28,
            "link_quality": "Poor",
        },
        {
            "data_type": "Housekeeping",
            "urgency": "Medium",
            "data_size_kb": 120,
            "battery_level": 70,
            "link_quality": "Fair",
        },
    ]

    print("LinkWise - predict.py Self-Test:")
    print("=" * 60)
    for idx, pkt in enumerate(test_packets, 1):
        p, expl = predict_and_explain(pkt)
        print(f"Sample {idx}: {pkt}")
        print(f"  -> Predicted Priority : {p}")
        print(f"  -> Decision Path Expl : {expl}\n")
