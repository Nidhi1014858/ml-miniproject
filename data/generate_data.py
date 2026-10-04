"""
data/generate_data.py
Generates the simulated satellite telemetry dataset for LinkWise.
Saves 1500 rows to DATA_PATH and the first 50 rows to SAMPLE_PATH.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add repository root to sys.path so config can be imported directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def assign_priority(row) -> str:
    """
    Decides the transmission priority (High, Medium, Low) for a single packet
    based on satellite operational constraints and communication pass feasibility.

    Scoring Logic:
    1. Base score by packet type:
       - Fault alert: 75 (critical anomalies require swift attention)
       - Housekeeping: 50 (routine health and telemetry status)
       - SSTV image: 40 (scientific/camera payload data)
       - Voice/Data: 30 (amateur transponder/user data)
    2. Safe mode adjustment:
       - +20 for Fault alert and Housekeeping (essential diagnostics)
       - -20 for SSTV image and Voice/Data (non-essential payloads shed)
    3. Low battery health priority:
       - If Housekeeping and battery_pct < 30: +10 (vital health monitoring)
    4. Payload link & power penalties (SSTV and Voice/Data only):
       - If battery_pct < 25: -15 (conserve battery power)
       - If link_quality == 'Poor': -10 (avoid packet corruptions)
       - If link_quality == 'Good': +5 (favorable RF downlink window)
    5. Pass transmission feasibility:
       - Speeds: Poor = 1 KB/s, Fair = 3 KB/s, Good = 6 KB/s
       - time_needed_min = size_kb / speed / 60
       - If time_needed_min > pass_time_min: -25 (packet cannot complete transfer)
    6. Priority thresholds:
       - Score >= 60: 'High'
       - 35 <= Score < 60: 'Medium'
       - Score < 35: 'Low'
    """
    # 1. Base score determined by data type
    base_scores = {
        config.DATA_TYPES[0]: 75,  # Fault alert
        config.DATA_TYPES[1]: 50,  # Housekeeping
        config.DATA_TYPES[2]: 40,  # SSTV image
        config.DATA_TYPES[3]: 30,  # Voice/Data
    }
    score = base_scores[row["data_type"]]

    # 2. Satellite operational mode adjustments
    if row["sat_mode"] == config.SAT_MODES[1]:  # Safe mode
        if row["data_type"] in [config.DATA_TYPES[0], config.DATA_TYPES[1]]:
            score += 20
        elif row["data_type"] in [config.DATA_TYPES[2], config.DATA_TYPES[3]]:
            score -= 20

    # 3. Critical housekeeping telemetry under low battery
    if row["data_type"] == config.DATA_TYPES[1] and row["battery_pct"] < 30:
        score += 10

    # 4. Power and RF conditions for payload transmissions (SSTV & Voice/Data)
    if row["data_type"] in [config.DATA_TYPES[2], config.DATA_TYPES[3]]:
        if row["battery_pct"] < 25:
            score -= 15
        if row["link_quality"] == config.LINK_QUALITIES[0]:  # Poor
            score -= 10
        elif row["link_quality"] == config.LINK_QUALITIES[2]:  # Good
            score += 5

    # 5. Feasibility check: will the packet fit in the contact pass window?
    link_speeds = {
        config.LINK_QUALITIES[0]: 1,  # Poor: 1 KB/s
        config.LINK_QUALITIES[1]: 3,  # Fair: 3 KB/s
        config.LINK_QUALITIES[2]: 6,  # Good: 6 KB/s
    }
    speed_kb_s = link_speeds[row["link_quality"]]
    time_needed_min = row["size_kb"] / speed_kb_s / 60.0

    if time_needed_min > row["pass_time_min"]:
        score -= 25

    # 6. Map calculated numerical score to discrete priority label
    if score >= 60:
        return config.PRIORITIES[0]  # High
    elif score >= 35:
        return config.PRIORITIES[1]  # Medium
    else:
        return config.PRIORITIES[2]  # Low


def generate_dataset(num_rows: int = 1500, random_seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """
    Synthesizes the complete telemetry DataFrame with 1500 rows and applies
    domain scoring rules and 6% realistic human-judgement noise.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Sample data_type (Fault alert 15%, Housekeeping 35%, SSTV image 25%, Voice/Data 25%)
    data_types = rng.choice(
        config.DATA_TYPES,
        size=num_rows,
        p=[0.15, 0.35, 0.25, 0.25],
    )

    # 2. Sample sat_mode (Normal 85%, Safe 15%)
    sat_modes = rng.choice(
        config.SAT_MODES,
        size=num_rows,
        p=[0.85, 0.15],
    )

    # 3. Sample size_kb dependent on data_type:
    #    Fault alert: 1-20, Housekeeping: 10-100, SSTV image: 200-800, Voice/Data: 50-400
    sizes_kb = []
    for dt in data_types:
        if dt == config.DATA_TYPES[0]:  # Fault alert
            sizes_kb.append(int(rng.integers(1, 21)))
        elif dt == config.DATA_TYPES[1]:  # Housekeeping
            sizes_kb.append(int(rng.integers(10, 101)))
        elif dt == config.DATA_TYPES[2]:  # SSTV image
            sizes_kb.append(int(rng.integers(200, 801)))
        else:  # Voice/Data
            sizes_kb.append(int(rng.integers(50, 401)))
    sizes_kb = np.array(sizes_kb, dtype=int)

    # 4. Sample battery_pct dependent on sat_mode:
    #    Normal: 10-100, Safe: 10-50
    batteries_pct = []
    for sm in sat_modes:
        if sm == config.SAT_MODES[0]:  # Normal
            batteries_pct.append(int(rng.integers(10, 101)))
        else:  # Safe
            batteries_pct.append(int(rng.integers(10, 51)))
    batteries_pct = np.array(batteries_pct, dtype=int)

    # 5. Sample link_quality (Poor 25%, Fair 35%, Good 40%)
    link_qualities = rng.choice(
        config.LINK_QUALITIES,
        size=num_rows,
        p=[0.25, 0.35, 0.40],
    )

    # 6. Sample pass_time_min (0.5 to 12.0, rounded to 1 decimal)
    pass_times_min = np.round(rng.uniform(0.5, 12.0, size=num_rows), 1)

    # Create temporary DataFrame to evaluate scoring rule row-by-row
    df = pd.DataFrame({
        "data_type": data_types,
        "size_kb": sizes_kb,
        "battery_pct": batteries_pct,
        "link_quality": link_qualities,
        "pass_time_min": pass_times_min,
        "sat_mode": sat_modes,
    })

    # Apply base deterministic scoring
    priorities = [assign_priority(row) for _, row in df.iterrows()]

    # 7. Add label noise to exactly 6% of rows
    #    (High -> Medium, Low -> Medium, Medium -> randomly High or Low)
    noise_count = int(0.06 * num_rows)
    noise_indices = rng.choice(num_rows, size=noise_count, replace=False)

    for idx in noise_indices:
        current_p = priorities[idx]
        if current_p == config.PRIORITIES[0]:  # High
            priorities[idx] = config.PRIORITIES[1]  # Medium
        elif current_p == config.PRIORITIES[2]:  # Low
            priorities[idx] = config.PRIORITIES[1]  # Medium
        else:  # Medium
            priorities[idx] = rng.choice([config.PRIORITIES[0], config.PRIORITIES[2]])

    df["priority"] = priorities

    # Reorder columns to match config.COLUMNS exactly
    df = df[config.COLUMNS]
    return df


def main():
    print("=" * 65)
    print("LinkWise - Satellite Downlink Telemetry Dataset Generator")
    print("=" * 65)

    # Ensure parent output directory exists
    Path(config.DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(config.SAMPLE_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Generate full dataset
    df = generate_dataset(num_rows=1500, random_seed=config.RANDOM_SEED)

    # Save 1500 rows to DATA_PATH
    df.to_csv(config.DATA_PATH, index=False)
    print(f"\n[OK] Full dataset (1500 rows) saved to: {config.DATA_PATH}")

    # Save first 50 rows to SAMPLE_PATH
    sample_df = df.head(50)
    sample_df.to_csv(config.SAMPLE_PATH, index=False)
    print(f"[OK] Sample dataset (50 rows) saved to: {config.SAMPLE_PATH}")

    # Print summary information
    total_rows = len(df)
    duplicate_rows = df.duplicated().sum()

    print("\n[+] Dataset Overview:")
    print(f"    - Total Rows: {total_rows}")
    print(f"    - Duplicate Rows: {duplicate_rows}")
    print(f"    - Columns ({len(df.columns)}): {list(df.columns)}")

    print("\n[+] Priority Class Distribution:")
    class_counts = df["priority"].value_counts()
    class_pcts = (class_counts / total_rows) * 100

    needs_threshold_suggestion = False
    for p in config.PRIORITIES:
        cnt = class_counts.get(p, 0)
        pct = class_pcts.get(p, 0.0)
        print(f"    - {p:7s}: {cnt:4d} rows ({pct:5.1f}%)")
        if pct < 20.0 or pct > 50.0:
            needs_threshold_suggestion = True

    print("\n[+] Priority by Data Type (Cross-Table):")
    cross_tab = pd.crosstab(df["data_type"], df["priority"])[config.PRIORITIES]
    print(cross_tab.to_string())

    if needs_threshold_suggestion:
        print("\n[!] Threshold Feedback:")
        print("    One or more priority classes fall outside the recommended 20% - 50% range.")
        print("    Suggestion: Adjust the score boundaries in Step 6 (e.g. modify High threshold from 60")
        print("    or Medium threshold from 35). Note: As requested, no change has been applied automatically.")
    else:
        print("\n[OK] Class balance check: All priority classes fall within the healthy 20% - 50% target range!")

    print("=" * 65)


if __name__ == "__main__":
    main()
