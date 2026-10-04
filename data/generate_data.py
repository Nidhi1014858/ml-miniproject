"""
data/generate_data.py
Generates a realistic synthetic satellite telemetry dataset for SomaiyaSat
and saves it to data/satellite_data.csv.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path so config can be imported directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def generate_satellite_data(num_samples: int = 450, random_seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """
    Generates realistic satellite data records with realistic prioritization rules.

    Parameters:
        num_samples (int): Total records to synthesize (default: 450).
        random_seed (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: Synthetic dataset with features and target priority.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Feature sampling with realistic distributions
    data_types = rng.choice(
        config.ALLOWED_DATA_TYPES,
        size=num_samples,
        p=[0.25, 0.30, 0.20, 0.25],  # TT&C, Housekeeping, SSTV, Voice/Data
    )

    urgencies = rng.choice(
        config.ALLOWED_URGENCY_LEVELS,
        size=num_samples,
        p=[0.35, 0.40, 0.25],  # Low, Medium, High
    )

    link_qualities = rng.choice(
        config.ALLOWED_LINK_QUALITIES,
        size=num_samples,
        p=[0.25, 0.45, 0.30],  # Poor, Fair, Good
    )

    # Battery level between 15% and 99%
    battery_levels = rng.integers(15, 100, size=num_samples)

    # Realistic data size in KB based on packet type
    data_sizes = []
    for dt in data_types:
        if dt == "TT&C":
            # Command & Telemetry packets are compact
            data_sizes.append(int(rng.integers(5, 75)))
        elif dt == "Housekeeping":
            # Diagnostic and subsystem sensor logs
            data_sizes.append(int(rng.integers(30, 220)))
        elif dt == "Voice/Data":
            # Audio transmissions & payload data
            data_sizes.append(int(rng.integers(120, 950)))
        else:  # SSTV
            # Image payloads are large
            data_sizes.append(int(rng.integers(450, 2400)))

    data_sizes = np.array(data_sizes)

    # 2. Domain-driven prioritization rules for SomaiyaSat
    priorities = []
    for i in range(num_samples):
        dt = data_types[i]
        urg = urgencies[i]
        link = link_qualities[i]
        bat = battery_levels[i]
        size = data_sizes[i]

        # Base scoring algorithm representing on-board scheduler logic
        if dt == "TT&C":
            if urg == "High":
                p = "High"
            elif urg == "Medium":
                p = "High" if bat >= 35 else "Medium"
            else:
                p = "Medium" if bat >= 40 else "Low"

        elif dt == "Housekeeping":
            if urg == "High":
                p = "High" if bat >= 30 else "Medium"
            elif urg == "Medium":
                p = "Medium" if bat >= 35 else "Low"
            else:
                p = "Medium" if (bat >= 70 and link == "Good") else "Low"

        elif dt == "SSTV":
            # Images require power and reliable channel
            if bat < 40 or link == "Poor" or size > 1800:
                p = "Low"
            elif urg == "High" and bat >= 60 and link != "Poor":
                p = "High"
            elif bat >= 50 and link == "Good":
                p = "Medium"
            else:
                p = "Low"

        else:  # Voice/Data
            if urg == "High" and bat >= 40:
                p = "High"
            elif urg == "Medium" and bat >= 45 and link != "Poor":
                p = "Medium"
            elif urg == "Low" and bat >= 80 and link == "Good":
                p = "Medium"
            else:
                p = "Low"

        # 3. Add slight realistic noise (~6% anomalous real-world flips)
        if rng.random() < 0.06:
            p = rng.choice(config.PRIORITY_CLASSES)

        priorities.append(p)

    df = pd.DataFrame({
        "data_type": data_types,
        "urgency": urgencies,
        "data_size_kb": data_sizes,
        "battery_level": battery_levels,
        "link_quality": link_qualities,
        "priority": priorities,
    })

    return df


def main():
    print("=" * 60)
    print("SatPrior - Synthetic Satellite Telemetry Dataset Generator")
    print("=" * 60)

    # Ensure output directory exists
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    df = generate_satellite_data(num_samples=450, random_seed=config.RANDOM_SEED)

    # Save to CSV
    df.to_csv(config.DATASET_PATH, index=False)

    # Display dataset info as specified in requirements
    print(f"\n[+] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\n[+] First 10 Rows:")
    print(df.head(10).to_string(index=True))

    print("\n[+] Target Priority Class Distribution:")
    dist = df["priority"].value_counts()
    for cls, count in dist.items():
        pct = (count / len(df)) * 100
        print(f"    - {cls:7s}: {count:3d} ({pct:5.1f}%)")

    print(f"\n[OK] Dataset successfully saved to: {config.DATASET_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
