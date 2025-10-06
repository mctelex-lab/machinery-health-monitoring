# ==============================================================
# 🧠 Evaluate Health Index (HI) for Naval Machinery Systems
# ==============================================================
# This module reads a machinery dataset, applies parameter thresholds,
# computes a normalized Health Index (HI), and assigns a condition label.
# ==============================================================

import pandas as pd
import numpy as np
import re

# --------------------------------------------------------------
# 1️⃣ Helper function to normalize column names
# --------------------------------------------------------------
def normalize_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


# --------------------------------------------------------------
# 2️⃣ Thresholds (adjustable domain-specific ranges)
# --------------------------------------------------------------
THRESHOLDS = {
    'lubeoilpressurebar': (3.5, 6.0),
    'lubeoiltemp°c': (60, 90),
    'exhaustgastemp°c': (350, 450),
    'oiltemp°c': (60, 85),
    'shaftvibrationmms': (0.5, 2.5),
    'bearingtemp°c': (50, 80),
    'voltagev': (420, 450),
    'frequencyhz': (59, 61),
    'load': (30, 85),
    'flowmh': (70, 100),
    'dischargepressurebar': (2.5, 4.5),
    'airpressurebar': (24, 30),
    'dischargetemp°c': (90, 110),
    'hydraulicpressurebar': (135, 160),
    'rudderresponses': (0.8, 1.2),
    'detectorresponses': (0, 5),
    'co2cylinderpressurebar': (160, 200),
    'cabintemp°c': (20, 26),
    'compressorcurrenta': (25, 35),
    'outputm³day': (10, 15)
}
NORM_THRESHOLDS = {normalize_name(k): v for k, v in THRESHOLDS.items()}


# --------------------------------------------------------------
# 3️⃣ Health Index computation for a single reading
# --------------------------------------------------------------
def evaluate_health_index(row: pd.Series, warn_margin: float = 0.1):
    scores = []
    for col in row.index:
        val = row[col]
        if pd.isna(val):
            continue
        try:
            num = float(str(val).split()[0])
        except:
            continue

        key = normalize_name(col)
        if key not in NORM_THRESHOLDS:
            continue

        low, high = NORM_THRESHOLDS[key]
        rng = high - low
        if low <= num <= high:
            score = 1.0
        elif (low - warn_margin * rng) <= num <= (high + warn_margin * rng):
            score = 0.5
        else:
            score = 0.0
        scores.append(score)

    if not scores:
        return np.nan, "Unknown"

    hi = np.mean(scores)
    if hi >= 0.8:
        status = "Healthy"
    elif hi >= 0.5:
        status = "Warning"
    else:
        status = "Critical"
    return hi, status


# --------------------------------------------------------------
# 4️⃣ Main evaluation entry point for a dataframe
# --------------------------------------------------------------
def evaluate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_eval = df.copy()

    # Identify numeric or relevant columns
    numeric_like = [c for c in df.columns if any(k in normalize_name(c) for k in NORM_THRESHOLDS.keys())]

    # Compute HI for each row
    hi_list, status_list = [], []
    for _, row in df[numeric_like].iterrows():
        hi, st = evaluate_health_index(row)
        hi_list.append(hi)
        status_list.append(st)

    df_eval["Health_Index"] = hi_list
    df_eval["Predicted_Status"] = status_list

    return df_eval
