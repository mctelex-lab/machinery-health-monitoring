# ==============================================================
# ⚓ evaluate_hi.py
# ==============================================================
# This module computes Health Index (HI) and assigns operational
# Status categories (Healthy, Warning, Critical) based on input
# datasets from naval machinery.
# ==============================================================

import pandas as pd
import numpy as np

# --------------------------------------------------------------
# ⚙️ Core Health Evaluation Function
# --------------------------------------------------------------

def evaluate_dataframe(df):
    """
    Compute Health Index (HI) and Status for a given dataset.
    Expects the following (at minimum) columns:
        - SYNTHETIC_VALUE
        - MIN_MAX_THRESHOLDS
    Returns:
        DataFrame with added 'Health_Index' and 'Status' columns
    """

    df = df.copy()  # Work on a safe copy

    # ----------------------------------------------------------
    # 🧩 Clean numeric fields
    # ----------------------------------------------------------
    if "SYNTHETIC_VALUE" in df.columns:
        df["SYNTHETIC_VALUE"] = (
            pd.to_numeric(df["SYNTHETIC_VALUE"], errors="coerce")
        )
    else:
        raise ValueError("❌ Missing 'SYNTHETIC_VALUE' column in dataset.")

    # Extract numeric ranges from MIN_MAX_THRESHOLDS column
    def parse_threshold(value):
        if isinstance(value, str):
            parts = value.replace("–", "-").split("-")
            if len(parts) == 2:
                try:
                    low = float(parts[0].strip())
                    high = float(parts[1].strip())
                    return low, high
                except ValueError:
                    return np.nan, np.nan
        return np.nan, np.nan

    thresholds = df["MIN_MAX_THRESHOLDS"].apply(parse_threshold)
    df["MIN_TH"] = thresholds.apply(lambda x: x[0])
    df["MAX_TH"] = thresholds.apply(lambda x: x[1])

    # ----------------------------------------------------------
    # 📈 Calculate Health Index
    # ----------------------------------------------------------
    def compute_health_index(row):
        val, low, high = row["SYNTHETIC_VALUE"], row["MIN_TH"], row["MAX_TH"]

        if pd.isna(val) or pd.isna(low) or pd.isna(high):
            return np.nan

        # Define scoring logic
        if low <= val <= high:
            return 1.0  # optimal
        elif (low - 0.1 * (high - low)) <= val <= (high + 0.1 * (high - low)):
            return 0.6  # warning range
        else:
            return 0.2  # critical range

    df["Health_Index"] = df.apply(compute_health_index, axis=1)

    # ----------------------------------------------------------
    # 🚨 Derive Health Status from HI
    # ----------------------------------------------------------
    df["Status"] = pd.cut(
        df["Health_Index"],
        bins=[-np.inf, 0.5, 0.8, np.inf],
        labels=["Critical", "Warning", "Healthy"]
    )

    # ----------------------------------------------------------
    # 🧮 Optional Summary Statistics
    # ----------------------------------------------------------
    hi_stats = {
        "Mean HI": round(df["Health_Index"].mean(), 3),
        "Std Dev HI": round(df["Health_Index"].std(), 3),
        "Skewness": round(df["Health_Index"].skew(), 3),
        "Kurtosis": round(df["Health_Index"].kurtosis(), 3),
        "Total Evaluated": int(df["Health_Index"].notna().sum()),
    }

    # Attach summary metrics to the DataFrame for easy access in the app
    df.attrs["metrics"] = hi_stats

    return df

# --------------------------------------------------------------
# 🧠 Example Usage (for local testing only)
# --------------------------------------------------------------
if __name__ == "__main__":
    sample_data = {
        "SYSTEM": ["Propulsion", "Cooling", "Electrical"],
        "EQUIPMENT": ["Main Engine", "Pump", "Generator"],
        "HEALTH_INDICATOR_HI": ["Oil pressure", "Water temp", "Voltage"],
        "SYNTHETIC_VALUE": [4.3, 92, 430],
        "MIN_MAX_THRESHOLDS": ["3.5-6.0", "60-90", "420-450"],
    }
    df_sample = pd.DataFrame(sample_data)
    evaluated = evaluate_dataframe(df_sample)
    print(evaluated)
    print("\nSummary Metrics:", evaluated.attrs["metrics"])
