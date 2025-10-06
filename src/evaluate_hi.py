import re
import numpy as np
import pandas as pd

def parse_min_max(s):
    """Extract minimum and maximum numeric values from threshold strings."""
    if pd.isna(s):
        return (np.nan, np.nan)
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(s))
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    elif len(nums) == 1:
        return float(nums[0]), np.nan
    return (np.nan, np.nan)

def parse_working_value(s):
    """Extract numeric value from working column strings."""
    if pd.isna(s):
        return np.nan
    match = re.search(r"([-+]?\d*\.\d+|\d+)", str(s).replace(",", "."))
    return float(match.group(0)) if match else np.nan

def compute_hi_for_row(row, warn_margin=0.1):
    """Compute Health Index and predicted status for one equipment row."""
    working_val = np.nan
    low, high = np.nan, np.nan

    # Detect likely column names
    working_cols = [col for col in row.index if "WORKING" in str(col).upper()]
    thresh_cols = [col for col in row.index if "THRESHOLD" in str(col).upper() or "MIN" in str(col).upper()]

    if working_cols:
        working_val = parse_working_value(row[working_cols[0]])
    if thresh_cols:
        low, high = parse_min_max(row[thresh_cols[0]])

    if np.isnan(working_val) or np.isnan(low) or np.isnan(high):
        return np.nan, "Unknown"

    if low <= working_val <= high:
        return 1.0, "Healthy"
    elif (low * (1 - warn_margin)) <= working_val <= (high * (1 + warn_margin)):
        return 0.5, "Warning"
    else:
        return 0.0, "Critical"

def evaluate_dataframe(df):
    """Compute Health Index (HI) and Predicted Status for all readings."""
    df_copy = df.copy()
    results = df_copy.apply(lambda row: compute_hi_for_row(row), axis=1)
    df_copy["Health_Index"] = [r[0] for r in results]
    df_copy["Predicted_Status"] = [r[1] for r in results]
    return df_copy
