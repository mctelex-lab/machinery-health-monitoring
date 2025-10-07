# app.py
# ==============================================================
# ⚓ Equipment Health Condition Monitoring Prediction for Naval Ships
# Final corrected Streamlit app
# ==============================================================

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))  # allow local src imports

import tempfile
from io import BytesIO
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chardet
from fpdf import FPDF

# local evaluation function (must exist in src/evaluate_hi.py)
from evaluate_hi import evaluate_dataframe

# --------- Config ----------
st.set_page_config(page_title="Equipment Health Condition Monitoring",
                   page_icon="⚓",
                   layout="wide")

LOG_PATH = "data/maintenance_log.csv"
os.makedirs("data", exist_ok=True)

# ---------- Helpers ----------

def sanitize_columns(df):
    """Return df with cleaned column names and a mapping from original->clean."""
    orig = list(df.columns)
    cleaned = (
        pd.Index(orig)
        .astype(str)
        .str.strip()
        .str.replace('\u2013', '-', regex=False)   # en-dash
        .str.replace('\xa0', '', regex=False)      # nbsp
        .str.replace(' ', '_')
        .str.replace('-', '_')
        .str.replace(r'[^0-9a-zA-Z_]', '', regex=True)
    )
    df.columns = cleaned
    mapping = dict(zip(orig, cleaned))
    return df, mapping

def find_best_column(df, candidates):
    """
    Return the first column name from df.columns that matches any candidate (case-insensitive, ignore punctuation).
    candidates: list of strings to match (e.g. ['SYNTHETIC_VALUE', 'SYNTHETIC VALUE'])
    """
    def norm(s):
        return ''.join(ch.lower() for ch in str(s) if ch.isalnum())
    cols_norm = {norm(c): c for c in df.columns}
    for cand in candidates:
        n = norm(cand)
        if n in cols_norm:
            return cols_norm[n]
    # fallback: try partial matches
    for cnorm, col in cols_norm.items():
        for cand in candidates:
            if norm(cand) in cnorm or cnorm in norm(cand):
                return col
    return None

def extract_numeric_from_string(s):
    """Extract first float / int found in string, return float or NaN."""
    if pd.isna(s):
        return np.nan
    try:
        # find pattern like -?digits(.digits)?
        import re
        m = re.search(r"[-+]?\d*\.\d+|\d+", str(s))
        if m:
            return float(m.group(0))
    except Exception:
        pass
    return np.nan

def save_figure_temp(fig):
    """Save matplotlib figure to a temporary file and return path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)
    tmp.close()
    return tmp.name

def update_maintenance_log(mean_hi, next_inspection_dt, interval_str):
    """Append an entry to the maintenance log CSV and return dataframe."""
    entry = {
        "Record_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Mean_Health_Index": round(float(mean_hi), 3) if pd.notna(mean_hi) else np.nan,
        "Next_Inspection_Date": next_inspection_dt.strftime("%Y-%m-%d"),
        "Interval": interval_str
    }
    if os.path.exists(LOG_PATH):
        df_log = pd.read_csv(LOG_PATH)
    else:
        df_log = pd.DataFrame(columns=list(entry.keys()))
    df_log = pd.concat([df_log, pd.DataFrame([entry])], ignore_index=True)
    df_log.to_csv(LOG_PATH, index=False)
    return df_log

# PDF generation - cover + pages + page numbers + footer
def generate_pdf_report(metrics, interpretation, suggestions, chart_paths,
                        authors="Navy Capt. Daya Abdullahi & Dr. Awujoola Olalekan J"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ------ Cover Page ------
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "⚓ NAVAL MACHINERY HEALTH CONDITION REPORT", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", "", 14)
    pdf.cell(0, 8, "Predictive Maintenance Model Summary", ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7,
                   f"Authors: {authors}\nDate generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nConfidential: This document contains sensitive technical and operational data. Unauthorized distribution is prohibited.",
                   align="C")

    # ------ Metrics & Interpretation ------
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Model Performance Metrics", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "", 11)
    for k, v in metrics.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)

    pdf.ln(6)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Interpretation & Maintenance Suggestions", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7, interpretation)
    pdf.ln(3)
    for s in suggestions:
        pdf.multi_cell(0, 7, "- " + s)

    # ------ Charts ------
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Analytical Visualizations", ln=True)
    pdf.ln(6)
    for title, path in chart_paths.items():
        pdf.set_font("Arial", "I", 11)
        pdf.cell(0, 8, title, ln=True)
        try:
            pdf.image(path, x=10, w=190)
        except Exception:
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 6, f"[Unable to embed {title}]", ln=True)
        pdf.ln(6)

    # Footer and page numbers
    n_pages = pdf.page_no()
    for p in range(1, n_pages + 1):
        pdf.page = p
        pdf.set_y(-12)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 10, f"Generated via Predictive Maintenance System | Page {p} of {n_pages}", 0, 0, "C")

    out = BytesIO()
    pdf.output(out)
    out.seek(0)
    return out

# ---------- App UI ----------

st.title("⚓ Equipment Health Condition Monitoring Prediction for Naval Ships")
st.caption("Upload your dataset (CSV/XLSX). App will auto-detect columns and compute Health Index, suggest maintenance, and produce a PDF report.")

st.sidebar.header("Actions")
page = st.sidebar.radio("Page", ["Analyze Data", "Maintenance Log", "Example Dataset", "About"])

if page == "About":
    st.header("About")
    st.markdown("""
    This application computes a Health Index (HI) for naval machinery using uploaded operational parameters.
    The report is designed for Naval use.  
    Developed by Dr. Awujoola Olalekan.
    """)
    st.stop()

if page == "Example Dataset":
    st.header("Example Dataset")
    sample = pd.DataFrame({
        "SYSTEM": ["Propulsion", "Propulsion", "Auxiliary"],
        "EQUIPMENT": ["Main Engine", "Gearbox", "Generator"],
        "HEALTH INDICATOR (HI)": ["Lube oil pressure (bar)", "Oil temp (°C)", "Voltage (V)"],
        "SYNTHETIC VALUE": [4.2, 78.0, 440.0],
        "MIN–MAX THRESHOLDS": ["3.5 – 6.0", "60 – 90", "420 – 450"],
        "WORKING VALUE ONBOARD": ["3.6 bar", "82 °C", "438 V"],
        "REMARKS": ["Normal", "Slightly high", "Nominal"],
        "Actual_Status": ["Healthy", "Warning", "Healthy"]
    })
    st.dataframe(sample)
    st.download_button("Download example CSV", sample.to_csv(index=False).encode("utf-8"), "example_machinery_health_data.csv", "text/csv")
    st.stop()

# else Analyze Data or Maintenance Log
if page == "Maintenance Log":
    st.header("Maintenance Log")
    if os.path.exists(LOG_PATH):
        df_log = pd.read_csv(LOG_PATH)
        st.dataframe(df_log)
        st.download_button("Download maintenance log (CSV)", df_log.to_csv(index=False).encode("utf-8"), "maintenance_log.csv", "text/csv")
    else:
        st.info("No maintenance log exists yet. Run an analysis to create the first entry.")
    st.stop()

# Page: Analyze Data
st.header("Upload & Analyze Dataset")
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if not uploaded_file:
    st.info("Upload a CSV or XLSX file. Use the Example Dataset if you don't have one.")
    st.stop()

# Read file robustly
try:
    raw_bytes = uploaded_file.read()
    enc = chardet.detect(raw_bytes).get("encoding") or "utf-8"
    uploaded_file.seek(0)
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding=enc, on_bad_lines='skip')
    else:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

# sanitize column names but keep original mapping for user display
df_orig = df.copy()
df, colmap = sanitize_columns(df)

# Find the important columns using flexible candidates
synthetic_col = find_best_column(df, ["SYNTHETIC_VALUE", "SYNTHETIC VALUE", "SYNTHETICVALUE", "SYNTHETIC"])
minmax_col = find_best_column(df, ["MIN_MAX_THRESHOLDS", "MIN–MAX_THRESHOLDS", "MIN_MAX_THRESHOLDS", "MINMAX", "MIN_MAX", "THRESHOLDS"])
working_col = find_best_column(df, ["WORKING_VALUE_ONBOARD", "WORKING VALUE ONBOARD", "WORKINGVALUE", "WORKING_VALUE", "WORKING"])
hi_indicator_col = find_best_column(df, ["HEALTH_INDICATOR_HI", "HEALTH_INDICATOR", "HEALTH_INDICATOR_(HI)", "HEALTHINDICATOR"])
actual_status_col = find_best_column(df, ["Actual_Status", "ACTUAL_STATUS", "ActualStatus", "Actual_Status".lower()])

# Attempt to ensure we have MIN_MAX_THRESHOLDS column (required by evaluator)
if minmax_col is None:
    st.error("Missing 'MIN–MAX THRESHOLDS' column. Please ensure your file has a column with thresholds (e.g., 'MIN–MAX THRESHOLDS').")
    st.stop()

# If synthetic_col missing, try to extract numeric from working_col
if synthetic_col is None:
    if working_col is not None:
        # create synthetic column from working
        df["SYNTHETIC_VALUE"] = df[working_col].apply(extract_numeric_from_string)
        synthetic_col = "SYNTHETIC_VALUE"
        st.info("No SYNTHETIC_VALUE column found — extracting numeric value from WORKING_VALUE_ONBOARD.")
    else:
        st.error("Missing both 'SYNTHETIC VALUE' and 'WORKING VALUE ONBOARD' columns. The evaluator needs at least one numeric reading column.")
        st.stop()
else:
    # ensure synthetic column numeric (coerce)
    df[synthetic_col] = pd.to_numeric(df[synthetic_col], errors="coerce")

# Prepare df for evaluator: rename the detected columns to the expected names
# evaluate_dataframe expects columns 'SYNTHETIC_VALUE' and 'MIN_MAX_THRESHOLDS' (per our evaluate_hi.py)
rename_map = {}
if synthetic_col != "SYNTHETIC_VALUE":
    rename_map[synthetic_col] = "SYNTHETIC_VALUE"
if minmax_col != "MIN_MAX_THRESHOLDS":
    rename_map[minmax_col] = "MIN_MAX_THRESHOLDS"
if hi_indicator_col and hi_indicator_col != "HEALTH_INDICATOR_HI":
    rename_map[hi_indicator_col] = "HEALTH_INDICATOR_HI"
if working_col and working_col != "WORKING_VALUE_ONBOARD":
    rename_map[working_col] = "WORKING_VALUE_ONBOARD"
if actual_status_col and actual_status_col != "Actual_Status":
    rename_map[actual_status_col] = "Actual_Status"

df_for_eval = df.rename(columns=rename_map)

# Display sanitized preview
st.subheader("Sanitized data preview (first 10 rows)")
st.dataframe(df_for_eval.head(10))

# Call evaluator
try:
    results = evaluate_dataframe(df_for_eval)
except Exception as e:
    st.error(f"Evaluator error: {e}")
    st.stop()

# ensure Status column exists (some evaluator versions named it 'Status' or 'Predicted_Status')
if "Status" not in results.columns:
    if "Predicted_Status" in results.columns:
        results = results.rename(columns={"Predicted_Status": "Status"})
    else:
        # derive from Health_Index
        if "Health_Index" in results.columns:
            results["Status"] = pd.cut(results["Health_Index"], bins=[-np.inf, 0.5, 0.8, np.inf], labels=["Critical", "Warning", "Healthy"])
        else:
            st.error("Evaluator did not return 'Health_Index' and no 'Status' could be derived.")
            st.stop()

# Show computed results
st.subheader("Computed Health Index & Status (sample)")
st.dataframe(results.head(10))

# Visualizations
st.subheader("Visualizations")

chart_paths = {}

# HI distribution
if "Health_Index" in results.columns:
    fig1, ax1 = plt.subplots(figsize=(8,4))
    sns.histplot(results["Health_Index"].dropna(), bins=10, kde=True, ax=ax1)
    ax1.set_title("Health Index Distribution")
    st.pyplot(fig1)
    chart_paths["Health Index Distribution"] = save_figure_temp(fig1)
else:
    st.info("No Health_Index values available for distribution chart.")

# Status pie
if "Status" in results.columns:
    fig2, ax2 = plt.subplots(figsize=(5,5))
    results["Status"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax2)
    ax2.set_ylabel("")
    ax2.set_title("Equipment Status Breakdown")
    st.pyplot(fig2)
    chart_paths["Status Breakdown"] = save_figure_temp(fig2)
else:
    st.info("No Status column available for pie chart.")

# Working value vs HI if available
if "WORKING_VALUE_ONBOARD" in results.columns or "SYNTHETIC_VALUE" in results.columns:
    work_col = "WORKING_VALUE_ONBOARD" if "WORKING_VALUE_ONBOARD" in results.columns else None
    synth_col = "SYNTHETIC_VALUE" if "SYNTHETIC_VALUE" in results.columns else None
    if synth_col:
        fig3, ax3 = plt.subplots(figsize=(8,5))
        sns.scatterplot(x=results[synth_col], y=results["Health_Index"], hue=results["Status"], ax=ax3)
        ax3.set_xlabel(synth_col)
        ax3.set_ylabel("Health_Index")
        ax3.set_title("Working (synthetic) value vs Health Index")
        st.pyplot(fig3)
        chart_paths["Working vs Health Index"] = save_figure_temp(fig3)

# Average HI per SYSTEM
if "SYSTEM" in results.columns and "Health_Index" in results.columns:
    avg_hi = results.groupby("SYSTEM")["Health_Index"].mean().reset_index().sort_values("Health_Index")
    fig4, ax4 = plt.subplots(figsize=(10,5))
    sns.barplot(data=avg_hi, x="SYSTEM", y="Health_Index", ax=ax4)
    ax4.set_title("Average Health Index by System")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig4)
    chart_paths["Average HI per System"] = save_figure_temp(fig4)

# Health Index by Indicator
if "HEALTH_INDICATOR_HI" in results.columns and "Health_Index" in results.columns:
    fig5, ax5 = plt.subplots(figsize=(10,6))
    sns.barplot(data=results, y="HEALTH_INDICATOR_HI", x="Health_Index", orient="h", ax=ax5)
    ax5.set_title("Health Index by Indicator")
    plt.tight_layout()
    st.pyplot(fig5)
    chart_paths["HI by Indicator"] = save_figure_temp(fig5)

# Metrics
st.subheader("Model Performance Metrics")
metrics = {
    "Total Readings": int(len(df_for_eval)),
    "Evaluated Readings": int(results["Health_Index"].notna().sum()) if "Health_Index" in results.columns else 0,
    "Coverage (%)": round((results["Health_Index"].notna().sum() / len(df_for_eval) * 100) if len(df_for_eval)>0 else 0, 2),
    "Mean HI": round(float(results["Health_Index"].mean()) if "Health_Index" in results.columns else np.nan, 3),
    "Std Dev HI": round(float(results["Health_Index"].std()) if "Health_Index" in results.columns else np.nan, 3),
    "Skewness": round(float(results["Health_Index"].skew()) if "Health_Index" in results.columns else np.nan, 3),
    "Kurtosis": round(float(results["Health_Index"].kurtosis()) if "Health_Index" in results.columns else np.nan, 3)
}
st.table(pd.DataFrame(list(metrics.items()), columns=["Metric","Value"]))

# Interpretation & maintenance suggestions
st.subheader("Interpretation & Suggested Actions")
interpretation = ""
if metrics["Mean HI"] > 0.8:
    interpretation = "Overall fleet health is excellent."
elif metrics["Mean HI"] > 0.6:
    interpretation = "Overall fleet health is satisfactory; schedule preventive maintenance."
else:
    interpretation = "Overall fleet health is poor; initiate immediate maintenance."

st.write(interpretation)

# generate suggestions
suggestions = []
if results["Status"].value_counts().get("Critical", 0) > 0:
    suggestions.append("Immediate inspection for Critical systems.")
if results["Status"].value_counts().get("Warning", 0) > 0:
    suggestions.append("Schedule maintenance for Warning systems.")
if not suggestions:
    suggestions.append("No urgent maintenance required; continue routine checks.")
for s in suggestions:
    st.markdown(f"- {s}")

# Maintenance schedule estimation (app display)
mean_hi = metrics["Mean HI"]
if mean_hi >= 0.8:
    next_dt = datetime.now() + timedelta(days=60)
    interval = "60 days (Excellent)"
    st.success(f"Recommended next inspection: {next_dt.strftime('%Y-%m-%d')} (60 days)")
elif mean_hi >= 0.6:
    next_dt = datetime.now() + timedelta(days=30)
    interval = "30 days (Satisfactory)"
    st.warning(f"Recommended next inspection: {next_dt.strftime('%Y-%m-%d')} (30 days)")
else:
    next_dt = datetime.now() + timedelta(days=7)
    interval = "7 days (Immediate)"
    st.error(f"Recommended next inspection: {next_dt.strftime('%Y-%m-%d')} (7 days) - Immediate action required")

# Update maintenance log
df_log = update_maintenance_log(metrics["Mean HI"], next_dt, interval)
st.subheader("Maintenance Log (recent entries)")
st.dataframe(df_log.tail(10))
st.download_button("Download maintenance log CSV", df_log.to_csv(index=False).encode("utf-8"), "maintenance_log.csv", "text/csv")

# Generate PDF report
st.subheader("Downloadable Report")
pdf_btn = st.button("Generate PDF report (cover + metrics + suggestions + charts)")

if pdf_btn:
    # Ensure chart_paths keys are present; if missing put placeholder
    chart_paths_for_pdf = chart_paths.copy()
    # generate the pdf bytes
    pdf_io = generate_pdf_report(metrics, interpretation, suggestions, chart_paths_for_pdf)
    st.download_button("Download PDF report", pdf_io.getvalue(), "Naval_Machinery_Health_Report.pdf", "application/pdf")

# cleanup temporary chart files
for p in chart_paths.values():
    try:
        os.remove(p)
    except Exception:
        pass

st.caption("Report format: Cover page with title, subtitle, authors (Navy Capt. Daya Abdullahi & Dr. Awujoola Olalekan J), date/time, confidentiality. Subsequent pages contain metrics, interpretation, suggestions, charts, footer and page numbers.")
