# ==============================================================
# ⚓ Naval Machinery Health Monitoring Web App
# ==============================================================
# Rule-based + Machine Learning Health Prediction
# Generates charts, performance metrics, maintenance reports (PDF)
# and logs all evaluations for historical tracking.
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io, os, re, pickle
from datetime import datetime
from pathlib import Path
from scipy.stats import skew, kurtosis
from fpdf import FPDF

# Allow importing from local src folder
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# ==============================================================
# 1️⃣ LOAD TRAINED MODEL
# ==============================================================

MODEL_PATH = Path("src/rf_model.pkl")
model_bundle = None
if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model_bundle = pickle.load(f)

# ==============================================================
# 2️⃣ PAGE CONFIG
# ==============================================================

st.set_page_config(
    page_title="Naval Machinery Health Monitoring System",
    layout="wide",
    page_icon="⚓"
)
st.title("⚓ Equipment Health Condition Monitoring for Naval Ships")
st.caption("Developed by the Naval Data Science Team (Capt. Daya Abdullahi & Dr. Awujoola Olalekan J)")

# ==============================================================
# 3️⃣ FILE UPLOAD
# ==============================================================

st.sidebar.header("Upload Machinery Data")
uploaded = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

mode = st.sidebar.radio("Choose Evaluation Mode:", ["Automatic Rule-based", "Machine Learning Prediction (Random Forest)"])

# ==============================================================
# 4️⃣ DATA LOADING & CLEANING
# ==============================================================

def load_dataset(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file, encoding="latin-1", on_bad_lines="skip")
    else:
        import openpyxl
        df = pd.read_excel(file, engine="openpyxl")

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
    )
    return df

def extract_min_max(s):
    if pd.isna(s):
        return (np.nan, np.nan)
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(s))
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    elif len(nums) == 1:
        return float(nums[0]), np.nan
    else:
        return (np.nan, np.nan)

def compute_health(row):
    val, low, high = row["Working_Value"], row["Min_Threshold"], row["Max_Threshold"]
    if np.isnan(val) or np.isnan(low) or np.isnan(high):
        return np.nan, "Unknown"
    if low <= val <= high:
        return 1.0, "Healthy"
    elif (low * 0.9) <= val <= (high * 1.1):
        return 0.5, "Warning"
    else:
        return 0.0, "Critical"

# ==============================================================
# 5️⃣ PDF GENERATION (UTF-8 SAFE)
# ==============================================================

def generate_pdf(metrics_df, interpretation, suggestions, charts):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=14)
    pdf.multi_cell(0, 10, "⚓ NAVAL MACHINERY HEALTH CONDITION REPORT", align="C")
    pdf.set_font("DejaVu", size=12)
    pdf.multi_cell(0, 8, "Predictive Maintenance Model Summary", align="C")
    pdf.ln(10)
    pdf.multi_cell(0, 8, "Authors: Navy Capt. Daya Abdullahi & Dr. Awujoola Olalekan J", align="C")
    pdf.multi_cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C")
    pdf.multi_cell(0, 8, "Confidential - For Maintenance Use Only", align="C")

    pdf.add_page()
    pdf.set_font("DejaVu", size=12)
    pdf.cell(0, 10, "Model Performance Metrics", ln=True)
    pdf.ln(5)
    for _, row in metrics_df.iterrows():
        pdf.cell(0, 8, f"{row['Metric']}: {row['Value']}", ln=True)

    pdf.ln(8)
    pdf.multi_cell(0, 8, "Interpretation:", align="L")
    pdf.multi_cell(0, 8, interpretation)
    pdf.ln(5)
    pdf.multi_cell(0, 8, "Suggested Maintenance Actions:", align="L")
    pdf.multi_cell(0, 8, suggestions)

    pdf.add_page()
    for name, chart in charts.items():
        img_path = Path(f"{name}.png")
        with open(img_path, "wb") as f:
            f.write(chart.getbuffer())
        pdf.image(str(img_path), x=10, w=180)
        pdf.ln(10)

    out_file = "naval_machinery_health_report.pdf"
    pdf.output(out_file)
    return out_file

# ==============================================================
# 6️⃣ MAIN LOGIC
# ==============================================================

if uploaded is not None:
    df = load_dataset(uploaded)

    # Derive numeric columns
    df[["Min_Threshold", "Max_Threshold"]] = df["MIN_MAX_THRESHOLDS"].apply(lambda x: pd.Series(extract_min_max(x)))
    df["Working_Value"] = (
        df["WORKING_VALUE_ONBOARD"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.extract(r"([-+]?\d*\.\d+|\d+)")
        .astype(float)
    )

    if mode == "Automatic Rule-based":
        df[["Health_Index", "Predicted_Status"]] = df.apply(lambda x: pd.Series(compute_health(x)), axis=1)

    elif mode == "Machine Learning Prediction (Random Forest)" and model_bundle:
        # Prepare features
        df[["Min_Threshold", "Max_Threshold"]] = df["MIN_MAX_THRESHOLDS"].apply(lambda x: pd.Series(extract_min_max(x)))
        df["Working_Value"] = (
            df["WORKING_VALUE_ONBOARD"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.extract(r"([-+]?\d*\.\d+|\d+)")
            .astype(float)
        )
        def calc_hi(row):
            v, lo, hi = row["Working_Value"], row["Min_Threshold"], row["Max_Threshold"]
            if np.isnan(v) or np.isnan(lo) or np.isnan(hi): return np.nan
            if lo <= v <= hi: return 1.0
            elif (lo * 0.9) <= v <= (hi * 1.1): return 0.5
            return 0.0
        df["Health_Index"] = df.apply(calc_hi, axis=1)

        X = df[["Working_Value", "Min_Threshold", "Max_Threshold", "Health_Index"]].fillna(0)
        X_scaled = model_bundle["scaler"].transform(X)
        preds = model_bundle["model"].predict(X_scaled)
        df["Predicted_Status"] = model_bundle["label_encoder"].inverse_transform(preds)

    # ==============================================================
    # 📊 METRICS
    # ==============================================================
    total = len(df)
    known = df["Health_Index"].notna().sum()
    coverage = known / total * 100
    hi_mean = df["Health_Index"].mean(skipna=True)
    hi_std = df["Health_Index"].std(skipna=True)
    hi_skew = skew(df["Health_Index"].dropna())
    hi_kurt = kurtosis(df["Health_Index"].dropna())

    metrics = pd.DataFrame({
        "Metric": ["Total Readings", "Evaluated Readings", "Coverage (%)", "Mean HI", "Std Dev HI", "Skewness", "Kurtosis"],
        "Value": [total, known, round(coverage,2), round(hi_mean,3), round(hi_std,3), round(hi_skew,3), round(hi_kurt,3)]
    })

    st.subheader("📈 Model Performance Metrics")
    st.dataframe(metrics, use_container_width=True)

    # ==============================================================
    # 📉 VISUALS
    # ==============================================================
    st.subheader("🔍 Visual Insights")
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    df["Predicted_Status"].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#f1c40f', '#e74c3c'])
    ax1.set_ylabel("")
    ax1.set_title("Overall Condition Distribution")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Health_Index"].dropna(), bins=10, kde=True, ax=ax2)
    ax2.set_title("Health Index Distribution")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.barplot(x="HEALTH_INDICATOR_HI", y="Health_Index", hue="Predicted_Status", data=df, ax=ax3)
    ax3.set_title("Health Index by Indicator")
    ax3.tick_params(axis='x', rotation=90)
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sys_hi = df.groupby("SYSTEM")["Health_Index"].mean().to_frame()
    sns.heatmap(sys_hi, annot=True, cmap="YlGnBu", linewidths=0.5, ax=ax4)
    ax4.set_title("Average Health Index per System")
    st.pyplot(fig4)

    # Scatter plot
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x="Working_Value", y="Health_Index", hue="Predicted_Status", data=df, ax=ax5)
    ax5.set_title("Working Value vs. Health Index")
    st.pyplot(fig5)

    # ==============================================================
    # 🧾 PDF REPORT
    # ==============================================================
    st.subheader("📘 Generate Detailed PDF Report")
    interpretation = f"The average Health Index is {hi_mean:.2f}, indicating an overall {'healthy' if hi_mean>0.8 else 'moderate' if hi_mean>0.5 else 'critical'} condition."
    suggestions = "Ensure routine inspection for all 'Warning' or 'Critical' systems. Schedule immediate maintenance for critical readings."

    charts = {"fig1": io.BytesIO(), "fig2": io.BytesIO(), "fig3": io.BytesIO()}
    fig1.savefig(charts["fig1"], format="png")
    fig2.savefig(charts["fig2"], format="png")
    fig3.savefig(charts["fig3"], format="png")

    pdf_path = generate_pdf(metrics, interpretation, suggestions, charts)
    with open(pdf_path, "rb") as pdf_file:
        st.download_button("📥 Download PDF Report", data=pdf_file, file_name=pdf_path, mime="application/pdf")

    # ==============================================================
    # 🧠 Maintenance Log
    # ==============================================================
    st.subheader("🧭 Maintenance Log")
    log_path = Path("maintenance_log.csv")
    log_entry = pd.DataFrame([{
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Mean_HI": round(hi_mean, 3),
        "Coverage_%": round(coverage, 2),
        "Evaluation_Mode": mode
    }])
    if log_path.exists():
        old_log = pd.read_csv(log_path)
        new_log = pd.concat([old_log, log_entry], ignore_index=True)
    else:
        new_log = log_entry
    new_log.to_csv(log_path, index=False)
    st.dataframe(new_log.tail(10), use_container_width=True)
    with open(log_path, "rb") as f:
        st.download_button("💾 Download Maintenance Log", f, file_name="maintenance_log.csv")

else:
    st.info("Please upload a dataset to begin analysis.")
