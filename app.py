# app.py
# ==============================================================
# ⚓ Hybrid Predictive Maintenance App (Rule-based HI + RandomForest ML)
# ==============================================================

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))  # allow local src imports

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chardet
import pickle
from io import BytesIO
from datetime import datetime, timedelta
from pathlib import Path

# sklearn for metrics (ML comparison)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# local rule-based evaluator (must exist in src/evaluate_hi.py)
from evaluate_hi import evaluate_dataframe

# -------- Basic configuration --------
st.set_page_config(page_title="Naval Machinery Predictive Maintenance",
                   page_icon="⚓",
                   layout="wide")

DATA_DIR = "data"
LOG_PATH = os.path.join(DATA_DIR, "maintenance_log.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# -------- Helpers: I/O, sanitization, extraction --------
def detect_encoding(file_bytes):
    res = chardet.detect(file_bytes)
    return res.get("encoding") or "utf-8"

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace('\u2013', '-', regex=False)
        .str.replace('\xa0', '', regex=False)
        .str.replace(' ', '_')
        .str.replace('-', '_')
        .str.replace(r'[^0-9a-zA-Z_]', '', regex=True)
    )
    return df

def norm_key(s: str) -> str:
    return ''.join(ch.lower() for ch in str(s) if ch.isalnum())

def find_column(df: pd.DataFrame, candidates: list):
    # return first matching column from df for any candidate token
    cols_norm = {norm_key(c): c for c in df.columns}
    for cand in candidates:
        nk = norm_key(cand)
        if nk in cols_norm:
            return cols_norm[nk]
    # try fuzzy contains
    for cand in candidates:
        nk = norm_key(cand)
        for cn, col in cols_norm.items():
            if nk in cn or cn in nk:
                return col
    return None

import re
def extract_numeric(val):
    if pd.isna(val):
        return np.nan
    s = str(val)
    m = re.search(r"[-+]?\d*\.\d+|\d+", s)
    if m:
        try:
            return float(m.group(0))
        except:
            return np.nan
    return np.nan

def save_fig_to_temp(fig):
    tmp = BytesIO()
    fig.savefig(tmp, format="png", bbox_inches="tight")
    tmp.seek(0)
    plt.close(fig)
    return tmp

# -------- Maintenance log --------
def append_maintenance_log(mean_hi, next_date, interval_label):
    entry = {
        "Record_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Mean_Health_Index": round(float(mean_hi), 4) if not pd.isna(mean_hi) else "",
        "Next_Inspection_Date": next_date.strftime("%Y-%m-%d"),
        "Interval": interval_label
    }
    if os.path.exists(LOG_PATH):
        df_log = pd.read_csv(LOG_PATH)
    else:
        df_log = pd.DataFrame(columns=list(entry.keys()))
    df_log = pd.concat([df_log, pd.DataFrame([entry])], ignore_index=True)
    df_log.to_csv(LOG_PATH, index=False)
    return df_log

# -------- PDF generation (cover + metrics + charts + suggestions) --------
from fpdf import FPDF

def generate_pdf(metrics_dict, interpretation, suggestions, charts: dict, authors="Navy Capt. Daya Abdullahi & Dr. Awujoola Olalekan J"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Cover
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

    # Metrics & interpretation
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Model Performance Metrics", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "", 11)
    for k, v in metrics_dict.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)

    pdf.ln(6)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Interpretation", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7, interpretation)
    pdf.ln(6)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Maintenance Recommendations", ln=True)
    pdf.set_font("Arial", "", 11)
    for s in suggestions:
        pdf.multi_cell(0, 7, "- " + s)
    pdf.ln(6)

    # charts
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Analytical Visualizations", ln=True)
    pdf.ln(6)
    for title, buf in charts.items():
        pdf.set_font("Arial", "I", 11)
        pdf.cell(0, 8, title, ln=True)
        try:
            # write buffer to a temp file for FPDF.image
            tmp_path = f"/tmp/{title.replace(' ','_')}.png"
            with open(tmp_path, "wb") as f:
                f.write(buf.getvalue())
            pdf.image(tmp_path, x=10, w=190)
            os.remove(tmp_path)
        except Exception as e:
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 6, f"[Could not embed chart: {e}]", ln=True)
        pdf.ln(6)

    # page numbers
    n_pages = pdf.page_no()
    for p in range(1, n_pages + 1):
        pdf.page = p
        pdf.set_y(-12)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 10, f"Generated by Predictive Maintenance System | Page {p} of {n_pages}", 0, 0, "C")

    out = BytesIO()
    pdf.output(out)
    out.seek(0)
    return out

# -------- UI: Sidebar, uploads, mode selection --------
st.title("⚓ Naval Machinery Predictive Maintenance System")
st.caption("Hybrid system: Rule-based Health Index (HI) and Random Forest ML predictions. Project Director Navy Capt Daya Abdullahi and Model Developer Dr. Awujoola Olalekan")

st.sidebar.header("Actions")
mode = st.sidebar.radio("Choose mode:", ["Rule-based (HI thresholds)", "Machine Learning (Random Forest)", "Compare both"])

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel dataset", type=["csv", "xlsx"])

# -------- Load ML model bundle if available --------
MODEL_PATH = Path("src/rf_model.pkl")
ml_available = MODEL_PATH.exists()
model_bundle = None
if ml_available:
    try:
        with open(MODEL_PATH, "rb") as f:
            model_bundle = pickle.load(f)
        # Expecting a dict with keys: model, scaler, label_encoder, feature_columns (optional)
        if not isinstance(model_bundle, dict) or "model" not in model_bundle:
            st.sidebar.warning("rf_model.pkl found but in unexpected format. Expected dict with 'model'. ML mode may fail.")
    except Exception as e:
        st.sidebar.warning(f"Unable to load ML model: {e}")
        ml_available = False

if not ml_available:
    st.sidebar.info("No ML model found in src/rf_model.pkl — ML modes are disabled until a model bundle is added.")

# -------- No file uploaded yet --------
if uploaded_file is None:
    st.info("Please upload your dataset (CSV or XLSX). Use Example Dataset in sidebar if needed.")
    # provide example button
    if st.sidebar.button("Download example dataset"):
        example = pd.DataFrame({
            "SYSTEM": ["Propulsion", "Cooling", "Auxiliary"],
            "EQUIPMENT": ["Main Engine", "Cooling Pump", "Generator"],
            "HEALTH INDICATOR (HI)": ["Lube oil pressure (bar)", "Oil temp (°C)", "Voltage (V)"],
            "SYNTHETIC VALUE": [4.2, 78.0, 440.0],
            "MIN–MAX THRESHOLDS": ["3.5 – 6.0", "60 – 90", "420 – 450"],
            "WORKING VALUE ONBOARD": ["3.6 bar", "82 °C", "438 V"],
            "REMARKS": ["Normal", "Slightly high", "Nominal"],
            "Actual_Status": ["Healthy", "Warning", "Healthy"]
        })
        st.sidebar.download_button("Download example CSV", example.to_csv(index=False).encode("utf-8"), "example.csv", "text/csv")
    st.stop()

# -------- Read uploaded file robustly --------
try:
    raw = uploaded_file.read()
    enc = detect_encoding(raw)
    uploaded_file.seek(0)
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding=enc, on_bad_lines="skip")
    else:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

# sanitize columns
df = sanitize_columns(df)

# Attempt to find core columns
synthetic_col = find_column(df, ["SYNTHETIC_VALUE", "SYNTHETIC VALUE", "WORKING_VALUE_ONBOARD", "WORKING VALUE ONBOARD", "WORKING_VALUE", "WORKING VALUE"])
minmax_col = find_column(df, ["MIN_MAX_THRESHOLDS", "MIN–MAX_THRESHOLDS", "MIN_MAX", "MIN_MAX_THRESHOLDS", "THRESHOLDS"])
hi_indicator_col = find_column(df, ["HEALTH_INDICATOR_HI", "HEALTH_INDICATOR", "HEALTH_INDICATOR_(HI)"])
working_col = find_column(df, ["WORKING_VALUE_ONBOARD", "WORKING VALUE ONBOARD", "WORKING_VALUE", "WORKINGVALUE"])
actual_status_col = find_column(df, ["ACTUAL_STATUS", "Actual_Status", "ActualStatus", "Actual_Status".lower()])

# Ensure minmax exists
if minmax_col is None:
    st.error("Missing MIN–MAX THRESHOLDS column. Please include the 'MIN–MAX THRESHOLDS' column in your dataset.")
    st.stop()

# If synthetic numeric missing, try to extract from working_col
if synthetic_col is None or synthetic_col not in df.columns:
    if working_col is not None:
        df["SYNTHETIC_VALUE"] = df[working_col].apply(extract_numeric)
        synthetic_col = "SYNTHETIC_VALUE"
        st.info("SYNTHETIC_VALUE not found; numeric values extracted from WORKING_VALUE_ONBOARD.")
    else:
        st.error("No numeric reading column found (SYNTHETIC_VALUE or WORKING_VALUE_ONBOARD).")
        st.stop()
else:
    # coerce to numeric
    df[synthetic_col] = pd.to_numeric(df[synthetic_col], errors="coerce")
    if synthetic_col != "SYNTHETIC_VALUE":
        df = df.rename(columns={synthetic_col: "SYNTHETIC_VALUE"})
        synthetic_col = "SYNTHETIC_VALUE"

# normalize minmax column name
if minmax_col != "MIN_MAX_THRESHOLDS":
    df = df.rename(columns={minmax_col: "MIN_MAX_THRESHOLDS"})
    minmax_col = "MIN_MAX_THRESHOLDS"

# rename indicator and working columns if found
if hi_indicator_col and hi_indicator_col != "HEALTH_INDICATOR_HI":
    df = df.rename(columns={hi_indicator_col: "HEALTH_INDICATOR_HI"})
if working_col and working_col != "WORKING_VALUE_ONBOARD":
    df = df.rename(columns={working_col: "WORKING_VALUE_ONBOARD"})
if actual_status_col and actual_status_col != "Actual_Status":
    df = df.rename(columns={actual_status_col: "Actual_Status"})

# show preview
st.subheader("Sanitized dataset preview")
st.dataframe(df.head(8))

# -------- Run rule-based evaluator --------
rule_results = None
try:
    rule_results = evaluate_dataframe(df)
    # make sure it has columns Health_Index and Status
    if "Health_Index" not in rule_results.columns:
        st.warning("Rule evaluator returned no Health_Index. The rule-based output may be incomplete.")
except Exception as e:
    st.error(f"Rule-based evaluator error: {e}")

# -------- Run ML prediction if requested and available --------
ml_results = None
if mode in ["Machine Learning (Random Forest)", "Compare both"]:
    if not ml_available:
        st.warning("ML model not available; please upload rf_model.pkl to src/ to enable ML mode.")
    else:
        # prepare numeric features for model
        try:
            model = model_bundle.get("model")
            scaler = model_bundle.get("scaler")
            le = model_bundle.get("label_encoder", None)
            feature_columns = model_bundle.get("feature_columns", None)  # optional list of feature names

            # Determine features
            if feature_columns and all(col in df.columns for col in feature_columns):
                X = df[feature_columns].copy()
            else:
                # fallback to numeric columns in df
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    st.error("No numeric features available for ML prediction.")
                X = df[numeric_cols].copy()
                feature_columns = numeric_cols

            # Fill NA and scale
            X = X.fillna(X.mean())
            if scaler is not None:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X.values

            y_pred = model.predict(X_scaled)
            if le is not None:
                y_pred_labels = le.inverse_transform(y_pred)
            else:
                y_pred_labels = y_pred.astype(str)

            ml_results = df.copy()
            ml_results["ML_Predicted_Status"] = y_pred_labels

            # If model gives probabilities, include them
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled)
                # keep class-proba mapping if label encoder present
                if le is not None:
                    classes = le.inverse_transform(np.arange(proba.shape[1]))
                else:
                    classes = [f"class_{i}" for i in range(proba.shape[1])]
                for idx, cls in enumerate(classes):
                    ml_results[f"prob_{cls}"] = proba[:, idx]

            st.success("ML prediction completed.")
        except Exception as e:
            st.error(f"Error during ML prediction: {e}")
            ml_results = None

# -------- Display outputs depending on mode --------
st.header("Results")

# Utility: create charts for PDF (store buffers)
chart_buffers = {}

def generate_common_visuals(results_df, prefix=""):
    bufs = {}
    # HI distribution
    if "Health_Index" in results_df.columns:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(results_df["Health_Index"].dropna(), bins=10, kde=True, ax=ax)
        ax.set_title("Health Index Distribution")
        buf = save_fig_to_temp(fig) if False else save_fig_buf(fig)
        plt.close(fig)
    return bufs

# small helper using BytesIO (we used save_fig_to_temp earlier; define alternative)
def save_fig_buf(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

# show rule-based if selected
if mode in ["Rule-based (HI thresholds)", "Compare both"]:
    st.subheader("Rule-based Health Index Evaluation")
    if rule_results is None:
        st.error("Rule-based evaluation not available.")
    else:
        st.dataframe(rule_results.head(10))

        # Visuals for rule-based
        col1, col2 = st.columns(2)
        with col1:
            if "Health_Index" in rule_results.columns:
                fig1, ax1 = plt.subplots()
                sns.histplot(rule_results["Health_Index"].dropna(), bins=10, kde=True, ax=ax1)
                ax1.set_title("Health Index Distribution")
                st.pyplot(fig1)
                chart_buffers["Rule_HI_Distribution"] = save_fig_buf(fig1)
            else:
                st.info("No Health_Index for rule-based results.")

        with col2:
            if "Status" in rule_results.columns:
                fig2, ax2 = plt.subplots()
                rule_results["Status"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax2)
                ax2.set_ylabel("")
                ax2.set_title("Status Breakdown (Rule-based)")
                st.pyplot(fig2)
                chart_buffers["Rule_Status_Breakdown"] = save_fig_buf(fig2)
            else:
                st.info("No Status for rule-based results.")

        # HI by indicator
        if "HEALTH_INDICATOR_HI" in rule_results.columns and "Health_Index" in rule_results.columns:
            fig3, ax3 = plt.subplots(figsize=(10,6))
            sns.barplot(data=rule_results, y="HEALTH_INDICATOR_HI", x="Health_Index", orient="h", ax=ax3)
            ax3.set_title("Health Index by Indicator (Rule-based)")
            st.pyplot(fig3)
            chart_buffers["Rule_HI_by_Indicator"] = save_fig_buf(fig3)

        # Average HI per System heatmap
        if "SYSTEM" in rule_results.columns and "Health_Index" in rule_results.columns:
            sys_hi = rule_results.groupby("SYSTEM")["Health_Index"].mean().to_frame()
            fig4, ax4 = plt.subplots(figsize=(8,4))
            sns.heatmap(sys_hi, annot=True, cmap="YlGnBu", linewidths=0.5, ax=ax4)
            ax4.set_title("Average Health Index per System (Rule-based)")
            st.pyplot(fig4)
            chart_buffers["Rule_Avg_HI_per_System"] = save_fig_buf(fig4)

        # Scatter: Working vs HI
        if "SYNTHETIC_VALUE" in rule_results.columns and "Health_Index" in rule_results.columns:
            fig5, ax5 = plt.subplots(figsize=(8,5))
            sns.scatterplot(x=rule_results["SYNTHETIC_VALUE"], y=rule_results["Health_Index"], hue=rule_results.get("Status"), ax=ax5)
            ax5.set_title("Working Value vs Health Index (Rule-based)")
            st.pyplot(fig5)
            chart_buffers["Rule_Working_vs_HI"] = save_fig_buf(fig5)

# show ML results if selected
if mode in ["Machine Learning (Random Forest)", "Compare both"]:
    st.subheader("Machine Learning Prediction (Random Forest)")
    if ml_results is None:
        st.warning("ML results not available.")
    else:
        st.dataframe(ml_results.head(10))

        col1, col2 = st.columns(2)
        with col1:
            # predicted distribution
            figm1, axm1 = plt.subplots()
            ml_results["ML_Predicted_Status"].value_counts().plot(kind="bar", ax=axm1)
            axm1.set_title("ML Predicted Status Distribution")
            st.pyplot(figm1)
            chart_buffers["ML_Status_Bar"] = save_fig_buf(figm1)

        with col2:
            # if actual status exists, compute confusion/accuracy
            if "Actual_Status" in ml_results.columns:
                y_true = ml_results["Actual_Status"].astype(str)
                y_pred = ml_results["ML_Predicted_Status"].astype(str)
                try:
                    acc = accuracy_score(y_true, y_pred)
                    st.metric("ML Accuracy", f"{acc:.3f}")
                    cm = confusion_matrix(y_true, y_pred, labels=np.unique(np.concatenate([y_true.unique(), y_pred.unique()])))
                    figcm, axcm = plt.subplots(figsize=(5,4))
                    sns.heatmap(cm, annot=True, fmt="d", ax=axcm,
                                xticklabels=np.unique(np.concatenate([y_true.unique(), y_pred.unique()])),
                                yticklabels=np.unique(np.concatenate([y_true.unique(), y_pred.unique()])))
                    axcm.set_xlabel("Predicted")
                    axcm.set_ylabel("Actual")
                    axcm.set_title("Confusion Matrix (ML)")
                    st.pyplot(figcm)
                    chart_buffers["ML_Confusion_Matrix"] = save_fig_buf(figcm)
                except Exception as e:
                    st.info(f"Could not compute ML metrics: {e}")
            else:
                st.info("Actual_Status not present — cannot compute ML accuracy/confusion matrix.")

# Compare both side-by-side if chosen
if mode == "Compare both":
    st.subheader("Comparison: Rule-based vs ML (side-by-side)")

    if rule_results is None or ml_results is None:
        st.info("Both results are required for comparison. Ensure ML model is available and runable.")
    else:
        # merge by index
        comp = rule_results.copy()
        comp = comp.rename(columns={"Status": "Rule_Status", "Health_Index": "Rule_Health_Index"})
        comp["ML_Predicted_Status"] = ml_results["ML_Predicted_Status"].values
        # show sample and comparison metrics
        st.dataframe(comp.head(10))

        # If Actual_Status exists, compute per-mode accuracy
        if "Actual_Status" in comp.columns:
            y_true = comp["Actual_Status"].astype(str)
            rule_y = comp["Rule_Status"].astype(str)
            ml_y = comp["ML_Predicted_Status"].astype(str)
            try:
                rule_acc = accuracy_score(y_true, rule_y)
                ml_acc = accuracy_score(y_true, ml_y)
                st.write(f"Rule-based Accuracy: {rule_acc:.3f} | ML Accuracy: {ml_acc:.3f}")
                st.write("Classification report (ML):")
                st.text(classification_report(y_true, ml_y))
            except Exception as e:
                st.info(f"Could not compute comparison metrics: {e}")
        else:
            st.info("No Actual_Status to compute accuracy. Comparison is limited to predicted labels.")

# -------- Metrics summary for report, maintenance schedule, and log --------
# Create metrics using rule_results as base if present, otherwise ML
base_results = rule_results if rule_results is not None else (ml_results if ml_results is not None else None)

if base_results is not None and "Health_Index" in base_results.columns:
    mean_hi = float(base_results["Health_Index"].mean())
else:
    mean_hi = np.nan

metrics_for_report = {
    "Total Readings": int(len(df)),
    "Evaluated Readings": int(base_results["Health_Index"].notna().sum()) if base_results is not None and "Health_Index" in base_results.columns else 0,
    "Coverage (%)": round((base_results["Health_Index"].notna().sum() / len(df) * 100) if base_results is not None and len(df)>0 else 0, 2),
    "Mean HI": round(mean_hi, 4) if not np.isnan(mean_hi) else "N/A",
    "Std Dev HI": round(float(base_results["Health_Index"].std()), 4) if base_results is not None and "Health_Index" in base_results.columns else "N/A",
    "Skewness": round(float(base_results["Health_Index"].skew()), 4) if base_results is not None and "Health_Index" in base_results.columns else "N/A",
    "Kurtosis": round(float(base_results["Health_Index"].kurtosis()), 4) if base_results is not None and "Health_Index" in base_results.columns else "N/A"
}

# Interpretation & suggestions
if not pd.isna(mean_hi):
    if mean_hi >= 0.8:
        interpretation = "Overall fleet health is excellent."
        next_dt = datetime.now() + timedelta(days=60)
        interval_label = "60 days"
    elif mean_hi >= 0.6:
        interpretation = "Overall fleet health is satisfactory; schedule preventive maintenance."
        next_dt = datetime.now() + timedelta(days=30)
        interval_label = "30 days"
    else:
        interpretation = "Overall fleet health is poor; immediate maintenance required."
        next_dt = datetime.now() + timedelta(days=7)
        interval_label = "7 days"
else:
    interpretation = "Insufficient data to interpret."
    next_dt = datetime.now()
    interval_label = "N/A"

# generate suggestions list
suggestions = []
if base_results is not None and "Status" in base_results.columns:
    if base_results["Status"].value_counts().get("Critical", 0) > 0:
        suggestions.append("Immediate inspection required for systems flagged Critical.")
    if base_results["Status"].value_counts().get("Warning", 0) > 0:
        suggestions.append("Schedule maintenance for Warning systems.")
if not suggestions:
    suggestions.append("No urgent maintenance required; continue routine inspections.")

# append maintenance log
df_log = append_maintenance_log(mean_hi, next_dt, interval_label)

# display maintenance schedule and log
st.subheader("Maintenance Schedule Recommendation")
if not pd.isna(mean_hi):
    if mean_hi >= 0.8:
        st.success(f"Next inspection recommended: {next_dt.strftime('%Y-%m-%d')} (60 days)")
    elif mean_hi >= 0.6:
        st.warning(f"Next inspection recommended: {next_dt.strftime('%Y-%m-%d')} (30 days)")
    else:
        st.error(f"Next inspection recommended: {next_dt.strftime('%Y-%m-%d')} (7 days) - Immediate action required")
else:
    st.info("No maintenance schedule due to insufficient data.")

st.subheader("Maintenance Log (most recent entries)")
if os.path.exists(LOG_PATH):
    st.dataframe(pd.read_csv(LOG_PATH).tail(10))
    st.download_button("Download maintenance log (CSV)", pd.read_csv(LOG_PATH).to_csv(index=False).encode("utf-8"), "maintenance_log.csv", "text/csv")
else:
    st.info("No maintenance log yet.")

# -------- Report generation --------
st.subheader("Download Unified PDF Report")

if st.button("Generate PDF Report (cover + metrics + suggestions + charts)"):
    # choose which charts to embed: prefer rule charts then ml charts
    charts_for_pdf = {}
    for k, v in chart_buffers.items():
        charts_for_pdf[k] = v
    pdf_bytes = generate_pdf(metrics_for_report, interpretation, suggestions, charts_for_pdf)
    st.download_button("Click to download PDF", pdf_bytes.getvalue(), "Naval_Machinery_Health_Report.pdf", "application/pdf")

# cleanup (nothing left to remove since we used in-memory buffers)
st.caption("Report: Cover page with title & authors, subsequent pages with metrics, interpretation, suggestions, charts, footer & page numbers.")
