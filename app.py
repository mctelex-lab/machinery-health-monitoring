# ==============================================================
# ⚓ Equipment Health Condition Monitoring Prediction for Naval Ships
# Streamlit Web Application (Final Enhanced Version)
# ==============================================================
import sys
import os
import tempfile
from datetime import datetime
from io import BytesIO

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

from evaluate_hi import evaluate_dataframe  # ensure src/evaluate_hi.py exists

# ==============================================================
# PAGE CONFIGURATION
# ==============================================================
st.set_page_config(
    page_title="⚓ Naval Equipment Health Monitoring System",
    layout="wide",
    page_icon="⚓"
)

st.title("⚓ Equipment Health Condition Monitoring Prediction for Naval Ships")
st.caption(
    "A predictive maintenance tool for analyzing machinery health onboard naval vessels. "
    "Upload operational data (CSV/XLSX), compute Health Index (HI), visualize insights, "
    "and download a confidential PDF report."
)

# ==============================================================
# SIDEBAR
# ==============================================================
with st.sidebar:
    st.header("📘 Resources & Actions")
    st.markdown("[🌐 View Source on GitHub](https://github.com/mctelex-lab/manual-health-monitoring)")
    st.markdown("---")
    st.header("🧩 App Features")
    st.markdown(
        "- 📊 Automatic Health Index computation\n"
        "- ⚙️ Multi-system & indicator analysis\n"
        "- 🧭 Predicted condition (Healthy/Warning/Critical)\n"
        "- 📄 Auto-generated PDF report with charts\n"
        "- ⬇️ Exportable CSV of results"
    )
    st.markdown("---")
    st.header("🆘 About")
    st.info("Developed by **Dr. Awujoola Olalekan** — Predictive Maintenance and Naval Engineering.")

# ==============================================================
# EXAMPLE DATASET GENERATOR
# ==============================================================
st.subheader("📁 Example Dataset Generator")
st.markdown("Download a ready-to-use CSV sample if you don’t have a dataset yet.")

if st.button("📦 Generate Example Dataset"):
    sample_data = {
        "SYSTEM": ["Propulsion", "Propulsion", "Auxiliary", "Cooling", "Hydraulic"],
        "EQUIPMENT": ["Main Engine", "Gearbox", "Generator", "Sea Water Pump", "Rudder Actuator"],
        "HEALTH_INDICATOR_HI": [
            "Lube oil pressure (bar)", "Oil temp (°C)", "Voltage (V)", "Flow (m³/h)", "Hydraulic pressure (bar)"
        ],
        "SYNTHETIC_VALUE": [4.2, 78.0, 440, 88.0, 150.0],
        "MIN_MAX_THRESHOLDS": ["3.5 – 6.0", "60 – 85", "420 – 450", "70 – 100", "135 – 160"],
        "WORKING_VALUE_ONBOARD": ["3.6 bar", "82 °C", "438 V", "90 m³/h", "158 bar"],
        "REMARKS": ["Normal", "Slightly high", "Nominal", "Nominal", "Normal"],
        "Actual_Status": ["Healthy", "Warning", "Healthy", "Healthy", "Warning"]
    }
    df_example = pd.DataFrame(sample_data)
    buffer = BytesIO()
    df_example.to_csv(buffer, index=False)
    st.download_button(
        label="⬇️ Download Example CSV",
        data=buffer.getvalue(),
        file_name="example_machinery_health_data.csv",
        mime="text/csv"
    )
    st.success("✅ Example dataset ready for download!")

st.markdown("---")

# ==============================================================
# FILE UPLOAD
# ==============================================================
uploaded_file = st.file_uploader("📂 Upload your CSV or Excel file", type=["csv", "xlsx"])

# ==============================================================
# HELPER FUNCTIONS
# ==============================================================
def interpret_results(results_df):
    if "Predicted_Status" not in results_df.columns:
        return "ℹ️ No predictions available to interpret."
    counts = results_df["Predicted_Status"].value_counts().to_dict()
    notes = []
    if counts.get("Critical", 0) > 0:
        notes.append("⚠️ Critical: Immediate inspection required for critical equipment.")
    if counts.get("Warning", 0) > 0:
        notes.append("🔧 Warning: Schedule preventive maintenance for affected systems.")
    if counts.get("Healthy", 0) > 0:
        notes.append("✅ Healthy: Most systems are stable. Continue periodic monitoring.")
    return "\n".join(notes) if notes else "ℹ️ No valid readings available to interpret."

def save_figure_to_temp(fig, suffix=".png"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    fig.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

def generate_pdf_report_with_charts(metrics_df, interpretation, chart_paths):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ================= COVER PAGE =================
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "⚓ NAVAL MACHINERY HEALTH CONDITION REPORT", ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Arial", "", 14)
    pdf.cell(0, 10, "Predictive Maintenance Model Summary", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", "I", 11)
    pdf.multi_cell(0, 8, "Automated Equipment Condition Assessment for Naval Ships", align="C")
    pdf.ln(8)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Prepared by: Navy Capt. Daya Abdullahi & Dr. Awujoola Olalekan J", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(
        0, 7,
        "Confidential: This report contains sensitive technical and operational data pertaining to naval vessel equipment. "
        "Unauthorized distribution or duplication is strictly prohibited.",
        align="C"
    )

    # ================= DETAILED REPORT =================
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "📊 Model Performance Metrics", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "", 11)
    for _, row in metrics_df.iterrows():
        val = "N/A" if pd.isna(row["Value"]) else f"{row['Value']:.3f}"
        pdf.cell(0, 8, f"{row['Metric']}: {val}", ln=True)
    pdf.ln(6)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "🧠 Interpretation & Recommendations", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7, interpretation)
    pdf.ln(6)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "📈 Analytical Visualizations", ln=True)
    pdf.ln(4)

    for path in chart_paths:
        try:
            pdf.image(path, x=10, w=190)
            pdf.ln(6)
        except:
            continue

    pdf.ln(4)
    pdf.set_font("Arial", "I", 9)
    pdf.cell(0, 8, "Generated using Predictive Maintenance AI System", ln=True)
    pdf.cell(0, 8, "Developed by Dr. Awujoola Olalekan", ln=True)

    # PAGE NUMBERING
    for i in range(1, pdf.page_no() + 1):
        pdf.page = i
        pdf.set_y(-15)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 10, f"Page {i} of {pdf.page_no()}", 0, 0, "C")

    output = BytesIO()
    pdf.output(output)
    return output.getvalue()

# ==============================================================
# MAIN APP LOGIC
# ==============================================================
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.subheader("📋 Uploaded Data Preview")
        st.dataframe(df.head())

        results = evaluate_dataframe(df)
        st.subheader("🔍 Computed Health Index and Predicted Status")
        st.dataframe(results.head(10))

        # METRICS
        total = len(results)
        evaluated = results["Health_Index"].notna().sum()
        coverage = (evaluated / total * 100) if total else 0
        mean_hi = results["Health_Index"].mean()
        std_hi = results["Health_Index"].std()
        skewness = results["Health_Index"].skew()
        kurtosis = results["Health_Index"].kurtosis()

        metrics = pd.DataFrame({
            "Metric": ["Total Readings", "Evaluated Readings", "Coverage (%)", "Mean HI", "Std Dev HI", "Skewness", "Kurtosis"],
            "Value": [total, evaluated, coverage, mean_hi, std_hi, skewness, kurtosis]
        })
        st.subheader("📊 Model Performance Metrics")
        st.dataframe(metrics.style.format(precision=3))

        # CHARTS
        chart_paths = []

        # HI Distribution
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.histplot(results["Health_Index"], bins=10, kde=True, color="teal", ax=ax1)
        ax1.set_xlabel("Health Index (HI)")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)
        chart_paths.append(save_figure_to_temp(fig1))

        # Status Pie Chart
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        results["Predicted_Status"].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, ax=ax2)
        ax2.set_ylabel("")
        st.pyplot(fig2)
        chart_paths.append(save_figure_to_temp(fig2))

        # Average HI per System
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=results, x="SYSTEM", y="Health_Index", ax=ax3, palette="coolwarm")
        plt.xticks(rotation=45)
        st.pyplot(fig3)
        chart_paths.append(save_figure_to_temp(fig3))

        # Interpretation
        interpretation = interpret_results(results)
        st.subheader("🧠 Model Interpretation & Suggested Actions")
        st.markdown(interpretation)

        # PDF Report
        st.subheader("📄 Generate PDF Report")
        if st.button("🖨️ Download Full Report as PDF"):
            pdf_data = generate_pdf_report_with_charts(metrics, interpretation, chart_paths)
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_data,
                file_name="Naval_Health_Monitoring_Report.pdf",
                mime="application/pdf"
            )

        # Cleanup temporary chart files
        for path in chart_paths:
            try:
                os.remove(path)
            except:
                pass

        # CSV Download
        st.subheader("⬇️ Download Evaluated Results")
        buffer = BytesIO()
        results.to_csv(buffer, index=False)
        st.download_button("Download Results CSV", buffer.getvalue(), "evaluated_health_results.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("Please upload a `.csv` or `.xlsx` file to start analysis.")
