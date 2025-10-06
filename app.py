# ==============================================================
# ⚓ Equipment Health Condition Monitoring Prediction for Naval Ships
# Streamlit Web Application
# ==============================================================
# Upload CSV/Excel → Compute Health Index (HI)
# Visualize results → Generate interpretive PDF report
# ==============================================================

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from fpdf import FPDF
from evaluate_hi import evaluate_dataframe

# --------------------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------------------
st.set_page_config(
    page_title="⚓ Equipment Health Monitoring | Naval Ships",
    layout="wide",
    page_icon="⚓"
)

# --------------------------------------------------------------
# Title & Caption
# --------------------------------------------------------------
st.title("⚓ Equipment Health Condition Monitoring Prediction for Naval Ships")
st.caption("""
A predictive maintenance application for analyzing machinery health conditions onboard naval vessels.  
Upload operational data, compute Health Index (HI), visualize system health, and generate an interpretive performance report.
""")

# --------------------------------------------------------------
# Sidebar Section
# --------------------------------------------------------------
with st.sidebar:
    st.header("📘 Resources & Actions")
    st.markdown("[🌐 View Source on GitHub](https://github.com/mctelex-lab/manual-health-monitoring)")

    st.markdown("---")
    st.header("🧩 App Features")
    st.markdown("""
- 📊 Automatic Health Index computation  
- ⚙️ Multi-system & indicator analysis  
- 🧭 Predicted condition (Healthy/Warning/Critical)  
- 🧠 Auto-generated PDF report with interpretation  
- 📈 Visual analytics for decision-making  
""")

    st.markdown("---")
    st.header("🆘 About")
    st.info("Developed by **Dr. Awujoola Olalekan** for predictive maintenance and naval engineering analytics using Python and Streamlit.")

# --------------------------------------------------------------
# Example Dataset Generator
# --------------------------------------------------------------
st.subheader("📁 Example Dataset Generator")
st.markdown("Click below to download a ready-to-use dataset for testing the system.")

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

# --------------------------------------------------------------
# File Upload Section
# --------------------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV or Excel file", type=["csv", "xlsx"])

# --------------------------------------------------------------
# Helper: Generate interpretive suggestions
# --------------------------------------------------------------
def interpret_results(results_df):
    status_counts = results_df["Predicted_Status"].value_counts().to_dict()
    insights = []
    if status_counts.get("Critical", 0) > 0:
        insights.append("⚠️ Several systems are in *Critical* state — immediate inspection and possible shutdown are recommended.")
    if status_counts.get("Warning", 0) > 0:
        insights.append("🔧 Some equipment show *Warning* status — schedule preventive maintenance soon.")
    if status_counts.get("Healthy", 0) > 0:
        insights.append("✅ Most systems are *Healthy* — continue routine checks and oil analysis.")
    if not insights:
        insights.append("ℹ️ No valid readings available to interpret.")
    return "\n".join(insights)

# --------------------------------------------------------------
# Helper: Generate PDF Report
# --------------------------------------------------------------
def generate_pdf_report(metrics_df, interpretation_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "⚓ Equipment Health Condition Monitoring Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Summary of Model Performance Metrics", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "", 10)
    for i, row in metrics_df.iterrows():
        pdf.cell(0, 8, f"{row['Metric']}: {row['Value']:.3f}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Interpretation and Recommendations", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 8, interpretation_text)

    pdf.ln(10)
    pdf.set_font("Arial", "I", 9)
    pdf.cell(0, 8, "Generated using Predictive Maintenance AI System", ln=True)
    pdf.cell(0, 8, "Developed by Dr. Awujoola Olalekan", ln=True)
    output = BytesIO()
    pdf.output(output)
    return output.getvalue()

# --------------------------------------------------------------
# Main Logic
# --------------------------------------------------------------
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="latin1")
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("📋 Uploaded Data Preview")
        st.dataframe(df.head())

        # Compute HI
        results = evaluate_dataframe(df)
        st.subheader("🔍 Computed Health Index and Predicted Status")
        st.dataframe(results.head())

        # Metrics
        total = len(results)
        evaluated = int(results["Health_Index"].notna().sum())
        mean_hi = results["Health_Index"].mean()
        std_hi = results["Health_Index"].std()
        skewness = results["Health_Index"].skew()
        kurtosis = results["Health_Index"].kurtosis()

        metrics = pd.DataFrame({
            "Metric": ["Total Readings", "Evaluated Readings", "Coverage (%)", "Mean HI", "Std Dev HI", "Skewness", "Kurtosis"],
            "Value": [total, evaluated, (evaluated / total * 100 if total > 0 else 0), mean_hi, std_hi, skewness, kurtosis]
        })
        st.subheader("📊 Model Performance Metrics")
        st.dataframe(metrics.style.format(precision=3))

        # Visualization
        st.subheader("📈 Health Index Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(results["Health_Index"].dropna(), bins=10, kde=True, color="teal", ax=ax)
        ax.set_xlabel("Health Index (HI)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        st.subheader("🧭 Predicted Status Breakdown")
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        results["Predicted_Status"].value_counts().plot.pie(
            autopct="%1.1f%%", startangle=90, colors=["#4CAF50", "#FFC107", "#F44336", "#9E9E9E"], ax=ax2
        )
        ax2.set_ylabel("")
        st.pyplot(fig2)

        # Interpretation
        interpretation = interpret_results(results)
        st.subheader("🧠 Model Interpretation & Suggested Actions")
        st.markdown(interpretation)

        # PDF Report Generator
        st.subheader("📄 Generate PDF Report")
        if st.button("🖨️ Download Model Report (PDF)"):
            pdf_bytes = generate_pdf_report(metrics, interpretation)
            st.download_button(
                label="⬇️ Click to Download Report",
                data=pdf_bytes,
                file_name="Health_Monitoring_Report.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("Please upload a `.csv` or `.xlsx` file to start analysis.")
