# ==============================================================
# ⚓ Equipment Health Condition Monitoring Prediction for Naval Ships
# ==============================================================
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))  # ✅ Allow local src imports

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chardet
from io import BytesIO
from fpdf import FPDF
from evaluate_hi import evaluate_dataframe  # local evaluation module

# --------------------------------------------------------------
# 🎨 Streamlit Page Setup
# --------------------------------------------------------------
st.set_page_config(
    page_title="Equipment Health Condition Monitoring Prediction for Naval Ships",
    page_icon="⚓",
    layout="wide"
)

# --------------------------------------------------------------
# 🧭 Sidebar Navigation
# --------------------------------------------------------------
st.sidebar.title("⚙️ Navigation")
page = st.sidebar.radio("Select a section", [
    "Upload & Analyze Data",
    "Example Dataset Generator",
    "About"
])

# --------------------------------------------------------------
# 🧾 Maintenance Suggestion Function
# --------------------------------------------------------------
def maintenance_suggestions(results):
    """Generate maintenance recommendations based on status patterns."""
    suggestions = []
    if "Status" not in results.columns:
        return ["No status data available for suggestions."]
    
    total = len(results)
    healthy = (results["Status"] == "Healthy").sum()
    warning = (results["Status"] == "Warning").sum()
    critical = (results["Status"] == "Critical").sum()

    if critical > 0:
        suggestions.append("⚠️ Immediate inspection is required for systems flagged as Critical. Verify oil pressure, temperature, and vibration levels.")
    if warning > 0:
        suggestions.append("🧰 Schedule maintenance for equipment in the Warning state within the next operational cycle.")
    if healthy / total > 0.8:
        suggestions.append("✅ Most systems are healthy. Continue routine checks and oil level verification.")
    if results["Health_Index"].mean() < 0.6:
        suggestions.append("🔧 Overall Health Index is below standard threshold. Review maintenance logs and inspect cooling and lubrication systems.")
    if not suggestions:
        suggestions.append("🟢 All systems are within optimal performance range. Maintain current operating protocols.")

    return suggestions

# --------------------------------------------------------------
# 🧾 PDF Report Generator
# --------------------------------------------------------------
def generate_pdf_report(metrics, charts, suggestions, author="Navy Capt Daya Abdullahi and Dr. Awujoola Olalekan J"):
    pdf = FPDF()
    pdf.add_page()

    # Cover Page
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "⚓ NAVAL MACHINERY HEALTH CONDITION REPORT", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"Predictive Maintenance Model Summary\n\nAuthor: {author}")
    pdf.ln(10)
    pdf.cell(0, 10, "Confidential - For Naval Technical Use Only", ln=True, align="C")

    # Detailed Report
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "📊 Model Performance Metrics", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    for k, v in metrics.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "🧭 Maintenance Recommendations", ln=True)
    pdf.set_font("Arial", "", 11)
    for s in suggestions:
        pdf.multi_cell(0, 8, "- " + s)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "📈 Charts and Visualizations", ln=True)
    for name, chart_data in charts.items():
        pdf.ln(5)
        pdf.set_font("Arial", "I", 11)
        pdf.cell(0, 10, name, ln=True)
        pdf.image(chart_data, x=15, w=180)
        pdf.ln(10)

    pdf.set_y(-30)
    pdf.set_font("Arial", "I", 9)
    pdf.cell(0, 10, "Generated via Streamlit | Naval Technical Command | Confidential", align="C")

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# --------------------------------------------------------------
# 🧰 Example Dataset Generator
# --------------------------------------------------------------
def generate_example_dataset():
    data = {
        "SYSTEM": ["Propulsion", "Cooling", "Electrical", "Hydraulic", "Air System"],
        "EQUIPMENT": ["Main Engine", "Cooling Pump", "Generator", "Hydraulic Pump", "Air Compressor"],
        "HEALTH_INDICATOR_HI": ["Lube oil pressure (bar)", "Water temp (°C)", "Voltage (V)", "Pressure (bar)", "Air Pressure (bar)"],
        "SYNTHETIC_VALUE": [4.2, 72, 440, 150, 28],
        "MIN_MAX_THRESHOLDS": ["3.5-6.0", "60-85", "420-450", "135-160", "24-30"],
        "WORKING_VALUE_ONBOARD": ["3.6 bar", "70 °C", "435 V", "145 bar", "29 bar"],
        "REMARKS": ["OK", "Normal", "Normal", "Normal", "Normal"],
        "Actual_Status": ["Healthy", "Healthy", "Healthy", "Healthy", "Healthy"]
    }
    return pd.DataFrame(data)

# --------------------------------------------------------------
# 📊 Main App Logic
# --------------------------------------------------------------
if page == "Upload & Analyze Data":
    st.title("⚓ Equipment Health Condition Monitoring Prediction for Naval Ships")
    st.caption("Developed by Dr. Awujoola Olalekan — Naval Technical Command")

    uploaded_file = st.file_uploader("📁 Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            # Detect encoding
            raw_data = uploaded_file.read()
            result = chardet.detect(raw_data)
            detected_encoding = result["encoding"] or "utf-8"
            st.info(f"Detected file encoding: {detected_encoding}")
            uploaded_file.seek(0)

            # Load data safely
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file, encoding=detected_encoding, on_bad_lines='skip')
            else:
                df = pd.read_excel(uploaded_file, engine="openpyxl")

            # Clean column names
            df.columns = (
                df.columns.astype(str)
                .str.strip()
                .str.replace('\u2013', '-', regex=False)
                .str.replace('\xa0', '', regex=False)
                .str.replace(' ', '_')
                .str.replace('-', '_')
                .str.replace(r'[^0-9a-zA-Z_]', '', regex=True)
            )

            st.success("✅ File loaded and sanitized successfully!")
            st.dataframe(df.head())

            # Evaluate dataset
            results = evaluate_dataframe(df)
            st.subheader("🔍 Computed Health Index and Predicted Status")
            st.dataframe(results.head(10))

            # Maintenance suggestions
            st.subheader("🧭 Maintenance Recommendations")
            suggestions = maintenance_suggestions(results)
            for s in suggestions:
                st.markdown(f"- {s}")

            # Visualizations
            st.subheader("📊 Health Insights Dashboard")

            fig1, ax1 = plt.subplots()
            sns.histplot(results["Health_Index"], kde=True, bins=10, ax=ax1)
            ax1.set_title("Health Index Distribution")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            results["Status"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax2)
            ax2.set_ylabel("")
            ax2.set_title("Equipment Status Breakdown")
            st.pyplot(fig2)

            if "SYNTHETIC_VALUE" in results.columns:
                fig3, ax3 = plt.subplots()
                sns.scatterplot(x="SYNTHETIC_VALUE", y="Health_Index", data=results, ax=ax3)
                ax3.set_title("Working Value vs Health Index")
                st.pyplot(fig3)

            # Metrics Summary
            st.subheader("📈 Model Performance Metrics")
            metrics = {
                "Total Readings": len(df),
                "Evaluated Readings": len(results),
                "Coverage (%)": round((len(results) / len(df)) * 100, 2),
                "Mean HI": round(results["Health_Index"].mean(), 3),
                "Std Dev HI": round(results["Health_Index"].std(), 3),
                "Skewness": round(results["Health_Index"].skew(), 3),
                "Kurtosis": round(results["Health_Index"].kurtosis(), 3)
            }
            st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

            # Save charts for report
            chart_buffers = {
                "Health Index Distribution": BytesIO(),
                "Equipment Status Breakdown": BytesIO()
            }
            fig1.savefig(chart_buffers["Health Index Distribution"], format="png")
            fig2.savefig(chart_buffers["Equipment Status Breakdown"], format="png")

            # Generate PDF
            pdf_buffer = generate_pdf_report(metrics, chart_buffers, suggestions)
            st.download_button(
                label="📥 Download Model Performance Report (PDF)",
                data=pdf_buffer,
                file_name="Naval_Machinery_Health_Report.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# --------------------------------------------------------------
# 🧪 Example Dataset
# --------------------------------------------------------------
elif page == "Example Dataset Generator":
    st.title("🧪 Example Dataset Generator")
    st.caption("Use this to test the model if you don't have a dataset.")
    df_sample = generate_example_dataset()
    st.dataframe(df_sample)
    csv = df_sample.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Example Dataset", csv, "example_dataset.csv", "text/csv")

# --------------------------------------------------------------
# ℹ️ About Section
# --------------------------------------------------------------
else:
    st.title("ℹ️ About This Application")
    st.markdown("""
    This intelligent monitoring platform analyzes and predicts the operational health of naval ship machinery.  
    It computes a **Health Index (HI)** from operational indicators and categorizes equipment into:
    - 🟢 **Healthy** — Normal operational condition  
    - 🟡 **Warning** — Requires attention  
    - 🔴 **Critical** — Immediate maintenance required  

    The app also generates **actionable maintenance recommendations** and allows users to **download a PDF report** 
    summarizing performance and insights.  

    **Developed by Dr. Awujoola Olalekan (Naval Technical Command)**  
    """)

