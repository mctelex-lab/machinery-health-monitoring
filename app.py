# ==============================================================
# ⚓ Equipment Health Condition Monitoring Prediction for Naval Ships
# Developed by Dr. Awujoola Olalekan
# Streamlit Web Application
# ==============================================================
# Upload CSV/Excel → Compute Health Index (HI)
# Visualize results → Download outputs → Generate example dataset
# ==============================================================

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from evaluate_hi import evaluate_dataframe

# --------------------------------------------------------------
# Streamlit Page Configuration
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
st.subheader("👨‍💻 Developed by Dr. Awujoola Olalekan")
st.caption("""
A smart predictive maintenance tool for analyzing machinery health conditions onboard naval vessels.  
Upload sensor or manual input data, predict equipment health status, and visualize real-time operational insights.
""")

# --------------------------------------------------------------
# Sidebar Section
# --------------------------------------------------------------
with st.sidebar:
    st.header("📘 Resources & Actions")
    st.markdown("[🌐 View Source on GitHub](https://github.com/mctelex-lab/manual-health-monitoring)")

    try:
        with open("reports/Predictive_Maintenance_Model_Performance_Report.pdf", "rb") as f:
            st.download_button(
                label="📄 Download Performance Report (PDF)",
                data=f.read(),
                file_name="Predictive_Maintenance_Model_Performance_Report.pdf",
                mime="application/pdf"
            )
    except FileNotFoundError:
        st.info("Performance report not found. Upload data to generate one.")

    st.markdown("---")
    st.header("🧩 App Features")
    st.markdown("""
- 📊 Automatic Health Index computation  
- ⚙️ Multi-system and indicator analysis  
- 🧭 Predicted condition status (Healthy/Warning/Critical)  
- 📈 Insightful visualizations  
- ⬇️ Exportable reports and metrics  
""")

    st.markdown("---")
    st.header("🆘 About")
    st.info("Developed by Dr. Awujoola Olalekan for predictive maintenance in naval engineering systems using Python, Pandas, and Streamlit.")

# --------------------------------------------------------------
# Example Dataset Generator
# --------------------------------------------------------------
st.subheader("📁 Example Dataset Generator")
st.markdown("Click below to download a ready-to-use dataset for testing if you don’t have your own data yet.")

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
# Data Processing and Visualization
# --------------------------------------------------------------
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="latin1")
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("📋 Uploaded Data Preview")
        st.dataframe(df.head())

        # Compute Health Index
        results = evaluate_dataframe(df)

        st.subheader("🔍 Computed Health Index and Predicted Status")
        st.dataframe(results.head())

        # ----------------------------------------------------------
        # Performance Metrics
        # ----------------------------------------------------------
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

        # ----------------------------------------------------------
        # Visualization 1: Health Index Distribution
        # ----------------------------------------------------------
        st.subheader("📈 Health Index Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(results["Health_Index"].dropna(), bins=10, kde=True, ax=ax, color="teal")
        ax.set_xlabel("Health Index (HI)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Health Index Across Equipment")
        st.pyplot(fig)

        # ----------------------------------------------------------
        # Visualization 2: Equipment Status Breakdown
        # ----------------------------------------------------------
        st.subheader("🧭 Predicted Status Breakdown")
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        results["Predicted_Status"].value_counts().plot.pie(
            autopct="%1.1f%%", startangle=90,
            colors=["#4CAF50", "#FFC107", "#F44336", "#9E9E9E"], ax=ax2
        )
        ax2.set_ylabel("")
        ax2.set_title("Predicted Equipment Condition Distribution")
        st.pyplot(fig2)

        # ----------------------------------------------------------
        # Visualization 3: Working Value vs Health Index
        # ----------------------------------------------------------
        if "WORKING_VALUE_ONBOARD" in results.columns:
            st.subheader("⚙️ Working Value vs Health Index")
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            sns.scatterplot(
                x=pd.to_numeric(results["WORKING_VALUE_ONBOARD"].str.extract(r"(\d+\.*\d*)")[0], errors="coerce"),
                y=results["Health_Index"],
                hue=results["Predicted_Status"],
                palette="viridis",
                s=70,
                ax=ax3
            )
            ax3.set_xlabel("Working Value (numeric)")
            ax3.set_ylabel("Health Index (HI)")
            ax3.set_title("Relationship Between Working Value and Health Index")
            st.pyplot(fig3)

        # ----------------------------------------------------------
        # Visualization 4: Average Health Index per System
        # ----------------------------------------------------------
        if "SYSTEM" in results.columns:
            st.subheader("🏗️ Average Health Index per System")
            avg_hi = results.groupby("SYSTEM")["Health_Index"].mean().reset_index()
            fig4, ax4 = plt.subplots(figsize=(10, 5))
            sns.barplot(data=avg_hi, x="SYSTEM", y="Health_Index", ax=ax4, palette="coolwarm")
            ax4.set_title("Average Health Index by System")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig4)

        # ----------------------------------------------------------
        # Visualization 5: Health Index by Indicator
        # ----------------------------------------------------------
        if "HEALTH_INDICATOR_HI" in results.columns:
            st.subheader("🔧 Health Index by Indicator")
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=results,
                y="HEALTH_INDICATOR_HI",
                x="Health_Index",
                orient="h",
                palette="mako"
            )
            ax5.set_title("Health Index by Indicator")
            plt.tight_layout()
            st.pyplot(fig5)

        # ----------------------------------------------------------
        # Download Results
        # ----------------------------------------------------------
        st.subheader("⬇️ Download Evaluated Results")
        output = BytesIO()
        results.to_csv(output, index=False)
        st.download_button(
            label="💾 Download Results as CSV",
            data=output.getvalue(),
            file_name="health_index_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error while processing: {e}")

else:
    st.info("Please upload a `.csv` or `.xlsx` file to start your analysis.")
