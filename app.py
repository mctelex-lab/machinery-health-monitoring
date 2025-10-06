# ==============================================================
# ⚙️ Machinery Health Monitoring System
# Streamlit Web Application
# ==============================================================
# Upload CSV/Excel → Compute Health Index (HI)
# Visualize metrics → Download results and performance report
# ==============================================================

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))  # allow local src imports

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from evaluate_hi import evaluate_dataframe

# Streamlit App Config
st.set_page_config(
    page_title="Machinery Health Monitoring System",
    layout="wide",
    page_icon="⚙️"
)

# Title
st.title("⚙️ Machinery Health Monitoring System")
st.write("Upload your machinery health readings (CSV or Excel) to compute the **Health Index (HI)** and evaluate overall condition.")

# Sidebar
with st.sidebar:
    st.header("📘 Resources")
    st.markdown("[Visit on GitHub](https://github.com/mctelex-lab/manual-health-monitoring)")
    try:
        with open("reports/Predictive_Maintenance_Model_Performance_Report.pdf", "rb") as f:
            st.download_button(
                label="📄 Download Performance Report (PDF)",
                data=f.read(),
                file_name="Predictive_Maintenance_Model_Performance_Report.pdf",
                mime="application/pdf"
            )
    except FileNotFoundError:
        st.info("Performance report not found. Upload CSV below to begin analysis.")
    st.markdown("---")
    st.caption("Developed by **mctelex-lab** for predictive maintenance applications.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your data file", type=["csv", "xlsx"])

# Process file if uploaded
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="latin1")
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("📋 Uploaded Data Preview")
        st.dataframe(df.head())

        # Compute Health Index and Status
        results = evaluate_dataframe(df)

        st.subheader("🔍 Computed Health Index and Predicted Status")
        st.dataframe(results.head())

        # Compute summary metrics
        total = len(results)
        evaluated = int(results["Health_Index"].notna().sum()) if "Health_Index" in results.columns else 0
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

        # Visualizations
        st.subheader("📈 Health Index Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(results["Health_Index"].dropna(), bins=10, kde=True, ax=ax, color="skyblue")
        ax.set_xlabel("Health Index (HI)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        st.subheader("🧭 Equipment Status Breakdown")
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        results["Predicted_Status"].value_counts().plot.pie(
            autopct="%1.1f%%",
            startangle=90,
            colors=["#4CAF50", "#FFC107", "#F44336", "#9E9E9E"],
            ax=ax2
        )
        ax2.set_ylabel("")
        ax2.set_title("Status Distribution")
        st.pyplot(fig2)

        # Download analyzed results
        st.subheader("⬇️ Download Evaluated Results")
        output = BytesIO()
        results.to_csv(output, index=False)
        st.download_button(
            label="Download Results as CSV",
            data=output.getvalue(),
            file_name="health_index_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error while processing: {e}")

else:
    st.info("Please upload a `.csv` or `.xlsx` file to start your analysis.")
