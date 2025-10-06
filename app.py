import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from src.evaluate_hi import evaluate_dataframe

st.set_page_config(
    page_title="Machinery Health Monitoring System",
    layout="wide",
    page_icon="⚙️"
)

st.title("⚙️ Machinery Health Monitoring System")
st.write("Upload your equipment health readings (CSV or Excel) to evaluate **Health Index (HI)** and **Predicted Status**.")

with st.sidebar:
    st.header("📘 Resources")
    st.markdown("Repository: [mctelex-lab/manual-health-monitoring](https://github.com/mctelex-lab/manual-health-monitoring)")
    try:
        with open("reports/Predictive_Maintenance_Model_Performance_Report.pdf", "rb") as f:
            pdf_bytes = f.read()
            st.download_button(
                label="Download Performance Report (PDF)",
                data=pdf_bytes,
                file_name="Predictive_Maintenance_Model_Performance_Report.pdf",
                mime="application/pdf"
            )
    except Exception:
        st.info("Performance report not available.")

uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="latin1")
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("📋 Uploaded Data Preview")
        st.dataframe(df.head())

        # Evaluate health condition
        results = evaluate_dataframe(df)

        st.subheader("🔍 Predicted Health Status (sample)")
        st.dataframe(results.head())

        # Calculate basic metrics
        total = len(results)
        evaluated = int(results["Health_Index"].notna().sum()) if "Health_Index" in results.columns else 0
        mean_hi = results["Health_Index"].mean() if "Health_Index" in results.columns else None
        std_hi = results["Health_Index"].std() if "Health_Index" in results.columns else None
        skewness = results["Health_Index"].skew() if "Health_Index" in results.columns else None
        kurtosis = results["Health_Index"].kurtosis() if "Health_Index" in results.columns else None

        metrics = pd.DataFrame({
            "Metric": ["Total Readings", "Evaluated Readings", "Coverage (%)", "Mean HI", "Std Dev HI", "Skewness", "Kurtosis"],
            "Value": [total, evaluated, (evaluated/total)*100 if total>0 else 0, mean_hi, std_hi, skewness, kurtosis]
        })

        st.subheader("📊 Performance Metrics")
        st.dataframe(metrics.style.format(precision=3))

        # Visualization
        st.subheader("📈 Health Index Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        if "Health_Index" in results.columns:
            sns.histplot(results["Health_Index"].dropna(), bins=10, kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.info("No Health_Index column computed. Check your file format.")

        st.subheader("🧭 Status Breakdown")
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        if "Predicted_Status" in results.columns:
            results["Predicted_Status"].value_counts().plot.pie(
                autopct="%1.1f%%", startangle=90, colors=["#4CAF50", "#FFC107", "#F44336", "#9E9E9E"], ax=ax2
            )
            ax2.set_ylabel("")
            st.pyplot(fig2)
        else:
            st.info("No Predicted_Status column computed.")

        # Download results
        st.subheader("⬇️ Download Evaluated Data")
        output = BytesIO()
        results.to_csv(output, index=False)
        st.download_button(
            label="Download Results as CSV",
            data=output.getvalue(),
            file_name="health_index_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("Please upload a CSV or Excel file to continue.")
