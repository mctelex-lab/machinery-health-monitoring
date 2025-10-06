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
from datetime import datetime, timedelta
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
    "Maintenance Log History",
    "Example Dataset Generator",
    "About"
])

# --------------------------------------------------------------
# 🧭 Maintenance Suggestion Function
# --------------------------------------------------------------
def maintenance_suggestions(results):
    """Generate maintenance recommendations based on system status."""
    suggestions = []
    if "Status" not in results.columns:
        return ["No status data available for recommendations."]
    
    total = len(results)
    healthy = (results["Status"] == "Healthy").sum()
    warning = (results["Status"] == "Warning").sum()
    critical = (results["Status"] == "Critical").sum()

    if critical > 0:
        suggestions.append("⚠️ Immediate inspection required for systems flagged as Critical. Check oil, temperature, and vibration.")
    if warning > 0:
        suggestions.append("🧰 Schedule maintenance for Warning systems within the next operational cycle.")
    if healthy / total > 0.8:
        suggestions.append("✅ Majority of systems are healthy. Maintain regular inspection routine.")
    if results["Health_Index"].mean() < 0.6:
        suggestions.append("🔧 Overall Health Index below standard. Inspect lubrication and cooling systems.")
    if not suggestions:
        suggestions.append("🟢 All systems are within optimal condition. Continue current maintenance routine.")

    return suggestions

# --------------------------------------------------------------
# 📄 Enhanced PDF Report Generator with Maintenance Schedule
# --------------------------------------------------------------
def generate_pdf_report(metrics, charts, suggestions, author="Navy Capt Daya Abdullahi and Dr. Awujoola Olalekan J"):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "⚓ NAVAL MACHINERY HEALTH CONDITION REPORT", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8,
        "This report presents the predictive health evaluation of naval ship machinery "
        "based on uploaded operational parameters. The computed Health Index (HI) and "
        "equipment status distribution provide actionable insights into system reliability "
        "and readiness for continued operations.\n"
    )
    pdf.ln(5)
    pdf.cell(0, 10, f"Prepared by: {author}", ln=True, align="L")
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%B %d, %Y')}", ln=True, align="L")
    pdf.ln(8)
    pdf.set_font("Arial", "I", 11)
    pdf.multi_cell(0, 8, "Confidential - For Naval Technical Command Internal Use Only.")
    pdf.add_page()

    # Model summary
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "📊 Model Performance Summary", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8,
        "The predictive model evaluates equipment health using operational indicators. "
        "The derived Health Index (HI) indicates machinery condition categorized as "
        "Healthy, Warning, or Critical.\n"
    )

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Computed Metrics:", ln=True)
    pdf.set_font("Arial", "", 11)
    for k, v in metrics.items():
        pdf.cell(0, 8, f"• {k}: {v}", ln=True)

    # Maintenance schedule logic
    mean_hi = metrics.get("Mean HI", 0)
    if mean_hi >= 0.8:
        next_maint = datetime.now() + timedelta(days=60)
        interval = "60 days (Excellent condition)"
    elif mean_hi >= 0.6:
        next_maint = datetime.now() + timedelta(days=30)
        interval = "30 days (Satisfactory condition)"
    else:
        next_maint = datetime.now() + timedelta(days=7)
        interval = "7 days (Immediate attention required)"

    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Maintenance Schedule Recommendation:", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Next Inspection Date: {next_maint.strftime('%B %d, %Y')}", ln=True)
    pdf.cell(0, 8, f"Suggested Interval: {interval}", ln=True)
    pdf.ln(10)

    interpret = (
        "Excellent operational condition — continue routine checks." if mean_hi > 0.8 else
        "Moderate condition — monitor specific indicators regularly." if mean_hi > 0.6 else
        "Critical condition detected — initiate maintenance immediately."
    )
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Interpretation:", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, f"Based on the overall Health Index, the fleet exhibits: {interpret}")
    pdf.ln(10)

    # Recommendations
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "🧭 Maintenance Recommendations", ln=True)
    pdf.set_font("Arial", "", 11)
    for s in suggestions:
        pdf.multi_cell(0, 8, "- " + s)

    # Charts
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "📈 Charts and Visualizations", ln=True)
    pdf.ln(5)
    for name, chart_data in charts.items():
        pdf.set_font("Arial", "I", 11)
        pdf.cell(0, 10, name, ln=True)
        pdf.image(chart_data, x=15, w=180)
        pdf.ln(10)

    pdf.set_y(-25)
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
        "SYSTEM": ["Propulsion", "Cooling", "Electrical"],
        "EQUIPMENT": ["Main Engine", "Cooling Pump", "Generator"],
        "HEALTH_INDICATOR_HI": ["Oil pressure", "Water temp", "Voltage"],
        "SYNTHETIC_VALUE": [4.2, 72, 440],
        "MIN_MAX_THRESHOLDS": ["3.5-6.0", "60-85", "420-450"],
        "REMARKS": ["Normal", "Normal", "Normal"],
        "Actual_Status": ["Healthy", "Healthy", "Healthy"]
    }
    return pd.DataFrame(data)

# --------------------------------------------------------------
# 🧾 Maintenance Log Management
# --------------------------------------------------------------
LOG_PATH = "data/maintenance_log.csv"

def update_maintenance_log(mean_hi, next_inspection, interval):
    os.makedirs("data", exist_ok=True)
    entry = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Mean_Health_Index": round(mean_hi, 3),
        "Next_Inspection": next_inspection.strftime("%Y-%m-%d"),
        "Interval": interval
    }

    if os.path.exists(LOG_PATH):
        df_log = pd.read_csv(LOG_PATH)
    else:
        df_log = pd.DataFrame(columns=entry.keys())

    df_log = pd.concat([df_log, pd.DataFrame([entry])], ignore_index=True)
    df_log.to_csv(LOG_PATH, index=False)
    return df_log

# --------------------------------------------------------------
# 📊 Main App Logic
# --------------------------------------------------------------
if page == "Upload & Analyze Data":
    st.title("⚓ Equipment Health Condition Monitoring Prediction for Naval Ships")
    st.caption("Developed by Dr. Awujoola Olalekan — Naval Technical Command")

    uploaded_file = st.file_uploader("📁 Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            raw_data = uploaded_file.read()
            result = chardet.detect(raw_data)
            encoding = result["encoding"] or "utf-8"
            uploaded_file.seek(0)

            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file, encoding=encoding, on_bad_lines='skip')
            else:
                df = pd.read_excel(uploaded_file, engine="openpyxl")

            st.success("✅ File loaded successfully!")
            st.dataframe(df.head())

            results = evaluate_dataframe(df)
            if "Status" not in results.columns and "Health_Index" in results.columns:
                results["Status"] = pd.cut(results["Health_Index"],
                                           bins=[-np.inf, 0.5, 0.8, np.inf],
                                           labels=["Critical", "Warning", "Healthy"])

            st.subheader("🔍 Computed Health Index and Status")
            st.dataframe(results.head(10))

            # Maintenance suggestions
            suggestions = maintenance_suggestions(results)

            # Visuals
            st.subheader("📊 Health Index Dashboard")
            fig1, ax1 = plt.subplots()
            sns.histplot(results["Health_Index"], kde=True, bins=10, ax=ax1)
            ax1.set_title("Health Index Distribution")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            results["Status"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax2)
            ax2.set_ylabel("")
            ax2.set_title("Status Breakdown")
            st.pyplot(fig2)

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

            # Maintenance schedule in app
            mean_hi = metrics["Mean HI"]
            if mean_hi >= 0.8:
                st.success("✅ Excellent performance. Next inspection in 60 days.")
                next_date = datetime.now() + timedelta(days=60)
                interval = "60 days"
            elif mean_hi >= 0.6:
                st.warning("⚠️ Moderate performance. Schedule inspection in 30 days.")
                next_date = datetime.now() + timedelta(days=30)
                interval = "30 days"
            else:
                st.error("🚨 Poor performance. Maintenance required within 7 days.")
                next_date = datetime.now() + timedelta(days=7)
                interval = "7 days"

            # Update and show maintenance log
            df_log = update_maintenance_log(mean_hi, next_date, interval)
            st.subheader("🧾 Maintenance Log History (auto-updated)")
            st.dataframe(df_log)
            csv_log = df_log.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Maintenance Log (CSV)", csv_log, "maintenance_log.csv", "text/csv")

            # PDF Report
            chart_buffers = {"Health Index Distribution": BytesIO(), "Status Breakdown": BytesIO()}
            fig1.savefig(chart_buffers["Health Index Distribution"], format="png")
            fig2.savefig(chart_buffers["Status Breakdown"], format="png")
            pdf_buffer = generate_pdf_report(metrics, chart_buffers, suggestions)

            st.download_button("📘 Download Full Health Report (PDF)", pdf_buffer,
                               file_name="Naval_Machinery_Health_Report.pdf", mime="application/pdf")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

elif page == "Maintenance Log History":
    st.title("🧾 Maintenance Log History")
    if os.path.exists(LOG_PATH):
        df_log = pd.read_csv(LOG_PATH)
        st.dataframe(df_log)
        st.download_button("📥 Download Maintenance Log", df_log.to_csv(index=False), "maintenance_log.csv", "text/csv")
    else:
        st.warning("No maintenance log found yet. Upload and analyze a dataset first.")

elif page == "Example Dataset Generator":
    st.title("🧪 Example Dataset Generator")
    df_sample = generate_example_dataset()
    st.dataframe(df_sample)
    st.download_button("📥 Download Example Dataset", df_sample.to_csv(index=False), "example_dataset.csv", "text/csv")

else:
    st.title("ℹ️ About This Application")
    st.markdown("""
    This predictive maintenance platform monitors naval ship machinery using a computed **Health Index (HI)**.  
    It provides real-time classification into **Healthy**, **Warning**, and **Critical** states, along with 
    actionable **maintenance recommendations** and a **scheduling plan** for inspections.

    **Developed by Dr. Awujoola Olalekan (Naval Technical Command)**  
    """)
