# Machinery Health Monitoring (Streamlit App)

This repository contains a Streamlit web application that computes a **Health Index (HI)** and predicted equipment status from CSV/Excel inputs.

## Features
- Upload CSV/Excel readings
- Compute Health Index (HI) and Predicted Status (Healthy/Warning/Critical)
- Visualize results (histogram, pie chart)
- Download evaluated results and performance PDF

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Push this repository to GitHub under `mctelex-lab/manual-health-monitoring`
2. Go to https://streamlit.io/cloud and sign in with GitHub
3. Create a new app and point to `app.py` in the `main` branch
4. Deploy

## Repository
https://github.com/mctelex-lab/manual-health-monitoring
