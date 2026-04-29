import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Sales Forecast MLOps", layout="wide")

st.title("📊 Sales Forecasting MLOps Dashboard")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df

df = load_data()

# -----------------------------
# Sidebar - Date Selection
# -----------------------------
st.sidebar.header("🔮 Forecast Settings")

selected_date = st.sidebar.date_input("Select Forecast Date")
selected_date = pd.to_datetime(selected_date)

# -----------------------------
# Show dataset overview
# -----------------------------
st.subheader("📈 Sales Trend Over Time")

fig, ax = plt.subplots()
ax.plot(df["date"], df["sales"], label="Sales")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()

st.pyplot(fig)

# -----------------------------
# Metrics
# -----------------------------
st.subheader("📊 Dataset Insights")

col1, col2, col3 = st.columns(3)

col1.metric("Average Sales", round(df["sales"].mean(), 2))
col2.metric("Max Sales", round(df["sales"].max(), 2))
col3.metric("Min Sales", round(df["sales"].min(), 2))

# -----------------------------
# FEATURE ENGINEERING (FIXED LOGIC)
# -----------------------------

# Filter only past data (important fix)
filtered_df = df[df["date"] <= selected_date]

# Validation check
if len(filtered_df) < 8:
    st.error("❌ Not enough historical data for prediction (need at least 7-8 days before selected date)")
    st.stop()

sales_series = filtered_df["sales"]

# Dynamic feature generation
lag_1 = float(sales_series.iloc[-1])
lag_7 = float(sales_series.iloc[-7])
rolling_mean_7 = float(sales_series.tail(7).mean())

# Show features
st.sidebar.subheader("📌 Auto Generated Features")
st.sidebar.json({
    "lag_1": lag_1,
    "lag_7": lag_7,
    "rolling_mean_7": rolling_mean_7
})

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("🔮 Forecast Sales")

if st.button("Predict Sales"):

    payload = {
        "date": str(selected_date.date()),
        "features": {
            "lag_1": lag_1,
            "lag_7": lag_7,
            "rolling_mean_7": rolling_mean_7
        }
    }

    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=payload
        )

        if response.status_code == 200:
            prediction = response.json()["predicted_sales"]

            st.success(f"📈 Predicted Sales: {prediction}")

            # Optional visualization of prediction
            st.subheader("📊 Prediction Summary")

            chart_data = pd.DataFrame({
                "Type": ["Lag_1", "Lag_7", "Rolling Mean", "Prediction"],
                "Value": [lag_1, lag_7, rolling_mean_7, prediction]
            })

            st.bar_chart(chart_data.set_index("Type"))

        else:
            st.error("❌ Prediction API failed")

    except Exception as e:
        st.error(f"❌ Connection Error: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("🚀 End-to-End MLOps Sales Forecasting System")