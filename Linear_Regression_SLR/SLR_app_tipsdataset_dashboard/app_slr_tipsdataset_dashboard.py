import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Linear Regression Dashboard",
    layout="wide"
)

# ------------------ Load CSS ------------------
def load_css(filename):
    css_path = Path(__file__).parent / filename
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

# ------------------ Header ------------------
st.markdown("""
<div class="card header-card" style='padding:16px 24px;'>
    <h1>Simple Linear Regression Dashboard</h1>
    <p>Predict <b>Tip Amount</b> from <b>Total Bill</b></p>
</div>
""", unsafe_allow_html=True)

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# ------------------ Prepare Data ------------------
X = df[["total_bill"]]
y = df["tip"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------ Train Model ------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# ------------------ Metrics ------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# ------------------ Dashboard Layout ------------------
col1, col2, col3 = st.columns([1.2, 1.6, 1.2])

# -------- Dataset Preview --------
with col1:
    st.markdown("""
    <div class="card">
        <h3>Dataset Preview</h3>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

# -------- Visualization --------
with col2:
    st.markdown("""
    <div class="card">
        <h3>Total Bill vs Tip</h3>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots()
    ax.scatter(df["total_bill"], df["tip"], alpha=0.6)
    ax.plot(
        df["total_bill"],
        model.predict(scaler.transform(X)),
        color="red"
    )
    ax.set_xlabel("Total Bill")
    ax.set_ylabel("Tip")
    st.pyplot(fig, use_container_width=True)

# -------- Metrics & Prediction --------
with col3:
    st.markdown("""
    <div class="card">
        <h3>Model Performance</h3>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", round(mae, 2))
    m2.metric("MSE", round(mse, 2))
    m3.metric("RMSE", round(rmse, 2))

    st.markdown("""
    <div class="card">
        <h3>Predict Tip Amount</h3>
    </div>
    """, unsafe_allow_html=True)

    bill = st.slider(
        "Bill Amount",
        float(df["total_bill"].min()),
        float(df["total_bill"].max())
    )

    tip = model.predict(scaler.transform([[bill]]))[0]
    st.markdown(
        f"<div class='prediction-box'>Predicted Tip: ₹ {tip:.2f}</div>",
        unsafe_allow_html=True
    )
