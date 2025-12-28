import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction - EDA & Model",
    layout="wide"
)

st.title("🏠 House Price Prediction Dashboard")
st.markdown("Exploratory Data Analysis (EDA) + Linear Regression Model")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("./house_price_regression_dataset.csv")

df = load_data()

# -----------------------------
# Sidebar Navigation
# -----------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Dataset Overview", "EDA", "Model Training & Evaluation", "Price Prediction"]
)

# -----------------------------
# Dataset Overview
# -----------------------------
if menu == "Dataset Overview":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

# -----------------------------
# EDA Section
# -----------------------------
elif menu == "EDA":
    st.subheader("Exploratory Data Analysis")

    # Correlation Heatmap
    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Distribution of Target Variable
    st.markdown("### House Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["House_Price"], kde=True, ax=ax)
    st.pyplot(fig)

    # Feature vs Target
    st.markdown("### Feature vs House Price")
    feature = st.selectbox(
        "Select Feature",
        df.columns.drop("House_Price")
    )

    fig, ax = plt.subplots()
    sns.scatterplot(x=df[feature], y=df["House_Price"], ax=ax)
    ax.set_xlabel(feature)
    ax.set_ylabel("House Price")
    st.pyplot(fig)

# -----------------------------
# Model Training & Evaluation
# -----------------------------
elif menu == "Model Training & Evaluation":
    st.subheader("Linear Regression Model")

    X = df.drop("House_Price", axis=1)
    y = df["House_Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    # Metrics
    st.markdown("### Model Evaluation Metrics")
    col1, col2, col3 = st.columns(3)

    col1.metric("R² Score", round(r2_score(y_test, y_pred), 4))
    col2.metric("RMSE", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
    col3.metric("MAE", round(mean_absolute_error(y_test, y_pred), 2))

    # Coefficients
    st.markdown("### Feature Coefficients")
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=False)

    st.dataframe(coef_df)

# -----------------------------
# Prediction Section
# -----------------------------
elif menu == "Price Prediction":
    st.subheader("Predict House Price")

    # Train model (again for prediction)
    X = df.drop("House_Price", axis=1)
    y = df["House_Price"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    # User Inputs
    st.markdown("### Enter House Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        sqft = st.number_input("Square Footage", min_value=200)
        bedrooms = st.number_input("Bedrooms", min_value=1, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=1, step=1)

    with col2:
        year_built = st.number_input("Year Built", min_value=1800, max_value=2025)
        lot_size = st.number_input("Lot Size")
        garage_size = st.number_input("Garage Size", min_value=0, step=1)

    with col3:
        neighborhood_quality = st.slider("Neighborhood Quality", 1, 10)

    if st.button("Predict Price"):
        input_data = np.array([[
            sqft,
            bedrooms,
            bathrooms,
            year_built,
            lot_size,
            garage_size,
            neighborhood_quality
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        st.success(f"💰 Estimated House Price: ₹ {prediction[0]:,.2f}")
