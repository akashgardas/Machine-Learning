import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import streamlit as st
import os

# --- Page Config ---
st.set_page_config(page_title="Customer Risk System", layout="wide")

# --- 1️⃣ App Header ---
st.title("Customer Risk Prediction System")
st.markdown("#### This system predicts customer risk by comparing them with similar customers.")
st.markdown(
    ":violet-badge[:material/star: KNearestNeighbors] :orange-badge[Classification]"
)
st.divider()

# --- Dataset Cache ---
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'cleaned_credit_risk_dataset.csv')
    return pd.read_csv(csv_path)

df = load_data()

# --- 2️⃣ Sidebar – User Input ---
st.sidebar.header("Customer Information")

age = st.sidebar.slider("Age", 18, 90, 30)
income = st.sidebar.number_input("Annual Income ($)", value=50000, step=1000)
loan = st.sidebar.number_input("Loan Amount ($)", value=10000, step=500)
credit_hist = st.sidebar.radio("Good Credit History?", ("Yes", "No"))
k_value = st.sidebar.slider("K Value (Number of Neighbors)", 1, 25, 3)

# Convert categorical input to numerical
credit_binary = 1 if credit_hist == "Yes" else 0

# --- 3️⃣ Prediction Logic ---
# Prepare data
X = df[['person_age', 'person_income', 'loan_amnt', 'cb_person_default_on_file']]
y = df['loan_status']

# Scaling is mandatory for KNN distance metrics
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define and Fit Model
knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X_scaled, y)

# Prepare user input for prediction
user_input = np.array([[age, income, loan, credit_binary]])
user_input_scaled = scaler.transform(user_input)

# --- 4️⃣ Main Prediction Button & Output ---
if st.button("Predict Customer Risk"):
    prediction = knn.predict(user_input_scaled)[0]
    
    st.subheader("Prediction Result")
    if prediction == 0:
        st.error(f"🔴 Risky Customer")
    else:
        st.success(f"🟢 Not a Risky Customer")

    st.divider()

    # --- 5️⃣ Nearest Neighbors Explanation ---
    st.subheader("🔍 Neighbor Analysis")
    
    # Get distances and indices of the K nearest neighbors
    distances, indices = knn.kneighbors(user_input_scaled)
    neighbors_df = df[['person_age', 'person_income', 'loan_amnt', 'cb_person_default_on_file', 'loan_status']].iloc[indices[0]].copy()
    neighbors_df['Distance'] = distances[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Neighbors Considered (K)", k_value)
    with col2:
        # Calculate majority class manually for display
        majority = neighbors_df['loan_status'].mode()[0]
        st.metric("Majority Class", majority)

    st.write("**Top Nearest Customers found in dataset:**")
    st.table(neighbors_df)

    # --- 6️⃣ Business Insight Section ---
    st.info(
        "**Business Insight:** This decision is based on similarity with nearby customers in 'feature space'. "
        "By looking at the table above, you can see how customers with similar age, income, and credit "
        "history have behaved in the past."
    )
else:
    st.write("Adjust the parameters in the sidebar and click the button to see the risk assessment.")