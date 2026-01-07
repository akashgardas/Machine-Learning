import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import streamlit as st
import os

st.set_page_config('Telco Customer Churn Analysis - Logistic Regression', layout='centered')

# -------------
# Title
# -------------
st.title('☎️Telco Customer Churn Analysis - Logistic Regression')
st.text('Predict if a customer will stay or leave')

# --------------
# Load dataset
# --------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return pd.read_csv(csv_path)

df = load_data()
df.drop('customerID', axis=1, inplace=True)

# -----------------
# Dataset Overview
# -----------------
st.header('Dataset Overview')
st.dataframe(df.head())

st.subheader('Shape')
st.write(df.shape)

st.subheader('Data Types')
st.write(df.dtypes)

st.subheader('Stats Summary')
st.dataframe(df.describe())

# -------------------
# EDA Analysis 
# -------------------
st.header('Exploratory Data Analysis')

# Distribution of Target Variable
st.subheader('Churn Count')
fig, ax = plt.subplots()
sns.countplot(data=df, x='Churn', ax=ax)
st.pyplot(fig)

# --------------------
# Encoding
# --------------------
# label encoder
le = LabelEncoder()

# encoding all object columns
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# -----------------------------
# Model Training and Evaluation
# -----------------------------
st.header('Model Training and Evaluation')

# Feature Separation
X = df.drop(['Churn'], axis=1)
y = df.Churn

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Building
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Model Score
st.subheader('Model Score')
st.text(model.score(X_train_scaled, y_train))

# Testing Score
y_pred = model.predict(X_test_scaled)
st.subheader('Accuracy Score')
st.text(accuracy_score(y_test, y_pred))
st.subheader(f'Classification Report:')
st.dataframe(classification_report(y_test, y_pred, output_dict=True))

# Confusion Matrix
st.subheader('Confusion Matrix')
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    ax=ax
)
st.pyplot(fig)
