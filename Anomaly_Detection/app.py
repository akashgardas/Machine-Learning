import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from os import path

BASE_DIR = path.dirname(path.abspath(__file__))
 
# Set page configuration
st.set_page_config(page_title="Anomaly Detection System", layout="wide")
 
# Title and description
st.title("🔍 Anomaly Detection System")
st.markdown("---")
st.markdown("This application uses **Isolation Forest** to detect anomalies in air quality data.")
 
# Load the trained model
model_path = path.join(BASE_DIR, 'model', 'isolation_forest_air_quality.joblib')
 
try:
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error(f"❌ Model file not found: {model_path}")
    st.info("Please ensure the model file 'isolation_forest_air_quality.joblib' is available.")
    st.stop()
 
# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Model Information"])
 
# ===== TAB 1: Single Prediction =====
with tab1:
    st.subheader("Single Data Point Prediction")
   
    col1, col2 = st.columns(2)
   
    with col1:
        co_gt = st.number_input("CO(GT) - Carbon Monoxide", min_value=0.0, value=2.0, step=0.1)
        nox_gt = st.number_input("NOx(GT) - Nitrogen Oxides", min_value=0.0, value=150.0, step=1.0)
   
    with col2:
        c6h6_gt = st.number_input("C6H6(GT) - Benzene", min_value=0.0, value=10.0, step=0.1)
        no2_gt = st.number_input("NO2(GT) - Nitrogen Dioxide", min_value=0.0, value=100.0, step=1.0)
   
    if st.button("🔮 Predict", key="single_predict"):
        # Prepare input data
        input_data = pd.DataFrame({
            'CO(GT)': [co_gt],
            'C6H6(GT)': [c6h6_gt],
            'NOx(GT)': [nox_gt],
            'NO2(GT)': [no2_gt]
        })
       
        # Make prediction
        prediction = model.predict(input_data)[0]
        anomaly_score = model.score_samples(input_data)[0]
       
        # Display results
        st.markdown("---")
        col1, col2 = st.columns(2)
       
        with col1:
            if prediction == -1:
                st.error("⚠️ ANOMALY DETECTED")
            else:
                st.success("✅ NORMAL")
       
        with col2:
            st.metric("Anomaly Score", f"{anomaly_score:.4f}")
       
        # Display input values
        st.subheader("Input Values:")
        st.table(input_data)
 
# ===== TAB 2: Batch Prediction =====
with tab2:
    st.subheader("Batch Prediction from CSV")
   
    uploaded_file = st.file_uploader("Upload a CSV file with columns: CO(GT), C6H6(GT), NOx(GT), NO2(GT)",
                                      type="csv")
   
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
           
            # Check if required columns exist
            required_columns = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
            if not all(col in df.columns for col in required_columns):
                st.error(f"❌ CSV must contain columns: {required_columns}")
            else:
                # Select only required columns
                df_filtered = df[required_columns]
               
                if st.button("🔮 Predict Batch", key="batch_predict"):
                    # Make predictions
                    predictions = model.predict(df_filtered)
                    anomaly_scores = model.score_samples(df_filtered)
                   
                    # Add predictions to dataframe
                    results_df = df_filtered.copy()
                    results_df['Prediction'] = predictions
                    results_df['Anomaly_Score'] = anomaly_scores
                    results_df['Status'] = results_df['Prediction'].apply(
                        lambda x: '⚠️ ANOMALY' if x == -1 else '✅ NORMAL'
                    )
                   
                    # Display results
                    st.success(f"✅ Processed {len(results_df)} records")
                   
                    col1, col2 = st.columns(2)
                    with col1:
                        normal_count = (results_df['Prediction'] == 1).sum()
                        st.metric("Normal Records", normal_count)
                    with col2:
                        anomaly_count = (results_df['Prediction'] == -1).sum()
                        st.metric("Anomalies Detected", anomaly_count)
                   
                    # Display dataframe
                    st.subheader("Detailed Results:")
                    st.dataframe(results_df, use_container_width=True)
                   
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="anomaly_predictions.csv",
                        mime="text/csv"
                    )
                   
                    # Visualization
                    st.subheader("Anomaly Score Distribution:")
                    fig, ax = plt.subplots(figsize=(10, 5))
                   
                    normal_scores = anomaly_scores[predictions == 1]
                    anomaly_scores_filtered = anomaly_scores[predictions == -1]
                   
                    ax.hist(normal_scores, bins=30, alpha=0.6, label='Normal', color='blue')
                    ax.hist(anomaly_scores_filtered, bins=30, alpha=0.6, label='Anomaly', color='red')
                    ax.set_xlabel('Anomaly Score')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Anomaly Scores')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                   
                    st.pyplot(fig)
                   
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
 
# ===== TAB 3: Model Information =====
with tab3:
    st.subheader("Model Information")
   
    st.write("**Model Type:** Isolation Forest")
    st.write("**Purpose:** Detect anomalies in air quality data")
   
    model_params = model.get_params()
   
    col1, col2 = st.columns(2)
   
    with col1:
        st.write("**Model Parameters:**")
        for key, value in model_params.items():
            st.write(f"- {key}: {value}")
   
    with col2:
        st.write("**Features Used:**")
        features = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
        for feature in features:
            st.write(f"- {feature}")
   
    st.markdown("---")
    st.write("**Model Interpretation:**")
    st.info("""
    - **Prediction = 1**: Data point is NORMAL
    - **Prediction = -1**: Data point is an ANOMALY
    - **Anomaly Score**: Lower scores indicate more anomalous behavior
    """)
 
 