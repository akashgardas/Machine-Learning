import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Page Config
st.set_page_config('Linear Regression', layout='centered')

# Load CSS
def load_css(file):
    with open (file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('./styles.css')

# Load Data
@st.cache_data
def load_data():
    return sns.load_dataset('tips')
    
df = load_data()

# Prepare Data
X, y = df[['total_bill']], df['tip']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('''
            <div class='card'>
                <h1>Linear Regresion (SLR)</h1>
                <p>Predict <b>Tip Amount</b> from <b>Bill Amount</b></p>
            </div>
            ''', unsafe_allow_html=True)
    
with col2:
    # Dataset Preview
    st.markdown('''
                <div class='card' style="color: blue; text-align: center; background-color: lightblue; padding: 0px;">
                    <h3>Dataset Preview</h3>
                </div>
                ''', unsafe_allow_html=True
                )
    st.dataframe(df.head())

with col3:
    st.markdown('''
            <div class='card' style="color: blue; text-align: center; background-color: lightblue; padding: 0px;">
                <h3>Total Bill vs Tip</h3>
            </div>
            ''', unsafe_allow_html=True
            )

    fig, ax = plt.subplots()
    ax.scatter(df['total_bill'], df['tip'], alpha=0.6)
    ax.plot(df['total_bill'], model.predict(scaler.transform(X)), color='red')
    ax.set_xlabel('Total Bill')
    ax.set_ylabel('Tip')
    st.pyplot(fig)
