import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Page Config
st.set_page_config('Linear Regression', layout='centered')

# Load CSS
def load_css(file):
    css_path = Path(__file__).parent / file
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('styles.css')

st.markdown('''
            <div class='card'>
                <h1>Linear Regresion (SLR)</h1>
                <p>Predict <b>Tip Amount</b> from <b>Bill Amount</b></p>
            </div>
            ''', unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    return sns.load_dataset('tips')
    
df = load_data()

# Dataset Preview
st.markdown('''
            <div class='card' style="color: blue; text-align: center; background-color: lightblue; padding: 0px;">
                <h3>Dataset Preview</h3>
            </div>
            ''', unsafe_allow_html=True
            )
st.dataframe(df.head())

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

# Visualization
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
# st.write(fig)
st.pyplot(fig)

# Performance
st.markdown('''
            <div class='card' style="color: blue; text-align: center; background-color: lightblue; padding: 0px;">
                <h3>Model Performance</h3>
            </div>
            ''', unsafe_allow_html=True
            )

# c1, c2, c3 = st.columns(3)
# c1.metric('MAE', round(mae, 2))
# c2.metric('MSE', round(mse, 2))
# c3.metric('RMSE', round(rmse, 2))
st.dataframe(pd.DataFrame(data={
    'Metric': ['MAE', 'MSE', 'RMSE'],
    'Value':[mae, mse, rmse] 
}))

# Slope and Intercept
st.markdown(f'''
            <div class='card' style="background-color: lightblue;">
                <h3>Model Interception</h3>
                <p> <b>Slope</b>: {model.coef_[0]}</p>
                <p> <b>Intercept</b>: {model.intercept_}</p>
            </div>
            ''', unsafe_allow_html=True
            )

# Prediction
st.markdown('''
            <div class='card' style="color: blue; text-align: center; background-color: lightblue; padding: 0px;">
                <h3>Predict Tip Amount</h3>
            </div>
            ''', unsafe_allow_html=True
            )

bill = st.slider('What is the Bill Amount?', np.min(df['total_bill']), np.max(df['total_bill']))

tip = model.predict(scaler.transform([[bill]]))[0]
st.markdown(f'<div class="prediction-box"> Predicted Tip: {tip:.2f}</div>', unsafe_allow_html=True)

