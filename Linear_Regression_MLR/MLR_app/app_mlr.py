import streamlit as st
import  seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

st.set_page_config("Mutliple Linear Regression", layout="centered")

# load css
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")



# Title

st.markdown("""
    <div class="card">
        <h1>Mutliple Linear Regression </h1>
        <p>Predict <b> Tip Amount </b> from <b>Total Bill and Size </b> using Linear Regression... </p>   

    </div> 
            """, unsafe_allow_html=True)


# Load dataset


@st.cache_data

def load_data():
    data = sns.load_dataset('tips')
    return data

df = load_data()

# Dataset Preview

st.markdown('<div class = "card">'
            '<h3>Dataset Preview</h2>', unsafe_allow_html=True)
st.dataframe(df[['total_bill', 'size', 'tip']].head())
st.markdown('</div>', unsafe_allow_html=True)

# Prepare data

X, y = df[['total_bill', 'size']], df['tip']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# model evaluation

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
Adujusted_R2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
st.markdown('<div class = "card">'
            '<h3>Model Evaluation Metrics</h2>', unsafe_allow_html=True)
st.markdown(f'''
    <ul>
        <li>Mean Absolute Error (MAE): {mae:.2f}</li>
        <li>Root Mean Squared Error (RMSE): {rmse:.2f}</li>
        <li>R-squared (R2 ): {r2:.2f}</li>
        <li>Adjusted R-squared: {Adujusted_R2:.2f}</li>
    </ul>
''', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# visualization

st.markdown('<div class = "card">'
            "<h3>Total Bill vs Tip (with multiple linear regression)</h3>", unsafe_allow_html=True)
fig,ax = plt.subplots()
ax.scatter(df["total_bill"], df["tip"], alpha=0.6)
ax.plot(df['total_bill'], model.predict(scaler.transform(X)), color='red')
ax.set_xlabel("Total Bill ($)")
ax.set_ylabel("Tip ($)")
st.pyplot(fig)

st.markdown("""
</div>
""", unsafe_allow_html=True)

# performance
st.markdown("""
    <div class="card">
            """
            '<h3>Model Performance</h3>',
            unsafe_allow_html=True)
c1,c2,c3,c4 = st.columns(4)
c1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
c2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
c3.metric("R-squared (R²) Score", f"{r2:.2f}")
c4.metric("Adjusted R-squared", f"{Adujusted_R2:.2f}")

st.markdown("""
    </div>
            """, unsafe_allow_html=True)


# Slope(m) and Intercept(c)
st.markdown(f"""
    <div class="card">
            <h3>Model Interception</h3>
            <p><B> co-efficient (Total bill) : </B> {model.coef_[0]:.3f}<br>
            <p><B> co-efficient (Size) : </B> {model.coef_[1]:.3f}<br>
            <b>Intercept: </b> {model.intercept_:.3f}</p>
            <p>
            Tip Depends upon the <b>Total Bill</b> and <b> Size </b> of the table.
            </p>
    </div>
            """, unsafe_allow_html=True)



# Predicition

st.markdown("""
    <div class="card">
        <h3>Predict Tip Amount</h3>
            """, unsafe_allow_html=True)

max_bill = float(df.total_bill.max())
min_bill = float(df.total_bill.min())
value = 30.0

st.markdown(
    f"""
    <div class="card">
        <p>Max bill: {max_bill}</p>
        <p>Min bill: {min_bill}</p>
        <p>Value: {value}</p>
    """,
    unsafe_allow_html=True,
)

total_bill = st.slider(
    "Select Total Bill Amount ($)",
    min_value=min_bill,
    max_value=max_bill,
    value=value,
    step=0.1,
)

max_size = int(df.shape[0])
min_size = 1
value_size = 2

st.markdown(
    f"""
    <div class="card">
        <p>Max Size: {max_size}</p>
        <p>Min Size: {min_size}</p>
        <p>Value: {value_size}</p>
    """,
    unsafe_allow_html=True,
)

table_size = st.slider(
    "Select Table Size",
    min_value=min_size,
    max_value=max_size,
    value=value_size,
    step=1,
)

input_scaled = scaler.transform([[total_bill, table_size]])

tip = model.predict(input_scaled)[0]

st.markdown(f"""
    <div class="prediction-box"> Predicted Tip Amount: $ {tip:.2f} </div>
            """, unsafe_allow_html=True)

st.markdown("""
</div>
            """, unsafe_allow_html=True)