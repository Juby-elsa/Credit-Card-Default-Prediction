import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Credit Card Default Prediction", layout="centered")

# ---------- CACHED FUNCTIONS ----------

@st.cache_data
def load_data():
    df = pd.read_csv("UCI_Credit_Card.csv")
    df.drop("ID", axis=1, inplace=True)
    return df

@st.cache_resource
def train_model(df):
    X = df.drop("default.payment.next.month", axis=1)
    y = df["default.payment.next.month"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    feature_means = X.mean()
    return model, scaler, feature_means, X.columns

# ---------- LOAD ONCE ----------

df = load_data()
model, scaler, feature_means, feature_order = train_model(df)

# ---------- UI ----------

st.title("💳 Credit Card Default Prediction")
st.write(
    "This application predicts whether a customer is likely to **default on their next credit card payment** "
    "based on repayment history and financial profile."
)

st.header("Enter Customer Details")

limit_bal = st.slider("Credit Limit", 10000, 1000000, 200000, step=10000)
age = st.slider("Age", 21, 75, 30)

sex = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["Graduate School", "University", "High School", "Other"])
marriage = st.selectbox("Marital Status", ["Married", "Single", "Other"])

pay_0 = st.slider("Last Month Payment Delay (PAY_0)", -1, 8, 0)
pay_2 = st.slider("Payment Delay 2 Months Ago (PAY_2)", -1, 8, 0)
pay_3 = st.slider("Payment Delay 3 Months Ago (PAY_3)", -1, 8, 0)
pay_4 = st.slider("Payment Delay 4 Months Ago (PAY_4)", -1, 8, 0)
pay_5 = st.slider("Payment Delay 5 Months Ago (PAY_5)", -1, 8, 0)
pay_6 = st.slider("Payment Delay 6 Months Ago (PAY_6)", -1, 8, 0)

sex_val = 1 if sex == "Male" else 2
edu_map = {"Graduate School": 1, "University": 2, "High School": 3, "Other": 4}
mar_map = {"Married": 1, "Single": 2, "Other": 3}

input_data = feature_means.copy()

input_data.update({
    "LIMIT_BAL": limit_bal,
    "AGE": age,
    "SEX": sex_val,
    "EDUCATION": edu_map[education],
    "MARRIAGE": mar_map[marriage],
    "PAY_0": pay_0,
    "PAY_2": pay_2,
    "PAY_3": pay_3,
    "PAY_4": pay_4,
    "PAY_5": pay_5,
    "PAY_6": pay_6
})

input_df = pd.DataFrame([input_data])[feature_order]

# ---------- PREDICT ----------

if st.button("Predict Default Risk"):
    prob = model.predict_proba(input_df)[0][1]

    if prob >= 0.6:
        st.error(f"⚠️ High Risk of Default\n\nProbability: {prob*100:.2f}%")
    elif prob <= 0.4:
        st.success(f"✅ Low Risk (No Default Expected)\n\nProbability: {prob*100:.2f}%")
    else:
        st.warning(f"⚖️ Moderate Risk\n\nProbability: {prob*100:.2f}%") 