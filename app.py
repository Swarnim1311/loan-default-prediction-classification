"""Loan default prediction — small Streamlit app.

Loads models/loan_default_model.joblib (full preprocessing + model pipeline)
and predicts default probability from raw borrower inputs.
"""
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/loan_default_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("Loan Default Prediction")
st.write("Enter borrower details to estimate the probability of default.")

model = load_model()
FEATURES = list(model.feature_names_in_)

st.subheader("Loan information")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=69, value=43)
    income = st.number_input("Income", min_value=15000, max_value=149999, value=82466)
    loan_amount = st.number_input("Loan amount", min_value=5000, max_value=249999, value=127556)
    credit_score = st.number_input("Credit score", min_value=300, max_value=849, value=574)
    months_employed = st.number_input("Months employed", min_value=0, max_value=119, value=60)
with col2:
    num_credit_lines = st.selectbox("Credit lines", [1, 2, 3, 4], index=1)
    interest_rate = st.number_input("Interest rate (%)", min_value=2.0, max_value=25.0, value=13.46)
    loan_term = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60], index=2)
    dti_ratio = st.number_input("DTI ratio", min_value=0.1, max_value=0.9, value=0.5)
    education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"], index=1)

col3, col4 = st.columns(2)
with col3:
    employment = st.selectbox("Employment", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    marital = st.selectbox("Marital status", ["Divorced", "Married", "Single"], index=1)
    purpose = st.selectbox("Loan purpose", ["Auto", "Business", "Education", "Home", "Other"])
with col4:
    mortgage = st.selectbox("Has mortgage", ["No", "Yes"])
    dependents = st.selectbox("Has dependents", ["No", "Yes"])
    cosigner = st.selectbox("Has co-signer", ["No", "Yes"])

if st.button("Predict"):
    row = {
        "Age": age, "Income": income, "LoanAmount": loan_amount,
        "CreditScore": credit_score, "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines, "InterestRate": interest_rate,
        "LoanTerm": loan_term, "DTIRatio": dti_ratio,
        "loan_to_income": loan_amount / income if income else 0,
        "Education": education, "EmploymentType": employment,
        "MaritalStatus": marital, "HasMortgage": mortgage,
        "HasDependents": dependents, "LoanPurpose": purpose,
        "HasCoSigner": cosigner,
    }
    X_one = pd.DataFrame([{c: row[c] for c in FEATURES}])
    proba = float(model.predict_proba(X_one)[0, 1])

    st.subheader("Result")
    st.write(f"**Default probability:** {proba:.1%}")
    st.write(f"**Prediction:** {'Default' if proba >= 0.5 else 'No Default'}")

st.subheader("About this prediction")
st.write(
    "This is a model estimate from a Logistic Regression trained on historical loan data, "
    "not certainty. Do not use it as the sole basis for real lending decisions."
)
