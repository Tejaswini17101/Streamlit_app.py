# Gender >- 1 Female 0 Male
# Churn >- 1 Yes 0 No
# Scaler is exported as scaler.pkl
# Model is exported as model.pkl
# Oder of the x -> 'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st
import joblib
import numpy as np


scaler = joblib.load("scaler.pkl")
model= joblib.load("model.pkl")


st.ttile("Churn Prediction App")

st.divider()

st.write("Please enter he values and hit the predic button for get a prediction.")

st.divider()

age = st.number_input("Enter Age", min_value=10, max_value = 100, value=30)


tenure = st.number_input("Enter Tenure", min_value = 0, max_value=130, value =10)

monthlycharge = st.number_input("Enter MonthlyCharge", min_value=30, max_value=150)

gender = st.selectbox("Enter the Gender",["Female", "Male"])

st.divider()

predictbutton = st.button("Predict!")

if predictbutton:

    gender_selected = 1 if gender == "Female" else 0

    x = [age, gender_selected, tenure, monthlycharge]

    x1 = np.array(x)

    x_array = scaler.transform([x1])

    prediction = model.predict(x_array)[0]

    predicted = "Yes" if prediction == 1 else "No"

    st.write(f"Predicted: {predicted}")

else:
    st.write("Please enter the values and use predict button")
