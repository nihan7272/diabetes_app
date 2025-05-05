# diabetes_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and prepare data
@st.cache_data
def load_model():
    df = pd.read_csv("diabetes_clean.csv")
    
    # Replace zeros with median in certain columns
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zeros:
        df[col] = df[col].replace(0, df[col][df[col] != 0].median())
        
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    return model, scaler

# Load model
model, scaler = load_model()

# App UI
st.title("🧪 Diabetes Prediction App")
st.write("Enter the following health metrics to predict the likelihood of diabetes:")

# Form input
with st.form("diabetes_form"):
    pregnancies = st.number_input("Pregnancies", 0, 20, step=1)
    glucose = st.number_input("Glucose", 50, 200, step=1)
    blood_pressure = st.number_input("Blood Pressure", 30, 150, step=1)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, step=1)
    insulin = st.number_input("Insulin", 0, 900, step=1)
    bmi = st.number_input("BMI", 10.0, 70.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, step=0.01)
    age = st.number_input("Age", 10, 100, step=1)

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.markdown("---")
        if prediction == 1:
            st.error("⚠️ The model predicts **diabetes**.")
        else:
            st.success("✅ The model predicts **no diabetes**.")

# diabetes_app
