import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load model and dataset
model = joblib.load("model/lgbm_model.pkl")
dataset = pd.read_csv("data/cleaned_diabetes_data.csv")

# Streamlit page config
st.set_page_config(page_title="SUGAR AI: Diabetes Predictor", page_icon="🩺", layout="centered")

# Logo and title
st.image("images/logo.jpeg", width=250)
st.title("🩺 SUGAR AI: Diabetes Predictor")
st.markdown("#### Your Personal Diabetes Risk Assessment Tool")

st.markdown("""
Provide the patient's health details below, and SUGAR AI will estimate their risk of diabetes,
along with a comparison to a broader population.
""")

# Input form
with st.form("diabetes_form"):
    st.header("👤 Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
        hypertension = st.checkbox("Hypertension (High Blood Pressure)")
        heart_disease = st.checkbox("Heart Disease")

    with col2:
        smoking_history = st.selectbox("Smoking History", [
            "never", "No Info", "former", "current", "ever", "not current"
        ])
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, step=0.1, value=22.0)
        hba1c_level = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, step=0.1, value=5.5)
        blood_glucose_level = st.number_input("Blood Glucose Level (mg/dL)", min_value=50, max_value=300, step=1, value=100)

    submitted = st.form_submit_button("✨ Predict Diabetes Risk")

if submitted:
    # Encode gender and smoking history to match model expectations
    gender_map = {"Female": 0, "Male": 1, "Other": 2}
    smoking_map = {
        "never": 0, "No Info": 1, "former": 2,
        "current": 3, "ever": 4, "not current": 5
    }

    input_data = pd.DataFrame([{
        "gender": gender_map[gender],
        "age": age,
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),
        "smoking_history": smoking_map[smoking_history],
        "bmi": bmi,
        "hba1c_level": hba1c_level,
        "blood_glucose_level": blood_glucose_level
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    probability_percent = round(probability * 100, 2)

    # Display result
    st.subheader("💡 Prediction Results")
    if prediction == 1:
        st.markdown(f"<div style='color: #F44336; font-size: 24px; font-weight: bold;'>🚨 High likelihood of Diabetes</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='color: #4CAF50; font-size: 24px; font-weight: bold;'>✅ No Diabetes Detected</div>", unsafe_allow_html=True)

    st.markdown(f"**Probability of Diabetes:** `{probability_percent}%`")

    # Risk interpretation
    if probability >= 0.7:
        st.error("❗ High Risk: Please consult a medical professional.")
    elif probability >= 0.4:
        st.warning("🔶 Moderate Risk: Consider lifestyle changes and regular check-ups.")
    else:
        st.success("🟢 Low Risk: Maintain healthy habits and preventive care.")

    # KDE plot
    st.subheader("📊 Your Risk vs. Population Distribution")

    # Calculate KDE
    population_probs = model.predict_proba(dataset.drop(columns=["diabetes"]))[:, 1]
    kde = gaussian_kde(population_probs)
    x_vals = np.linspace(0, 1, 1000)
    y_vals = kde(x_vals)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_vals, y_vals, color='#3F51B5')
    ax.fill_between(x_vals, y_vals, alpha=0.1)

    # Mark user probability
    ax.axvline(probability, color='#FF5722', linestyle="--", linewidth=2)
    ax.set_title("Distribution of Diabetes Risk in Historical Data")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")

    st.pyplot(fig)

    # Percentile rank
    percentile_rank = round((population_probs < probability).mean() * 100, 1)
    st.markdown(f"Your risk is higher than **{percentile_rank}%** of individuals in our data.")

    # Optional: Detailed input summary
    with st.expander("🧾 View Submitted Information"):
        st.json(input_data.to_dict(orient="records")[0])
