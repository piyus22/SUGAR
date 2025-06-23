import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import base64
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "optimized_lightgbm_model.pkl")
data_path = os.path.join(BASE_DIR, "data", "diabetes_prediction_dataset.csv")
logo_path = os.path.join(BASE_DIR, "images", "logo.jpeg")
performance_image1 = os.path.join(BASE_DIR, "images", "lightGBM_fine_tuned.png")
performance_image2 = os.path.join(BASE_DIR, "images", "LightGBM_fine_tuned_table.png")

# Online icon URLs
GITHUB_ICON_URL = "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg"
LINKEDIN_ICON_URL = "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg"

# --- Helper ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/png;base64,{encoded}"

# --- Load model & data ---
model = joblib.load(model_path)
dataset = pd.read_csv(data_path)

# --- Page setup ---
st.set_page_config(page_title="SUGAR AI: Diabetes Predictor", page_icon="🩺", layout="centered")

# --- Header + Disclaimer ---
st.markdown(
    f"""
    <div style="text-align: center; max-width: 700px; margin: auto;">
        <img src="{get_base64_image(logo_path)}" width="250" style="margin-bottom: 10px;" />
        <h1 style="color: #E91E63;">🩺 SUGAR AI: Diabetes Predictor</h1>
        <p style="font-size: 17px; color: #aaa;">Your Personal Diabetes Risk Assessment Tool</p>
    </div>
    <div style="margin: 10px auto 20px auto; background-color: rgba(240,240,240,0.1); padding: 12px 16px; border-radius: 8px; max-width: 700px; font-size: 15px; line-height: 1.5; text-align: justify; color: inherit;">
        <b>Disclaimer:</b> This application is designed for <b>educational and illustrative purposes only</b>. It is not intended to diagnose, treat, or replace medical advice. The model is trained on a public dataset available on 
        <a href="https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset" target="_blank" style="text-decoration: none;"><b>Kaggle</b></a>. 
        Always consult a qualified healthcare professional for any medical concerns.
    </div>
    """,
    unsafe_allow_html=True
)

# --- Input Form ---
with st.form("diabetes_form"):
    st.markdown("### 👤 Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        age = st.number_input("Age (years)", 1, 120, 30)
        hypertension = st.checkbox("Hypertension (High Blood Pressure)")
        heart_disease = st.checkbox("Heart Disease")
    with col2:
        smoking_history = st.selectbox("Smoking History", [
            "Never Smoked", "No Information", "Former Smoker", 
            "Current Smoker", "Ever Smoked", "Not Currently Smoking"
        ])
        bmi = st.number_input("BMI (Body Mass Index)", 10.0, 60.0, step=0.1, value=22.0)
        hba1c_level = st.number_input("HbA1c Level (%)", 3.0, 15.0, step=0.1, value=5.5)
        blood_glucose_level = st.number_input("Blood Glucose Level (mg/dL)", 50, 300, step=1, value=100)

    submitted = st.form_submit_button("✨ Predict Diabetes Risk")

# --- Mapping ---
gender_map = {"Female": "Female", "Male": "Male", "Other": "Other"}
gender_enc = {"Female": 0, "Male": 1, "Other": 2}
smoking_display = {
    "Never Smoked": "never", "No Information": "No Info", "Former Smoker": "former",
    "Current Smoker": "current", "Ever Smoked": "ever", "Not Currently Smoking": "not current"
}
smoking_enc = {
    "never": 0, "No Info": 1, "former": 2,
    "current": 3, "ever": 4, "not current": 5
}

# --- Prediction Output ---
if submitted:
    gender_val = gender_map[gender]
    smoking_backend = smoking_display[smoking_history]

    input_data = pd.DataFrame([{
        "gender": gender_val,
        "age": age,
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),
        "smoking_history": smoking_backend,
        "bmi": bmi,
        "HbA1c_level": hba1c_level,
        "blood_glucose_level": blood_glucose_level
    }])

    model_input = pd.DataFrame([{
        "gender": gender_enc[gender],
        "age": age,
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),
        "smoking_history": smoking_enc[smoking_backend],
        "bmi": bmi,
        "HbA1c_level": hba1c_level,
        "blood_glucose_level": blood_glucose_level
    }])

    prediction = model.predict(model_input)[0]
    probability = model.predict_proba(model_input)[0][1]
    prob_percent = round(probability * 100, 2)

    st.markdown("### 💡 Prediction Results")
    if prediction == 1:
        st.markdown(
            "<div style='color: #F44336; font-size: 24px; font-weight: bold;'>🚨 High likelihood of Diabetes</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='color: #4CAF50; font-size: 24px; font-weight: bold;'>✅ No Diabetes Detected</div>",
            unsafe_allow_html=True
        )

    st.markdown(f"<div style='font-size: 20px;'>📌 <b>Probability of Diabetes:</b> <span style='color:#FF5722; font-weight:bold;'>{prob_percent}%</span></div>", unsafe_allow_html=True)

    if probability >= 0.7:
        st.error("❗ High Risk: Please consult a medical professional.")
    elif probability >= 0.4:
        st.warning("🔶 Moderate Risk: Consider lifestyle changes and regular check-ups.")
    else:
        st.success("🟢 Low Risk: Maintain healthy habits and preventive care.")

    st.markdown("### 📊 Your Risk vs. Population Distribution")
    population_probs = model.predict_proba(dataset.drop(columns=["diabetes"]))[:, 1]
    kde = gaussian_kde(population_probs)
    x_vals = np.linspace(0, 1, 1000)
    y_vals = kde(x_vals)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_vals, y_vals, color='#3F51B5')
    ax.fill_between(x_vals, y_vals, alpha=0.1)
    ax.axvline(probability, color='#FF5722', linestyle="--", linewidth=2)
    ax.set_title("Distribution of Diabetes Risk in Dataset")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    st.pyplot(fig)

    percentile_rank = round((population_probs < probability).mean() * 100, 1)
    st.markdown(f"📈 Your diabetes risk is higher than <b>{percentile_rank}%</b> of individuals in the dataset.", unsafe_allow_html=True)

    gender_filtered = dataset[dataset["gender"] == gender_val]

    def plot_metric_distribution(values, input_value, label, unit):
        if len(values) > 1:
            kde = gaussian_kde(values)
            x_vals = np.linspace(values.min(), values.max(), 500)
            y_vals = kde(x_vals)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x_vals, y_vals, color="#3F51B5")
            ax.fill_between(x_vals, y_vals, alpha=0.1)
            ax.axvline(input_value, color="#FF5722", linestyle="--", linewidth=2)
            ax.set_title(f"{label} Distribution - {gender}")
            ax.set_xlabel(f"{label} ({unit})")
            ax.set_ylabel("Density")
            st.pyplot(fig)
            pct = round((values < input_value).mean() * 100, 1)
            st.markdown(f"📍 Your {label.lower()} is higher than <b>{pct}%</b> of {gender.lower()}s in the dataset.", unsafe_allow_html=True)

    if not gender_filtered.empty:
        st.markdown("### 🧪 HbA1c Level Distribution (Compared to Same Gender)")
        plot_metric_distribution(gender_filtered["HbA1c_level"].dropna(), hba1c_level, "HbA1c Level", "%")
        st.markdown("### 🡨‍🩺 Blood Glucose Level Distribution (Compared to Same Gender)")
        plot_metric_distribution(gender_filtered["blood_glucose_level"].dropna(), blood_glucose_level, "Blood Glucose Level", "mg/dL")

    with st.expander("🗒️ View Submitted Information"):
        st.json(input_data.to_dict(orient="records")[0])

    with st.expander("📊 View Model Performance (LightGBM Evaluation)"):
        st.image(performance_image1)
        st.image(performance_image2)

# --- Footer ---
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align:center; margin-top: 30px;">
        <p style="font-size:16px;">
            Built with ❤️ by <b>SugarAI</b><br>
            <img src="{get_base64_image(logo_path)}" width="40" style="margin: 10px;" />
            <a href="https://github.com/piyus22/SUGAR" target="_blank">
                <img src="{GITHUB_ICON_URL}" width="26" style="margin: 0 10px; filter: invert(1);" />
            </a>
            <a href="https://www.linkedin.com/in/piyus-mohanty/" target="_blank">
                <img src="{LINKEDIN_ICON_URL}" width="26" style="margin: 0 10px; filter: invert(1);" />
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
