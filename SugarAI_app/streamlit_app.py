import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import base64
import os

# --- Resolve absolute paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "optimized_lightgbm_model.pkl")
data_path = os.path.join(BASE_DIR, "data", "diabetes_prediction_dataset.csv")
logo_path = os.path.join(BASE_DIR, "images", "logo.jpeg")

# --- Helper to encode logo ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

# --- Load model and dataset ---
model = joblib.load(model_path)
dataset = pd.read_csv(data_path)

# --- Streamlit Page Config ---
st.set_page_config(page_title="SUGAR AI: Diabetes Predictor", page_icon="🩺", layout="centered")

# --- Display Header with Logo ---
image_base64 = get_base64_image(logo_path)
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="{image_base64}" width="250" style="margin-bottom: 20px;" />
        <h1 style="color: #E91E63;">🩺 SUGAR AI: Diabetes Predictor</h1>
        <p style="font-size: 18px; color: #666;">Your Personal Diabetes Risk Assessment Tool</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Disclaimer Section ---
st.markdown("""
<div style="padding: 15px; background-color: #f9f9f9; border-left: 4px solid #E91E63; font-size: 15px;">
⚠️ <strong>Disclaimer:</strong> This tool is intended for <strong>educational and informational purposes only</strong>. 
It is based on a machine learning model trained using publicly available health data from 
<a href="https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset" target="_blank"><strong>Kaggle</strong></a>.
<br><br>
It is <strong>not a substitute for professional medical advice, diagnosis, or treatment</strong>. Always consult with a qualified healthcare provider regarding any medical concerns or conditions.
</div>
""", unsafe_allow_html=True)

# --- Input Form ---
with st.form("diabetes_form"):
    st.markdown("### 👤 Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
        hypertension = st.checkbox("Hypertension (High Blood Pressure)")
        heart_disease = st.checkbox("Heart Disease")

    with col2:
        # Display-friendly labels but keep original backend encoding
        smoking_labels = {
            "never": "Never Smoked",
            "No Info": "No Information",
            "former": "Former Smoker",
            "current": "Current Smoker",
            "ever": "Has Smoked",
            "not current": "Not Currently Smoking"
        }
        smoking_display = st.selectbox("Smoking History", list(smoking_labels.values()))
        # Reverse map back to original
        smoking_history = [k for k, v in smoking_labels.items() if v == smoking_display][0]

        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, step=0.1, value=22.0)
        hba1c_level = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, step=0.1, value=5.5)
        blood_glucose_level = st.number_input("Blood Glucose Level (mg/dL)", min_value=50, max_value=300, step=1, value=100)

    submitted = st.form_submit_button("✨ Predict Diabetes Risk")

# --- Prediction Output ---
if submitted:
    gender_map = {"Female": "Female", "Male": "Male", "Other": "Other"}
    gender_val = gender_map[gender]

    input_data = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": hba1c_level,
        "blood_glucose_level": blood_glucose_level
    }])

    # Preprocess input for prediction
    gender_enc = {"Female": 0, "Male": 1, "Other": 2}
    smoking_enc = {
        "never": 0, "No Info": 1, "former": 2,
        "current": 3, "ever": 4, "not current": 5
    }

    model_input = pd.DataFrame([{
        "gender": gender_enc[gender],
        "age": age,
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),
        "smoking_history": smoking_enc[smoking_history],
        "bmi": bmi,
        "HbA1c_level": hba1c_level,
        "blood_glucose_level": blood_glucose_level
    }])

    prediction = model.predict(model_input)[0]
    probability = model.predict_proba(model_input)[0][1]
    probability_percent = round(probability * 100, 2)

    # --- Result Display ---
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

    st.markdown(f"<div style='font-size: 20px;'><strong>Probability of Diabetes:</strong> <span style='color: #3F51B5;'>{probability_percent}%</span></div>", unsafe_allow_html=True)

    # --- Risk Interpretation ---
    if probability >= 0.7:
        st.error("❗ High Risk: Please consult a medical professional.")
    elif probability >= 0.4:
        st.warning("🔶 Moderate Risk: Consider lifestyle changes and regular check-ups.")
    else:
        st.success("🟢 Low Risk: Maintain healthy habits and preventive care.")

    # --- Risk Distribution Plot ---
    st.markdown("### 📊 Your Risk vs. Population Distribution")
    population_probs = model.predict_proba(dataset.drop(columns=["diabetes"]))[:, 1]
    kde = gaussian_kde(population_probs)
    x_vals = np.linspace(0, 1, 1000)
    y_vals = kde(x_vals)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_vals, y_vals, color='#3F51B5')
    ax.fill_between(x_vals, y_vals, alpha=0.1)
    ax.axvline(probability, color='#FF5722', linestyle="--", linewidth=2)
    ax.set_title("Distribution of Diabetes Risk in Historical Data")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    st.pyplot(fig)

    percentile_rank = round((population_probs < probability).mean() * 100, 1)
    st.markdown(f"📈 Your diabetes risk is higher than **{percentile_rank}%** of individuals in the dataset.")

    # --- HbA1c Distribution ---
    st.markdown("### 🧪 HbA1c Level Distribution (Compared to Same Gender)")

    gender_filtered = dataset[dataset["gender"] == gender_val]
    if not gender_filtered.empty:
        hba1c_vals = gender_filtered["HbA1c_level"].dropna()
        if len(hba1c_vals) > 1:
            kde_hba1c = gaussian_kde(hba1c_vals)
            x_hba1c = np.linspace(hba1c_vals.min(), hba1c_vals.max(), 500)
            y_hba1c = kde_hba1c(x_hba1c)

            fig_hba1c, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(x_hba1c, y_hba1c, color="#3F51B5")
            ax1.fill_between(x_hba1c, y_hba1c, alpha=0.1)
            ax1.axvline(hba1c_level, color="#FF5722", linestyle="--", linewidth=2)
            ax1.set_title(f"HbA1c Distribution - {gender}")
            ax1.set_xlabel("HbA1c Level (%)")
            ax1.set_ylabel("Density")
            st.pyplot(fig_hba1c)

            hba1c_percentile = round((hba1c_vals < hba1c_level).mean() * 100, 1)
            st.markdown(f"Your HbA1c level is higher than **{hba1c_percentile}%** of {gender.lower()}s in the dataset.")
        else:
            st.warning("Not enough data for this gender to plot HbA1c distribution.")
    else:
        st.warning("No data available for selected gender to generate HbA1c distribution.")

    # --- Blood Glucose Distribution ---
    st.markdown("### 🩸 Blood Glucose Level Distribution (Compared to Same Gender)")

    if not gender_filtered.empty:
        glucose_vals = gender_filtered["blood_glucose_level"].dropna()
        if len(glucose_vals) > 1:
            kde_glucose = gaussian_kde(glucose_vals)
            x_glucose = np.linspace(glucose_vals.min(), glucose_vals.max(), 500)
            y_glucose = kde_glucose(x_glucose)

            fig_glucose, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(x_glucose, y_glucose, color="#3F51B5")
            ax2.fill_between(x_glucose, y_glucose, alpha=0.1)
            ax2.axvline(blood_glucose_level, color="#FF5722", linestyle="--", linewidth=2)
            ax2.set_title(f"Blood Glucose Distribution - {gender}")
            ax2.set_xlabel("Blood Glucose Level (mg/dL)")
            ax2.set_ylabel("Density")
            st.pyplot(fig_glucose)

            glucose_percentile = round((glucose_vals < blood_glucose_level).mean() * 100, 1)
            st.markdown(f"Your blood glucose level is higher than **{glucose_percentile}%** of {gender.lower()}s in the dataset.")
        else:
            st.warning("Not enough data for this gender to plot blood glucose distribution.")
    else:
        st.warning("No data available for selected gender to generate glucose distribution.")

    # --- Show Input ---
    with st.expander("🧾 View Submitted Information"):
        st.json(input_data.to_dict(orient="records")[0])
