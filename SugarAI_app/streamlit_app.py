# streamlit_app.py

import streamlit as st
import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Configure logging (Streamlit also has its own logging for app output)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants for File Paths ---
MODEL_PATH = 'optimized_lightgbm_model.pkl'
DATASET_PATH = 'diabetes_prediction_dataset.csv'
# Assuming 'image_9af9ba.jpg' is your logo. Make sure it's in the same directory.
LOGO_PATH = 'SugarAI_app/static/images/logo.jpeg'

# --- Custom CSS (from style.css) ---
# Embed this directly into Streamlit using st.markdown(unsafe_allow_html=True)
CUSTOM_CSS = """
<style>
:root {
    --primary-color: #E91E63; /* Pink */
    --secondary-color: #3F51B5; /* Deep Blue */
    --accent-color: #FF5722; /* Orange */
    --success-color: #4CAF50; /* Green */
    --warning-color: #FFC107; /* Yellow */
    --danger-color: #F44336; /* Red */
    --text-color: #333;
    --light-text-color: #666;
    --bg-color: #f9f9f9;
    --card-bg: #ffffff;
    --border-color: #ddd;
    --shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

body {
    font-family: 'Arial', sans-serif;
    line-height: 1.6;
    margin: 0;
    padding: 0;
    background-color: var(--bg-color); /* This won't directly apply to Streamlit's body, use Streamlit theming for main background */
    color: var(--text-color);
}

.stApp { /* Streamlit's main app container */
    background-color: var(--bg-color);
    color: var(--text-color);
}

.container {
    max-width: 900px;
    margin: 30px auto;
    padding: 30px;
    background-color: var(--card-bg);
    border-radius: 12px;
    box-shadow: var(--shadow);
}

header {
    text-align: center;
    margin-bottom: 30px;
}

.app-logo {
    max-width: 250px;
    height: auto;
    display: block;
    margin: 0 auto 20px auto;
    padding: 10px;
}

h1 {
    color: var(--primary-color);
    font-size: 3.5em;
    margin-bottom: 0.2em;
    font-weight: bold;
}

.tagline {
    font-size: 1.3em;
    color: var(--light-text-color);
    margin-top: -10px;
    font-style: italic;
}

h2 {
    color: var(--secondary-color);
    font-size: 2em;
    margin-top: 30px;
    margin-bottom: 15px;
    border-bottom: 2px solid var(--secondary-color);
    padding-bottom: 5px;
}

h3 {
    color: var(--secondary-color);
    font-size: 1.4em;
    margin-top: 20px;
    margin-bottom: 10px;
}

.icon {
    margin-right: 8px;
    font-size: 1.2em;
    vertical-align: middle;
}

.intro {
    text-align: center;
    margin-bottom: 30px;
    padding: 15px;
    background-color: #eaf0fb;
    border-radius: 8px;
    border: 1px solid #c9d8ee;
}

.intro p {
    font-size: 1.1em;
    color: var(--light-text-color);
}

/* Form Styling - Streamlit widgets handle most of this, but general container styles apply */
.st-emotion-cache-1xw8fcm, .st-emotion-cache-1cypd85 { /* Targeting Streamlit form columns/elements */
    background-color: #fcfcfc;
    padding: 25px;
    border-radius: 10px;
    border: 1px solid var(--border-color);
}

/* Specific adjustments for Streamlit elements to match design */
div[data-testid="stForm"] {
    border: none; /* Remove default Streamlit form border */
    padding: 0;
}

.stButton > button {
    background-color: var(--accent-color);
    color: white;
    padding: 12px 28px;
    border: none;
    border-radius: 8px;
    font-size: 1.1em;
    transition: background-color 0.3s ease;
    width: 100%; /* Make button full width */
    margin-top: 30px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #e64a00;
    color: white; /* Keep text white on hover */
}

/* Results Section */
.results-section {
    margin-top: 40px;
    padding: 25px;
    background-color: #f5faff;
    border-radius: 12px;
    border: 1px solid #d0e0f8;
    box-shadow: var(--shadow);
}

.result-box {
    text-align: center;
    margin-bottom: 25px;
    padding: 20px;
    background-color: #fff;
    border-radius: 10px;
    border: 1px solid var(--border-color);
}

.prediction-status {
    font-size: 2em;
    font-weight: bold;
    margin-bottom: 10px;
    padding: 10px;
    border-radius: 8px;
    display: inline-block;
}

.diabetes-risk {
    color: var(--danger-color);
    background-color: #ffe0e0;
}

.no-diabetes-risk {
    color: var(--success-color);
    background-color: #e0ffe0;
}

.probability {
    font-size: 1.8em;
    color: var(--primary-color);
    margin-top: 10px;
    font-weight: bold;
}

.highlight-prob {
    color: var(--primary-color);
}

.highlight-rank {
    color: var(--accent-color);
}

.risk-interpretation {
    margin-top: 20px;
    text-align: left;
}

/* Streamlit alerts will replace these */
/*
.alert {
    padding: 15px;
    margin-bottom: 15px;
    border-radius: 8px;
    font-size: 0.95em;
    line-height: 1.5;
}

.alert.success-alert {
    background-color: #e6ffed;
    border-left: 5px solid var(--success-color);
    color: #28a745;
}

.alert.warning-alert {
    background-color: #fff3e0;
    border-left: 5px solid var(--warning-color);
    color: #ff9800;
}

.alert.danger-alert {
    background-color: #ffebee;
    border-left: 5px solid var(--danger-color);
    color: #d32f2f;
}
*/
.error-alert { /* For general error messages */
    background-color: #ffe6e6;
    border-left: 5px solid #ff0000;
    color: #d8000c;
    padding: 15px;
    margin-bottom: 20px;
    border-radius: 8px;
    font-weight: bold;
}


.plotly-chart-section {
    text-align: center;
    margin-top: 40px;
    padding: 20px;
    background-color: #fff;
    border-radius: 10px;
    border: 1px solid var(--border-color);
}

.risk-rank-text {
    text-align: center;
    font-size: 1.1em;
    color: var(--light-text-color);
    margin-bottom: 20px;
}

.plot-image { /* St.pyplot will render directly, but if embedding image, use this */
    width: 100%;
    max-width: 700px;
    height: auto;
    display: block;
    margin: 0 auto;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
}

/* Footer */
footer {
    text-align: center;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid var(--border-color);
    color: var(--light-text-color);
    font-size: 0.9em;
}

.footer-text {
    margin-bottom: 10px;
}

.footer-links a {
    color: var(--secondary-color);
    text-decoration: none;
    margin: 0 10px;
}

.footer-links a:hover {
    text-decoration: underline;
}

.disclaimer {
    margin-top: 20px;
    font-size: 0.85em;
    color: #888;
    background-color: #f0f0f0;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
}

/* Responsive Design */
@media (max-width: 768px) {
    .container {
        margin: 20px;
        padding: 20px;
    }
    h1 {
        font-size: 2.5em;
    }
    h2 {
        font-size: 1.6em;
    }
    /* Streamlit's default responsive behavior is good for columns */
    .stButton > button {
        font-size: 1em;
        padding: 10px 20px;
    }
    .prediction-status {
        font-size: 1.5em;
    }
    .probability {
        font-size: 1.2em;
    }
    .plot-image {
        max-width: 95%;
    }
}
</style>
"""

# --- Resource Loading Function ---
# Use st.cache_resource to load model and data only once across Streamlit reruns
@st.cache_resource
def load_resources():
    logging.info("Attempting to load model and dataset...")

    model_pipeline = None
    X_original_cols = None
    X_train_for_distribution = None
    
    # Load Model
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        logging.info(f"Model '{MODEL_PATH}' loaded successfully!")
    except FileNotFoundError:
        logging.error(f"Error: Model file '{MODEL_PATH}' not found.")
        st.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure 'optimized_lightgbm_model.pkl' is in the app's directory.")
        return None, None, None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading the model: {e}")
        st.error(f"An unexpected error occurred while loading the model: {e}")
        return None, None, None

    # Load Dataset for Column Order and Distribution Plot
    df = None
    try:
        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Dataset '{DATASET_PATH}' loaded successfully!")
    except FileNotFoundError:
        logging.warning(f"Warning: Dataset '{DATASET_PATH}' not found. Generating synthetic data for demonstration.")
        np.random.seed(42)
        n_samples = 100000
        df = pd.DataFrame({
            'gender': np.random.choice(['Female', 'Male', 'Other'], n_samples, p=[0.49, 0.49, 0.02]),
            'age': np.random.randint(1, 90, n_samples),
            'hypertension': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'heart_disease': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'smoking_history': np.random.choice(['never', 'No Info', 'former', 'current', 'ever', 'not current'], n_samples),
            'bmi': np.random.uniform(15.0, 50.0, n_samples),
            'HbA1c_level': np.random.uniform(4.0, 10.0, n_samples),
            'blood_glucose_level': np.random.randint(70, 300, n_samples),
            'diabetes': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        })
        for col in ['bmi', 'HbA1c_level', 'blood_glucose_level', 'smoking_history']:
            missing_indices = np.random.choice(df.index, size=int(len(df) * 0.005), replace=False)
            df.loc[missing_indices, col] = np.nan
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading or generating data: {e}")
        st.error(f"An error occurred with the dataset: {e}")
        return model_pipeline, X_original_cols, X_train_for_distribution

    if df is not None:
        if 'diabetes' in df.columns:
            df.dropna(subset=["diabetes"], inplace=True)
            X = df.drop(columns="diabetes")
            y = df["diabetes"]

            X_original_cols = X.columns.tolist()
            logging.debug(f"X_original_cols (from loaded CSV) = {X_original_cols}")

            # Use a smaller subset for plotting population distribution for faster rendering
            # This balances performance and representative distribution
            _, X_train_for_distribution, _, _ = train_test_split(
                X, y, test_size=0.8, stratify=y, random_state=42
            )
            logging.debug(f"X_train_for_distribution columns = {X_train_for_distribution.columns.tolist()}")
            logging.info(f"Loaded {len(X_train_for_distribution)} samples for distribution plot.")
        else:
            logging.error("The loaded dataset does not contain a 'diabetes' column.")
            st.error("Dataset missing 'diabetes' column. Distribution plot unavailable.")
            X_original_cols = df.columns.tolist()
            X_train_for_distribution = None
    else:
        logging.critical("No dataset loaded or generated. Application will not function correctly.")
        st.error("No dataset loaded or generated. Please check files and server logs.")

    return model_pipeline, X_original_cols, X_train_for_distribution

# --- Seaborn Plot Generation Function ---
# Returns a matplotlib figure directly for st.pyplot()
def create_seaborn_probability_distribution_plot(all_probabilities, new_patient_prob):
    """
    Generates a Seaborn histogram + KDE overlay + vertical line for patient probability.
    Returns a matplotlib Figure object.
    """
    if not isinstance(all_probabilities, np.ndarray) or all_probabilities.size < 2:
        logging.error("Insufficient data for Seaborn plot. Need at least 2 valid probabilities.")
        return None

    try:
        clean_probabilities = all_probabilities[np.isfinite(all_probabilities)]

        if clean_probabilities.size < 2:
            logging.error("Insufficient finite data for Seaborn plot after cleaning.")
            return None

        percentile_rank = np.mean(clean_probabilities <= new_patient_prob) * 100

        # Dynamic X-axis Zoom
        upper_bound_percentile = np.percentile(clean_probabilities, 99.5)
        x_max = max(new_patient_prob * 1.2, upper_bound_percentile, 0.1)
        x_max = min(x_max, 1.0)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(clean_probabilities, kde=True, stat='density',
                     color='#3F51B5', # Using secondary-color from CSS
                     edgecolor='white', linewidth=0.5, bins=50,
                     label='Population Distribution', ax=ax)

        ax.axvline(new_patient_prob, color='#FF5722', linestyle='--', linewidth=3,
                    label=f'Your Probability ({new_patient_prob:.2f})')

        current_ylim = ax.get_ylim()
        annotation_y_pos = current_ylim[1] * 0.95

        ax.text(new_patient_prob, annotation_y_pos,
                 f'{percentile_rank:.1f}% Rank',
                 horizontalalignment='center', color='white',
                 bbox=dict(facecolor='#FF5722', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.5'),
                 fontsize=12, fontweight='bold')

        ax.set_title('Diabetes Risk Probability Distribution\nCompare Your Risk to the Population', fontsize=16)
        ax.set_xlabel('Predicted Probability of Diabetes (0 to 1)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        
        ax.set_xlim(0, x_max) 
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend()
        plt.tight_layout()

        return fig # Return the figure object

    except Exception as e:
        logging.error(f"Error generating Seaborn plot: {e}")
        return None

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="SUGAR AI: Diabetes Predictor")

# Inject custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Main content container to apply Flask's .container styling
st.markdown('<div class="container">', unsafe_allow_html=True)

# Header
st.markdown('<header>', unsafe_allow_html=True)
try:
    st.image(LOGO_PATH, caption="SUGAR AI Logo", use_column_width=False, output_format='auto', width=250)
    st.markdown(f'<style>img[src*="{LOGO_PATH}"] {{ margin: 0 auto; display: block; padding: 10px; }}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning(f"Logo file '{LOGO_PATH}' not found. Please ensure it's in the app's directory.")
st.markdown('<h1 style="text-align: center;">SUGAR AI: Diabetes Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Your Personal Diabetes Risk Assessment Tool</p>', unsafe_allow_html=True)
st.markdown('</header>', unsafe_allow_html=True)

# Intro section
st.markdown('<section class="intro">', unsafe_allow_html=True)
st.write("""
    Provide the patient's health details below, and SUGAR AI will estimate their risk of diabetes,
    along with a comparison to a broader population.
""")
st.markdown('</section>', unsafe_allow_html=True)

# Load resources
model_pipeline, X_original_cols, X_train_for_distribution = load_resources()

# Check if resources loaded successfully
if model_pipeline is None or X_original_cols is None or X_train_for_distribution is None:
    st.warning("Application could not load all necessary resources. Please check error messages above.")
    st.stop() # Stop the app if critical resources are missing

# Input Form Section
st.markdown('<section class="input-form-section">', unsafe_allow_html=True)
st.markdown('<h2><span class="icon">👤</span> Patient Information</h2>', unsafe_allow_html=True)

with st.form("prediction_form", clear_on_submit=False):
    col1, col2 = st.columns(2) # Replicate the two-column form layout

    with col1:
        st.markdown('<h3>Demographics & Basic Health</h3>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ('Female', 'Male', 'Other'), key='gender_input')
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1, key='age_input')
        hypertension = st.checkbox("Hypertension (High Blood Pressure)", value=False, key='hypertension_input')
        heart_disease = st.checkbox("Heart Disease", value=False, key='heart_disease_input')

    with col2:
        st.markdown('<h3>Lifestyle & Lab Results</h3>', unsafe_allow_html=True)
        smoking_history = st.selectbox("Smoking History", ('never', 'No Info', 'former', 'current', 'ever', 'not current'), key='smoking_input')
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, format="%.1f", key='bmi_input')
        hba1c_level = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=5.5, format="%.1f", key='hba1c_input')
        blood_glucose_level = st.number_input("Blood Glucose Level (mg/dL)", min_value=50, max_value=300, value=100, step=1, key='bgl_input')

    st.markdown('</div>', unsafe_allow_html=True) # Close form-columns div
    submitted = st.form_submit_button("✨ Predict Diabetes Risk")

st.markdown('</section>', unsafe_allow_html=True) # Close input-form-section div


# --- Prediction Logic and Results Display ---
if submitted:
    st.markdown('<section class="results-section">', unsafe_allow_html=True)
    st.markdown('<h2><span class="icon">💡</span> Prediction Results</h2>', unsafe_allow_html=True)
    
    with st.spinner("Calculating prediction..."):
        try:
            processed_data = {
                'gender': gender,
                'age': age,
                'hypertension': 1 if hypertension else 0, # Convert bool to int
                'heart_disease': 1 if heart_disease else 0, # Convert bool to int
                'smoking_history': smoking_history,
                'bmi': bmi,
                'HbA1c_level': hba1c_level,
                'blood_glucose_level': blood_glucose_level
            }

            input_df = pd.DataFrame([processed_data])
            input_df = input_df[X_original_cols] # Ensure column order

            predicted_class = model_pipeline.predict(input_df)[0]
            patient_probability = model_pipeline.predict_proba(input_df)[:, 1][0]

            prediction_result_text = 'Diabetes Risk Detected' if predicted_class == 1 else 'No Diabetes Risk Detected'
            
            # Result Box styling
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            if predicted_class == 1:
                st.markdown(f'<p class="prediction-status diabetes-risk">{prediction_result_text}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="prediction-status no-diabetes-risk">{prediction_result_text}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="probability">Probability of Diabetes: <strong>{patient_probability:.2f}</strong></p>', unsafe_allow_html=True)

            # Risk Interpretation
            st.markdown('<div class="risk-interpretation">', unsafe_allow_html=True)
            if patient_probability >= 0.7:
                st.markdown("""
                    <div class="alert danger-alert">
                        <strong>❗ High Risk:</strong> It is highly recommended to consult a medical professional for further evaluation and management. This indicates a significant likelihood of diabetes.
                    </div>
                """, unsafe_allow_html=True)
            elif patient_probability >= 0.4:
                st.markdown("""
                    <div class="alert warning-alert">
                        <strong>🔶 Moderate Risk:</strong> Consider lifestyle changes, regular health check-ups, and discussions with a doctor. Early intervention can make a difference.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="alert success-alert">
                        <strong>🟢 Low Risk:</strong> Maintain a healthy lifestyle, including balanced diet and regular exercise. Regular check-ups are always advisable for preventive health.
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True) # Close risk-interpretation
            st.markdown('</div>', unsafe_allow_html=True) # Close result-box

            # Plotly Chart Section
            st.markdown('<div class="plotly-chart-section">', unsafe_allow_html=True)
            st.markdown('<h2><span class="icon">📊</span> Your Risk vs. Population Distribution</h2>', unsafe_allow_html=True)

            # Only attempt plot if distribution data is available
            if not X_train_for_distribution.empty:
                population_probabilities = model_pipeline.predict_proba(X_train_for_distribution)[:, 1]
                percentile_rank_val = np.mean(population_probabilities <= patient_probability) * 100
                
                st.markdown(f"""
                    <p class="risk-rank-text">
                        Your predicted diabetes probability of <strong class="highlight-prob">{patient_probability:.2f}</strong> means your risk is higher than
                        <strong class="highlight-rank">{percentile_rank_val:.1f}%</strong> of individuals in our historical data.
                    </p>
                """, unsafe_allow_html=True)

                seaborn_figure = create_seaborn_probability_distribution_plot(population_probabilities, patient_probability)
                if seaborn_figure:
                    st.pyplot(seaborn_figure) # Display the Matplotlib figure
                else:
                    st.warning("Could not generate population distribution plot due to insufficient data or an error.")
            else:
                st.warning("Population distribution data is not available to generate the plot.")

            st.markdown('</div>', unsafe_allow_html=True) # Close plotly-chart-section

        except Exception as e:
            st.markdown(f'<div class="alert error-alert"><strong>Error:</strong> An unexpected error occurred during prediction: {e}</div>', unsafe_allow_html=True)
            logging.error(f"Prediction error: {e}")

    st.markdown('</section>', unsafe_allow_html=True) # Close results-section

# --- Footer ---
st.markdown('<footer>', unsafe_allow_html=True)
st.markdown('<p class="footer-text">Powered by SUGAR AI | Created with ❤️ Streamlit & Seaborn</p>', unsafe_allow_html=True)
st.markdown("""
    <div class="footer-links">
        <a href="https://github.com/your-github-profile" target="_blank">GitHub</a> |
        <a href="https://linkedin.com/in/your-linkedin-profile" target="_blank">LinkedIn</a>
    </div>
""", unsafe_allow_html=True)
st.markdown("""
    <div class="disclaimer">
        <strong>Disclaimer:</strong> This application is for informational and educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a healthcare professional.
    </div>
""", unsafe_allow_html=True)
st.markdown('</footer>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) # Close outer .container div