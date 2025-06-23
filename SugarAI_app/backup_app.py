# flask_app.py

import joblib
import logging
import json
import base64
import io

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from scipy.stats import gaussian_kde # Used for KDE calculation if needed, but Seaborn handles it
from sklearn.model_selection import train_test_split

# --- Seaborn/Matplotlib Imports for Plotting ---
import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- Global Variables for Model and Data (Loaded Once) ---
model_pipeline = None
X_original_cols = None
X_train_for_distribution = None

# --- Constants for File Paths ---
MODEL_PATH = 'optimized_lightgbm_model.pkl'
DATASET_PATH = 'diabetes_prediction_dataset.csv'

# --- Resource Loading Function ---
def load_resources():
    global model_pipeline, X_original_cols, X_train_for_distribution

    logging.info("Attempting to load model and dataset...")

    # Load Model
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        logging.info(f"Model '{MODEL_PATH}' loaded successfully!")
    except FileNotFoundError:
        logging.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it's in the same directory as flask_app.py.")
        model_pipeline = None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading the model: {e}")
        model_pipeline = None

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
        # Add some NaN values to the synthetic data for robustness testing
        for col in ['bmi', 'HbA1c_level', 'blood_glucose_level', 'smoking_history']:
            missing_indices = np.random.choice(df.index, size=int(len(df) * 0.005), replace=False)
            df.loc[missing_indices, col] = np.nan
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading or generating data: {e}")

    if df is not None:
        if 'diabetes' in df.columns:
            # Ensure no NaNs in target column before splitting
            df.dropna(subset=["diabetes"], inplace=True)
            X = df.drop(columns="diabetes")
            y = df["diabetes"]

            # Store original column names from the dataset for consistent input ordering
            X_original_cols = X.columns.tolist()
            logging.debug(f"X_original_cols (from loaded CSV) = {X_original_cols}")

            # Use a subset of the training data for plotting the distribution
            # This ensures the distribution reflects data seen by the model
            X_train_for_distribution, _, _, _ = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            logging.debug(f"X_train_for_distribution columns = {X_train_for_distribution.columns.tolist()}")
        else:
            logging.error("The loaded dataset does not contain a 'diabetes' column. Cannot prepare distribution data.")
            X_original_cols = df.columns.tolist() # Still get column names if possible
            X_train_for_distribution = None
    else:
        logging.critical("No dataset loaded or generated. Application will not function correctly.")

# Call resource loading when the app context is available (e.g., when app starts)
with app.app_context():
    load_resources()


# --- Seaborn Plot Generation Function ---
def create_seaborn_probability_distribution_plot(all_probabilities, new_patient_prob):
    """
    Generates a Seaborn histogram + KDE overlay + vertical line for patient probability.
    Returns the plot as a base64 encoded PNG image.
    """
    if not isinstance(all_probabilities, np.ndarray) or all_probabilities.size < 2:
        logging.error("Insufficient data for Seaborn plot. Need at least 2 valid probabilities.")
        return None

    try:
        # Filter out non-finite values (NaN, inf) which can cause issues with KDE
        clean_probabilities = all_probabilities[np.isfinite(all_probabilities)]

        if clean_probabilities.size < 2:
            logging.error("Insufficient finite data for Seaborn plot after cleaning.")
            return None

        percentile_rank = np.mean(clean_probabilities <= new_patient_prob) * 100

        # --- Dynamic X-axis Zoom ---
        # Determine appropriate x-axis limits to zoom into the relevant part of the distribution
        # This is crucial for making the histogram visible if probabilities are clustered
        # For 'no diabetes' cases, probabilities tend to be very low.
        
        # Calculate 99.5th percentile of probabilities (excluding patient's own prob for bound)
        upper_bound_percentile = np.percentile(clean_probabilities, 99.5)
        
        # Set x_max to at least the patient's probability + a buffer,
        # or the 99.5th percentile, whichever is larger, but not exceeding 1.
        x_max = max(new_patient_prob * 1.2, upper_bound_percentile, 0.1) # Minimum x_max of 0.1
        x_max = min(x_max, 1.0) # Ensure it doesn't go beyond 1.0

        # Create the plot
        plt.figure(figsize=(10, 6)) # Good default size for web display
        sns.histplot(clean_probabilities, kde=True, stat='density',
                     color='#3F51B5',  # Using secondary-color from CSS :root
                     edgecolor='white', linewidth=0.5, bins=50, # More bins for higher resolution
                     label='Population Distribution')

        # Add a vertical line for the new patient's probability
        plt.axvline(new_patient_prob, color='#FF5722', linestyle='--', linewidth=3,
                    label=f'Your Probability ({new_patient_prob:.2f})')

        # Add annotation for percentile rank
        # Use the current ylim to position the annotation
        current_ylim = plt.gca().get_ylim()
        annotation_y_pos = current_ylim[1] * 0.95 # Place at 95% of current max y

        plt.text(new_patient_prob, annotation_y_pos,
                 f'{percentile_rank:.1f}% Rank',
                 horizontalalignment='center', color='white',
                 bbox=dict(facecolor='#FF5722', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.5'),
                 fontsize=12, fontweight='bold')


        plt.title('Diabetes Risk Probability Distribution\nCompare Your Risk to the Population', fontsize=16)
        plt.xlabel('Predicted Probability of Diabetes (0 to 1)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        
        # Apply the dynamically calculated x-axis limits
        plt.xlim(0, x_max) 
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout() # Adjust layout to prevent labels from overlapping

        # Save plot to a BytesIO object
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close() # Close the plot to free up memory

        return f"data:image/png;base64,{plot_base64}"

    except Exception as e:
        logging.error(f"Error generating Seaborn plot: {e}")
        return None

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    probability = None
    seaborn_plot_url = None # Variable for the Seaborn plot
    percentile_rank_display = None
    error_message = None

    form_data = {
        'gender': 'Female',
        'age': 30,
        'hypertension': 'off',
        'heart_disease': 'off',
        'smoking_history': 'never',
        'bmi': 25.0,
        'hba1c_level': 5.5,
        'blood_glucose_level': 100
    }

    if request.method == 'POST':
        if model_pipeline is None or X_train_for_distribution is None or X_original_cols is None:
            error_message = "Application resources (model or data) are not fully loaded. Please check server logs for details."
            logging.error(error_message)
            return render_template('index.html', error=error_message, form_data=form_data)

        form_data.update({
            'gender': request.form.get('gender', form_data['gender']),
            'age': request.form.get('age', str(form_data['age'])),
            'hypertension': request.form.get('hypertension', form_data['hypertension']),
            'heart_disease': request.form.get('heart_disease', form_data['heart_disease']),
            'smoking_history': request.form.get('smoking_history', form_data['smoking_history']),
            'bmi': request.form.get('bmi', str(form_data['bmi'])),
            'hba1c_level': request.form.get('hba1c_level', str(form_data['hba1c_level'])),
            'blood_glucose_level': request.form.get('blood_glucose_level', str(form_data['blood_glucose_level']))
        })

        try:
            processed_data = {
                'gender': form_data['gender'],
                'age': int(form_data['age']),
                'hypertension': 1 if form_data['hypertension'] == 'on' else 0,
                'heart_disease': 1 if form_data['heart_disease'] == 'on' else 0,
                'smoking_history': form_data['smoking_history'],
                'bmi': float(form_data['bmi']),
                'HbA1c_level': float(form_data['hba1c_level']),
                'blood_glucose_level': int(form_data['blood_glucose_level'])
            }

            input_df = pd.DataFrame([processed_data])
            # Ensure the input DataFrame has columns in the same order as the model expects
            input_df = input_df[X_original_cols]
            logging.debug(f"input_df columns (after reorder) = {input_df.columns.tolist()}")

            predicted_class = model_pipeline.predict(input_df)[0]
            patient_probability = model_pipeline.predict_proba(input_df)[:, 1][0]

            prediction_result = 'Diabetes Risk Detected' if predicted_class == 1 else 'No Diabetes Risk Detected'
            probability = f"{patient_probability:.2f}"

            # Predict probabilities for the training data for the distribution plot
            population_probabilities = model_pipeline.predict_proba(X_train_for_distribution)[:, 1]
            logging.debug(f"all_probabilities_for_plot stats: min={np.min(population_probabilities):.4f}, max={np.max(population_probabilities):.4f}, mean={np.mean(population_probabilities):.4f}, len={len(population_probabilities)}")

            percentile_rank_val = np.mean(population_probabilities <= patient_probability) * 100
            percentile_rank_display = f"{percentile_rank_val:.1f}"

            # Generate Seaborn plot URL
            seaborn_plot_url = create_seaborn_probability_distribution_plot(population_probabilities, patient_probability)
            logging.debug(f"Seaborn plot URL generated. Is None? {seaborn_plot_url is None}")
            if seaborn_plot_url:
                logging.debug(f"Seaborn plot URL length: {len(seaborn_plot_url)}")
                logging.debug(f"Seaborn plot URL snippet (first 100 chars): {seaborn_plot_url[:100]}...")
            else:
                logging.warning("Seaborn plot URL is None or empty, plot will not be rendered.")


        except ValueError as ve:
            error_message = f"Invalid input. Please ensure all numerical fields are filled correctly: {ve}"
            logging.error(f"ValueError during prediction: {ve}")
        except KeyError as ke:
            error_message = f"Missing form data or column mismatch: {ke}. Please ensure all fields are present."
            logging.error(f"KeyError during prediction: {ke}")
        except Exception as e:
            error_message = f"An unexpected error occurred during prediction: {e}"
            logging.error(f"General Exception during prediction: {e}")

    render_form_data = form_data.copy()

    return render_template('index.html',
                           prediction_result=prediction_result,
                           probability=probability,
                           seaborn_plot_url=seaborn_plot_url, # Pass seaborn_plot_url
                           percentile_rank_display=percentile_rank_display,
                           form_data=render_form_data,
                           error=error_message)

if __name__ == '__main__':
    app.run(debug=True)