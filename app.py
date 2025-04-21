import json
import os
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import models
import joblib

# File paths for the required models and vectorizers
rf_model_path = 'rf_model.pkl'
ann_model_path = 'ann_model.h5'
cnn_model_path = 'cnn_model.h5'
vectorizer_path = 'count_vectorizer.pkl'
label_encoder_path = 'label_encoder_classes.pkl'

# Function to load pre-trained models and handle missing files
def load_models_and_vectorizer():
    if not os.path.exists(rf_model_path):
        st.error(f"Random Forest model file '{rf_model_path}' is missing.")
        return None, None, None, None, None
    if not os.path.exists(ann_model_path):
        st.error(f"ANN model file '{ann_model_path}' is missing.")
        return None, None, None, None, None
    if not os.path.exists(cnn_model_path):
        st.error(f"CNN model file '{cnn_model_path}' is missing.")
        return None, None, None, None, None
    if not os.path.exists(vectorizer_path):
        st.error(f"Vectorizer file '{vectorizer_path}' is missing.")
        return None, None, None, None, None
    if not os.path.exists(label_encoder_path):
        st.error(f"Label encoder file '{label_encoder_path}' is missing.")
        return None, None, None, None, None

    # Load models and vectorizer
    rf_model = joblib.load(rf_model_path)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = joblib.load(label_encoder_path)
    ann_model = models.load_model(ann_model_path)
    cnn_model = models.load_model(cnn_model_path)

    return rf_model, ann_model, cnn_model, vectorizer, label_encoder

# Load the models and vectorizer
rf_model, ann_model, cnn_model, vectorizer, label_encoder = load_models_and_vectorizer()

# Function to process JSON files and extract descriptions and CWE IDs
def process_json_files(uploaded_files):
    all_descriptions = []
    all_problem_types = []

    for uploaded_file in uploaded_files:
        try:
            # Load the JSON file
            data = json.load(uploaded_file)
        except json.JSONDecodeError:
            st.error(f"Error decoding JSON in file: {uploaded_file.name}")
            continue

        # Navigate through the JSON structure
        containers = data.get('containers', {}).get('cna', {})

        # Extract descriptions
        descriptions = containers.get('descriptions', [])
        if isinstance(descriptions, list):
            descriptions_df = pd.json_normalize(descriptions, errors='ignore')
            all_descriptions.append(descriptions_df)

        # Extract problemTypes
        problem_types = containers.get('problemTypes', [])
        if isinstance(problem_types, list):
            for item in problem_types:
                if isinstance(item, dict):
                    pt_descriptions = item.get('descriptions', [])
                    if isinstance(pt_descriptions, list):
                        pt_df = pd.json_normalize(pt_descriptions, errors='ignore')
                        all_problem_types.append(pt_df)

    # Combine the descriptions and problem types into single DataFrames
    combined_descriptions = pd.concat(all_descriptions, ignore_index=True) if all_descriptions else pd.DataFrame()
    combined_problem_types = pd.concat(all_problem_types, ignore_index=True) if all_problem_types else pd.DataFrame()

    return combined_descriptions, combined_problem_types

# Function to map severity based on CWE ID
def map_severity(cwe_id):
    severity_mapping = {
        "CWE-79": "High",  # Example: Cross-Site Scripting
        "CWE-89": "High",  # Example: SQL Injection
        "CWE-20": "Medium",  # Example: Improper Input Validation
        "CWE-22": "Medium",  # Example: Path Traversal
        "CWE-200": "Low",  # Example: Information Exposure
        "CWE-284": "Low",  # Example: Improper Access Control
    }
    return severity_mapping.get(cwe_id, "Unknown")  # Default to "Unknown" for unmapped CWE IDs

# Streamlit UI
st.title('Vulnerability Prediction Application')
st.write("Upload JSON files with vulnerability descriptions to predict CWE IDs and severity levels.")

# File uploader widget
uploaded_files = st.file_uploader("Choose JSON files", type="json", accept_multiple_files=True)

if uploaded_files:
    if rf_model and ann_model and cnn_model and vectorizer and label_encoder:
        # Process the uploaded JSON files
        combined_descriptions, combined_problem_types = process_json_files(uploaded_files)

        if not combined_descriptions.empty and 'value' in combined_descriptions:
            if not combined_problem_types.empty and 'cweId' in combined_problem_types:
                # Align descriptions and problem types
                aligned_data = pd.concat(
                    [combined_descriptions[['value']], combined_problem_types[['cweId']]], 
                    axis=1
                ).dropna()

                # Prepare features (X) and target (y)
                X = aligned_data['value']
                y = aligned_data['cweId']

                # Step 2: Make predictions
                X_vec = vectorizer.transform(X)

                # Predict using Random Forest model
                predictions_rf = rf_model.predict(X_vec)
                predicted_cwe_rf = label_encoder.inverse_transform(predictions_rf)

                # Predict using ANN model
                predictions_ann = ann_model.predict(X_vec.toarray()).argmax(axis=1)
                predicted_cwe_ann = label_encoder.inverse_transform(predictions_ann)

                # Predict using CNN model
                predictions_cnn = cnn_model.predict(X_vec.toarray()).argmax(axis=1)
                predicted_cwe_cnn = label_encoder.inverse_transform(predictions_cnn)

                # Map severity levels
                severity_rf = [map_severity(cwe) for cwe in predicted_cwe_rf]
                severity_ann = [map_severity(cwe) for cwe in predicted_cwe_ann]
                severity_cnn = [map_severity(cwe) for cwe in predicted_cwe_cnn]

                # Calculate evaluation metrics
                metrics = {}
                for model_name, predictions in zip(
                    ['RF', 'ANN', 'CNN'], 
                    [predicted_cwe_rf, predicted_cwe_ann, predicted_cwe_cnn]
                ):
                    metrics[model_name] = {
                        'accuracy': accuracy_score(y, predictions),
                        'precision': precision_score(y, predictions, average="weighted", zero_division=0),
                        'recall': recall_score(y, predictions, average="weighted", zero_division=0),
                        'f1': f1_score(y, predictions, average="weighted", zero_division=0)
                    }

                # Display evaluation metrics
                for model_name, metric_values in metrics.items():
                    st.subheader(f"Evaluation Metrics ({model_name}):")
                    st.write(f"Accuracy: {metric_values['accuracy']:.2f}")
                    st.write(f"Precision: {metric_values['precision']:.2f}")
                    st.write(f"Recall: {metric_values['recall']:.2f}")
                    st.write(f"F1 Score: {metric_values['f1']:.2f}")

                # Create a DataFrame of results
                results_df = pd.DataFrame({
                    'Description': X,
                    'Actual CWE ID': y,
                    'Predicted CWE ID (RF)': predicted_cwe_rf,
                    'Severity (RF)': severity_rf,
                    'Predicted CWE ID (ANN)': predicted_cwe_ann,
                    'Severity (ANN)': severity_ann,
                    'Predicted CWE ID (CNN)': predicted_cwe_cnn,
                    'Severity (CNN)': severity_cnn
                })

                # Display predictions with severity levels
                st.write("Prediction Results with Severity Levels:")
                st.dataframe(results_df)

                # Allow users to download predictions as a CSV file
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Prediction Results",
                    data=csv,
                    file_name='vulnerability_predictions.csv',
                    mime='text/csv'
                )
            else:
                st.error("CWE ID data is missing in the uploaded files. Please check the input.")
        else:
            st.error("Description data is missing in the uploaded files. Please check the input.")
    else:
        st.error("Required models or files are missing. Please upload them and try again.")
