import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Step 1: Set the folder path containing JSON files
folder_path = r'D:\OneDrive\Desktop\Masters\Machine Learning\ML_Project\json files'

# List all JSON files in the folder
json_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
print(f"Found {len(json_files)} JSON files.")

# Step 2: Load and Combine JSON Data
all_descriptions = []
all_problem_types = []

for json_file in json_files:
    with open(json_file, 'r') as file:
        data = pd.read_json(file, orient='records')

    # Extract relevant fields
    containers = data['containers']['cna']

    # Normalize descriptions
    descriptions = pd.json_normalize(containers['descriptions'], errors='ignore')
    all_descriptions.append(descriptions)

    # Normalize problem types
    problem_types = pd.json_normalize(containers['problemTypes'], record_path='descriptions', errors='ignore')
    all_problem_types.append(problem_types)

# Combine all descriptions and problem types
combined_descriptions = pd.concat(all_descriptions, ignore_index=True)
combined_problem_types = pd.concat(all_problem_types, ignore_index=True)

print("\nCombined Problem Types (CWE-IDs):")
print(combined_problem_types.head())
print("\nCombined Descriptions:")
print(combined_descriptions.head())

# Step 3: Align Features and Target Variables
if 'value' in combined_descriptions and 'cweId' in combined_problem_types:
    # Reset index to align data properly
    combined_descriptions = combined_descriptions.reset_index(drop=True)
    combined_problem_types = combined_problem_types.reset_index(drop=True)
    
    # Ensure alignment by dropping rows where either is missing
    aligned_data = pd.concat([combined_descriptions[['value']], combined_problem_types[['cweId']]], axis=1).dropna()
    
    # Update X and y
    X = aligned_data['value']
    y = aligned_data['cweId']
else:
    print("\nError: Missing required data for alignment.")
    exit()

# Encode the target variable (CWE-ID) using LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Vectorize textual features for machine learning models
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X).toarray()

# Ensure consistent length
print(f"Aligned Data - Features: {len(X_vec)}, Target: {len(y_encoded)}")

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y_encoded, test_size=0.3, random_state=42)

# Step 5: Random Forest Classifier Training
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Random Forest Evaluation
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(f"Precision: {precision_score(y_test, y_pred_rf, average='weighted')}")
print(f"Recall: {recall_score(y_test, y_pred_rf, average='weighted')}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf, average='weighted')}")

# Step 6: ANN Model
ann_model = models.Sequential()
ann_model.add(layers.Dense(128, input_dim=X_train.shape[1], activation='relu'))
ann_model.add(layers.Dense(64, activation='relu'))
ann_model.add(layers.Dense(32, activation='relu'))
ann_model.add(layers.Dense(len(le.classes_), activation='softmax'))

# Compile the ANN model
ann_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the ANN model
history_ann = ann_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Save the ANN model
ann_model.save('ann_model.h5')
print("ANN model saved as 'ann_model.h5'")


# ANN Model Evaluation
y_pred_ann = ann_model.predict(X_test).argmax(axis=1)
print("\nANN Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_ann)}")
print(f"Precision: {precision_score(y_test, y_pred_ann, average='weighted')}")
print(f"Recall: {recall_score(y_test, y_pred_ann, average='weighted')}")
print(f"F1 Score: {f1_score(y_test, y_pred_ann, average='weighted')}")

# Step 7: CNN Model
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

cnn_model = models.Sequential()
cnn_model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)))
cnn_model.add(layers.MaxPooling1D(pool_size=2))
cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(128, activation='relu'))
cnn_model.add(layers.Dense(len(le.classes_), activation='softmax'))

# Compile the CNN model
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
history_cnn = cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=64, validation_data=(X_test_cnn, y_test))

# Save the ANN model
cnn_model.save('cnn_model.h5')
print("CNN model saved as 'cnn_model.h5'")

# CNN Model Evaluation
y_pred_cnn = cnn_model.predict(X_test_cnn).argmax(axis=1)
print("\nCNN Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_cnn)}")
print(f"Precision: {precision_score(y_test, y_pred_cnn, average='weighted')}")
print(f"Recall: {recall_score(y_test, y_pred_cnn, average='weighted')}")
print(f"F1 Score: {f1_score(y_test, y_pred_cnn, average='weighted')}")

# Step 8: Results Visualization
plt.figure(figsize=(12, 6))

# ANN Accuracy
plt.plot(history_ann.history['accuracy'], label='ANN Training Accuracy')
plt.plot(history_ann.history['val_accuracy'], label='ANN Validation Accuracy')

# CNN Accuracy
plt.plot(history_cnn.history['accuracy'], label='CNN Training Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='CNN Validation Accuracy')

# Random Forest Accuracy
# Replicate the single accuracy value across the number of epochs for a flat line
rf_accuracy = accuracy_score(y_test, y_pred_rf)
epochs = range(1, len(history_ann.history['accuracy']) + 1)
plt.plot(epochs, [rf_accuracy] * len(epochs), label='Random Forest Accuracy', linestyle='--', color='green')

# Chart settings
plt.title('Training and Validation Accuracy (ANN, CNN, Random Forest)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Step 9: Model Inference
sample_description = ["Buffer overflow in XYZ application leading to code execution"]
sample_vec_ann = vectorizer.transform(sample_description).toarray()
sample_vec_cnn = sample_vec_ann.reshape(1, -1, 1)

# ANN Prediction
predicted_cwe_ann = le.inverse_transform(ann_model.predict(sample_vec_ann).argmax(axis=1))
print(f"\nPredicted CWE ID (ANN): {predicted_cwe_ann}")

# CNN Prediction
predicted_cwe_cnn = le.inverse_transform(cnn_model.predict(sample_vec_cnn).argmax(axis=1))
print(f"\nPredicted CWE ID (CNN): {predicted_cwe_cnn}")

# Step 18 Save the Model
# Save the trained Logistic Regression model
joblib.dump(rf_model, 'rf_model.pkl')

# Save the CountVectorizer
joblib.dump(vectorizer, 'count_vectorizer.pkl')

joblib.dump(le.classes_, 'label_encoder_classes.pkl')


print("Model and Vectorizer saved successfully!")