<h1 align="center">🧠 ML Classification of Vulnerabilities (CVE) using AI</h1>

<p align="center">
  <em>Empowering cybersecurity through intelligent vulnerability classification using ML models like ANN, CNN, and Random Forest.</em>
</p>

---

## 🔍 Overview

This project focuses on classifying cybersecurity vulnerabilities using machine learning. It leverages **Common Vulnerabilities and Exposures (CVE)** JSON data and applies advanced NLP techniques to train and evaluate models including:

- 🧠 Artificial Neural Networks (ANN)
- 🔎 Convolutional Neural Networks (CNN)
- 🌲 Random Forest (RF)

A **Flask web interface** provides real-time classification, making this project both functional and interactive.

---

## 🗂️ Project Structure

```bash
ML_Project/
│
├── ML_Project.py                # Main ML pipeline for training and evaluation
├── app.py                       # Flask web app for real-time predictions
├── ann_model.h5                 # Saved ANN model
├── cnn_model.h5                 # Saved CNN model
├── rf_model.pkl                 # Saved Random Forest model
├── count_vectorizer.pkl         # Text feature vectorizer
├── label_encoder_classes.pkl    # Label encoder classes
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview and setup
└── json files/                  # Sample CVE JSON input files
    ├── CVE-2024-9021.json
    └── CVE-2024-9022.json
```
---

## 🚀 Key Features

*🔐 CVE-Based Vulnerability Classification

Automatically classifies real security vulnerabilities using CVE data.

*📝 NLP-Powered Text Vectorization

Utilizes CountVectorizer to transform vulnerability descriptions into numerical features suitable for ML models.

*🧠 Model Support

Implements and compares multiple ML models:

-Artificial Neural Networks (ANN)

-Convolutional Neural Networks (CNN)

-Random Forest (RF)

*🌐 Flask Web Interface

Provides an interactive platform where users can upload CVE JSON files and receive predictions in real-time.

*📂 Input Format

Accepts CVE data in the JSON format. Example files are provided in the json_files/ directory.

---
## 💻 Installation

🔹 Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ML-Classification-of-Vulnerabilities.git
cd ML-Classification-of-Vulnerabilities
```
🔹 Install Dependencies
```bash
pip install -r requirements.txt
```
---
## ⚙️ Usage
🔁 Retrain Models (if needed)
```bash
python ML_Project.py
```
🌐 Run Flask Web App
```bash
python app.py
```
🔗 Then visit: http://localhost:5000

---
## 📥 Input Format

The application accepts CVE JSON files structured with vulnerability descriptions.
Examples are available in the json files/ folder.

---
## 🤖 Models Implemented

| Model Type     | Library Used     | Description                              |
|----------------|------------------|------------------------------------------|
| Random Forest  | scikit-learn     | Tree-based classifier                     |
| ANN            | TensorFlow/Keras | Deep learning classifier                  |
| CNN            | TensorFlow/Keras | Text-based convolutional classification   |

---
## 📊 Results & Analysis

Detailed results, charts, and analysis are available in the presentation:

📎 COSC5406002 - Machine Learning Classification of Vulnerabilities (CVE) Using AI_Machine Learning.pptx

---
## 📦 Requirements

The following Python libraries are required for this project:

- **flask**
- **numpy**
- **pandas**
- **matplotlib**
- **scikit-learn**
- **tensorflow**
- **joblib**
- **streamlit**

---
## 👩‍🎓 Author

### Sweta Patel, Nisha Raval

MSc Computer Science
Algoma University

---
## 📝 License

This project is intended for educational and academic purposes only.
