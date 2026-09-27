# Breast Cancer Detection using Machine Learning

A clean, modular Streamlit frontend for the pre-trained Breast Cancer Detection Machine Learning classification model.

This application provides an interactive clinical prediction interface connected to an existing trained model and preprocessor (trained on the Wisconsin Diagnostic Breast Cancer dataset).

---

## 📁 Project Architecture

```text
├── models/                         # Pre-trained ML artifacts
│   ├── breast_cancer_model.pkl    # Trained classification model (Logistic Regression)
│   └── scaler.pkl                 # Pre-trained StandardScaler (30 features)
├── prediction/                     # Dedicated ML inference & validation module
│   ├── __init__.py
│   └── predictor.py               # Model loading, validation, scaling, prediction & explainability
├── data/                           # Dataset files
│   └── data.csv                   # Wisconsin Breast Cancer Dataset (WDBC)
├── notebooks/                      # Exploratory data analysis & model development
│   └── Breast_Cancer_Detection (1).ipynb
├── app.py                          # Streamlit web application & user interface
├── requirements.txt                # Production Python dependencies
└── README.md                       # Project documentation
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit App
```bash
streamlit run app.py
```

The application will start and open automatically in your browser (typically at `http://localhost:8501` or port 3000 in cloud environments).

---

## 🔬 How Prediction Works

1. **User Input Collection**: The user enters 30 cellular nucleus metrics (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension across Mean, Standard Error, and Worst) or loads a verified sample case.
2. **Validation**: All numerical inputs are validated to ensure required values are present, finite, and non-negative.
3. **Preprocessing**: Inputs are ordered into the exact 30-feature sequence expected by the training pipeline and transformed using the pre-trained `scaler.pkl` (`StandardScaler`).
4. **Model Inference**: The scaled vector is passed to `breast_cancer_model.pkl` to compute the classification decision (`Benign` vs `Malignant`).
5. **Probability & Explainability**: `predict_proba()` calculates malignant and benign confidence percentages, and feature contribution scoring breaks down the most influential metrics for the diagnosis.

---

## ⚖️ Medical Disclaimer

This application is an educational and research machine-learning demonstration and is **not a certified medical diagnostic tool**. Predictions must not be used to make medical decisions. Always consult a qualified healthcare professional for medical evaluation and clinical diagnosis.
