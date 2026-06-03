import streamlit as st
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load("models/breast_cancer_model.pkl")

# --------------------------------------------------
# LOAD SCALER
# --------------------------------------------------

scaler = joblib.load("models/scaler.pkl")

# --------------------------------------------------
# APPLICATION UI
# --------------------------------------------------

st.title("🩺 Breast Cancer Detection System")

st.markdown("---")

st.success("Application Started Successfully")

st.write("Model Loaded Successfully")

st.write("Scaler Loaded Successfully")

st.write("Ready for Predictions")


