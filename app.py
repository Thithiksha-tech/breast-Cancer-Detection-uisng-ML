"""
Breast Cancer Detection - Streamlit Frontend Application
Modular web interface powered by Streamlit and connected to the pre-trained ML model.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from prediction.predictor import (
    ModelPredictor,
    FEATURE_NAMES,
    FEATURE_GROUPS,
    SAMPLE_CASES,
)

# 1. Page Configuration
st.set_page_config(
    page_title="Breast Cancer Detection | ML Risk Prediction",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom minimal styling to enhance readability without clutter
st.markdown(
    """
    <style>
    .metric-card-benign {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 8px;
        padding: 16px;
        margin-top: 12px;
        margin-bottom: 12px;
    }
    .metric-card-malignant {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 16px;
        margin-top: 12px;
        margin-bottom: 12px;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained machine learning model...")
def load_cached_predictor() -> ModelPredictor:
    """Loads and caches the existing pre-trained model and scaler."""
    return ModelPredictor()


def render_sidebar(model_info: Dict[str, Any]):
    """Renders the sidebar with project metadata, model details, and disclaimer."""
    with st.sidebar:
        st.header("🎗️ About the Project")
        st.write(
            "This diagnostic assistant utilizes a machine-learning model trained on the "
            "**Wisconsin Diagnostic Breast Cancer (WDBC)** dataset. It processes fine needle "
            "aspirate (FNA) digitized cell nuclei measurements to predict whether a mass is "
            "**Benign** or **Malignant**."
        )

        st.divider()

        st.subheader("🤖 Model Information")
        st.markdown(f"- **Algorithm:** `{model_info.get('model_type', 'Classifier')}`")
        st.markdown(f"- **Preprocessor:** `{model_info.get('scaler_type', 'StandardScaler')}`")
        st.markdown(f"- **Input Features:** `{model_info.get('num_features', 30)} quantitative metrics`")
        st.markdown(f"- **Target Classes:** `Benign (0)` / `Malignant (1)`")
        st.markdown(f"- **Probability Output:** `{'Supported' if model_info.get('supports_proba') else 'Not Supported'}`")

        st.divider()

        st.subheader("⚖️ Medical Disclaimer")
        st.warning(
            "This application is an educational and research machine-learning demonstration "
            "and is **not a certified medical diagnostic tool**. Predictions must not be used to "
            "make medical decisions. Please consult a qualified healthcare professional for "
            "clinical diagnosis."
        )


def main():
    # 2. Header Section
    st.title("Breast Cancer Detection")
    st.markdown("#### *Machine Learning Based Risk Prediction*")
    st.caption(
        "Enter cell nucleus features from fine needle aspirate (FNA) imaging or load a preset "
        "benchmark case to evaluate malignancy risk with the pre-trained classification model."
    )

    # 3. Model Loading & Verification
    try:
        predictor = load_cached_predictor()
        model_info = predictor.get_model_info()
    except FileNotFoundError as fnf_err:
        st.error(
            f"⚠️ **Model Artifact Missing:** {fnf_err}\n\n"
            "Please verify that `models/breast_cancer_model.pkl` and `models/scaler.pkl` exist."
        )
        return
    except Exception as err:
        st.error(f"⚠️ **Failed to load model:** {err}")
        return

    render_sidebar(model_info)

    st.write("")

    # 4. Preset Selector (Improves UX immensely - no need to type 30 numbers manually)
    preset_col1, preset_col2 = st.columns([2, 1])
    with preset_col1:
        st.subheader("1. Patient Clinical Metrics")
    with preset_col2:
        selected_preset = st.selectbox(
            "📋 Quick Load Sample Case:",
            options=["Custom Input", *SAMPLE_CASES.keys()],
            help="Choose a pre-filled verified case from the Wisconsin test dataset or enter custom values."
        )

    # Populate default values based on preset
    preset_data = {}
    if selected_preset in SAMPLE_CASES:
        preset_data = SAMPLE_CASES[selected_preset]
        st.info(f"Loaded feature values from **{selected_preset}**.")

    # 5. Feature Input Form
    # Group inputs logically into tabs/expanders using st.columns
    user_inputs: Dict[str, float] = {}

    input_tabs = st.tabs([
        "📊 Mean Nucleus Measurements",
        "📐 Standard Error (SE)",
        "⚠️ Worst / Largest Measurements"
    ])

    for tab_idx, (group_name, features) in enumerate(FEATURE_GROUPS.items()):
        with input_tabs[tab_idx]:
            st.caption(f"Enter the {group_name.lower()} computed for the digitized cell nuclei.")
            # Organize into 2 clean columns per tab
            col_left, col_right = st.columns(2)

            for i, (key, display_label, desc, min_v, max_v, default_v) in enumerate(features):
                target_col = col_left if i % 2 == 0 else col_right
                current_default = float(preset_data.get(key, default_v))

                with target_col:
                    val = st.number_input(
                        label=f"{display_label}",
                        min_value=0.0,
                        max_value=float(max_v * 2.0),
                        value=current_default,
                        step=0.001 if max_v < 1.0 else 0.1,
                        format="%.4f" if max_v < 1.0 else "%.2f",
                        help=f"{desc} (Typical range: {min_v} - {max_v})",
                        key=f"input_{key}",
                    )
                    user_inputs[key] = float(val)

    st.write("")
    st.divider()

    # 6. Prediction Action Section
    action_col1, action_col2 = st.columns([1, 2])
    with action_col1:
        predict_button = st.button("🔬 Predict Diagnostic Result", type="primary", use_container_width=True)

    with action_col2:
        st.caption("Press to standardize input features with the pre-trained scaler and run model inference.")

    # 7. Prediction Execution and Result Display
    if predict_button:
        with st.spinner("Processing clinical inputs through trained ML pipeline..."):
            try:
                # Validation check
                is_valid, validation_err = predictor.validate_inputs(user_inputs)
                if not is_valid:
                    st.error(f"❌ Input Validation Error: {validation_err}")
                    return

                # Perform prediction
                result = predictor.predict(user_inputs)

            except Exception as e:
                st.error(f"❌ Prediction encountered an error: {str(e)}")
                return

        st.subheader("2. Diagnostic Prediction Result")

        is_malignant = result["prediction"] == "Malignant"

        # Prominent Result Cards
        if is_malignant:
            st.markdown(
                f"""
                <div class="metric-card-malignant">
                    <h2 style="color: #b91c1c; margin: 0;">🔴 Result: Malignant Tumor Detected</h2>
                    <p style="color: #7f1d1d; margin-top: 8px; margin-bottom: 0;">
                        The model identified cellular characteristics consistent with malignant breast carcinoma.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="metric-card-benign">
                    <h2 style="color: #047857; margin: 0;">🟢 Result: Benign (Non-Cancerous)</h2>
                    <p style="color: #065f46; margin-top: 8px; margin-bottom: 0;">
                        The model identified cellular characteristics consistent with benign, non-cancerous tissue.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Metric cards & probability metrics
        if result.get("probabilities"):
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric(
                    label="Prediction Outcome",
                    value=result["prediction"],
                    delta="High Risk" if is_malignant else "Low Risk",
                    delta_color="inverse" if is_malignant else "normal",
                )
            with m_col2:
                st.metric(
                    label="Malignant Probability",
                    value=f"{result['malignant_prob'] * 100:.2f}%",
                )
            with m_col3:
                st.metric(
                    label="Benign Probability",
                    value=f"{result['benign_prob'] * 100:.2f}%",
                )

            # Visual progress bar for probabilities
            st.write("##### Probability Distribution")
            p_col1, p_col2 = st.columns([1, 1])
            with p_col1:
                st.write(f"**Benign:** {result['benign_prob'] * 100:.1f}%")
                st.progress(float(result["benign_prob"]))
            with p_col2:
                st.write(f"**Malignant:** {result['malignant_prob'] * 100:.1f}%")
                st.progress(float(result["malignant_prob"]))

        # 8. Feature Contribution Analysis (Explainability)
        if result.get("top_risk_factors"):
            with st.expander("🔍 View Significant Feature Contributions (Model Explainability)", expanded=True):
                st.write(
                    "The table below shows the features with the strongest influence on this prediction, "
                    "computed from standardized feature deviations and trained model coefficients:"
                )
                contrib_rows = []
                for factor in result["top_risk_factors"]:
                    feat_key = factor["feature"]
                    # Get friendly name
                    friendly_name = feat_key.replace("_", " ").title()
                    direction = "Towards Malignant 🔴" if factor["pushes_malignant"] else "Towards Benign 🟢"
                    contrib_rows.append({
                        "Feature": friendly_name,
                        "Patient Value": f"{factor['raw_value']:.4f}",
                        "Influence Direction": direction,
                        "Relative Impact Score": f"{abs(factor['impact']):.3f}"
                    })

                df_contrib = pd.DataFrame(contrib_rows)
                st.dataframe(df_contrib, use_container_width=True, hide_index=True)

    # 9. Educational Medical Disclaimer Footer
    st.write("")
    st.divider()
    st.caption(
        "⚕️ **Disclaimer:** This application is an educational/research machine-learning demonstration "
        "and is not a medical diagnostic tool. Predictions should not be used to make medical decisions. "
        "Please consult a qualified healthcare professional for medical evaluation."
    )


if __name__ == "__main__":
    main()
