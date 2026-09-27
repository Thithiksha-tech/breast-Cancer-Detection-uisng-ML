"""
Breast Cancer Detection - Machine Learning Prediction Module
Connects to the existing trained model and scaler to perform inference.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

# The exact 30 features in the exact order expected by the trained model
FEATURE_NAMES: List[str] = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

# Feature metadata for display and validation
FEATURE_GROUPS = {
    "Mean Values": [
        ("radius_mean", "Mean Radius", "Mean of distances from center to points on the perimeter", 6.0, 30.0, 14.12),
        ("texture_mean", "Mean Texture", "Standard deviation of gray-scale values", 9.0, 40.0, 19.28),
        ("perimeter_mean", "Mean Perimeter", "Mean size of the core tumor perimeter", 40.0, 200.0, 91.96),
        ("area_mean", "Mean Area", "Mean area of the tumor mass", 140.0, 2600.0, 654.88),
        ("smoothness_mean", "Mean Smoothness", "Local variation in radius lengths", 0.05, 0.20, 0.096),
        ("compactness_mean", "Mean Compactness", "Perimeter^2 / area - 1.0", 0.01, 0.35, 0.104),
        ("concavity_mean", "Mean Concavity", "Severity of concave portions of the contour", 0.0, 0.45, 0.088),
        ("concave points_mean", "Mean Concave Points", "Number of concave portions of the contour", 0.0, 0.25, 0.048),
        ("symmetry_mean", "Mean Symmetry", "Symmetry of the cell nuclei", 0.10, 0.35, 0.181),
        ("fractal_dimension_mean", "Mean Fractal Dim", "Coastline approximation - 1", 0.04, 0.10, 0.062),
    ],
    "Standard Error (SE)": [
        ("radius_se", "Radius SE", "Standard error for the mean of distances from center to perimeter", 0.1, 3.0, 0.405),
        ("texture_se", "Texture SE", "Standard error for standard deviation of gray-scale values", 0.3, 5.0, 1.216),
        ("perimeter_se", "Perimeter SE", "Standard error for perimeter measurement", 0.7, 25.0, 2.866),
        ("area_se", "Area SE", "Standard error for tumor area measurement", 6.0, 550.0, 40.33),
        ("smoothness_se", "Smoothness SE", "Standard error for smoothness measurement", 0.001, 0.04, 0.007),
        ("compactness_se", "Compactness SE", "Standard error for compactness measurement", 0.002, 0.15, 0.025),
        ("concavity_se", "Concavity SE", "Standard error for concavity measurement", 0.0, 0.40, 0.031),
        ("concave points_se", "Concave Points SE", "Standard error for concave points measurement", 0.0, 0.06, 0.011),
        ("symmetry_se", "Symmetry SE", "Standard error for symmetry measurement", 0.007, 0.08, 0.020),
        ("fractal_dimension_se", "Fractal Dim SE", "Standard error for fractal dimension measurement", 0.0008, 0.03, 0.003),
    ],
    "Worst / Largest Values": [
        ("radius_worst", "Worst Radius", "Largest/worst mean value for radius", 7.0, 40.0, 16.26),
        ("texture_worst", "Worst Texture", "Largest/worst mean value for texture", 12.0, 50.0, 25.67),
        ("perimeter_worst", "Worst Perimeter", "Largest/worst mean value for perimeter", 50.0, 260.0, 107.26),
        ("area_worst", "Worst Area", "Largest/worst mean value for area", 180.0, 4300.0, 880.58),
        ("smoothness_worst", "Worst Smoothness", "Largest/worst mean value for smoothness", 0.07, 0.25, 0.132),
        ("compactness_worst", "Worst Compactness", "Largest/worst mean value for compactness", 0.02, 1.10, 0.254),
        ("concavity_worst", "Worst Concavity", "Largest/worst mean value for concavity", 0.0, 1.30, 0.272),
        ("concave points_worst", "Worst Concave Points", "Largest/worst mean value for concave points", 0.0, 0.30, 0.114),
        ("symmetry_worst", "Worst Symmetry", "Largest/worst mean value for symmetry", 0.15, 0.70, 0.290),
        ("fractal_dimension_worst", "Worst Fractal Dim", "Largest/worst mean value for fractal dimension", 0.05, 0.22, 0.083),
    ],
}

# Example benchmark cases from the dataset for quick testing
SAMPLE_CASES = {
    "Benign Sample (Healthy / Non-Cancerous Case)": {
        "radius_mean": 13.54, "texture_mean": 14.36, "perimeter_mean": 87.46, "area_mean": 566.3,
        "smoothness_mean": 0.09779, "compactness_mean": 0.08129, "concavity_mean": 0.06664, "concave points_mean": 0.04781,
        "symmetry_mean": 0.1885, "fractal_dimension_mean": 0.05766,
        "radius_se": 0.2699, "texture_se": 0.7886, "perimeter_se": 2.058, "area_se": 23.56,
        "smoothness_se": 0.008462, "compactness_se": 0.0146, "concavity_se": 0.02387, "concave points_se": 0.01315,
        "symmetry_se": 0.0198, "fractal_dimension_se": 0.0023,
        "radius_worst": 15.11, "texture_worst": 19.26, "perimeter_worst": 99.7, "area_worst": 711.2,
        "smoothness_worst": 0.144, "compactness_worst": 0.1773, "concavity_worst": 0.239, "concave points_worst": 0.1288,
        "symmetry_worst": 0.2977, "fractal_dimension_worst": 0.07259
    },
    "Malignant Sample (Patient #10 from Notebook)": {
        "radius_mean": 16.02, "texture_mean": 23.24, "perimeter_mean": 102.7, "area_mean": 797.8,
        "smoothness_mean": 0.08206, "compactness_mean": 0.06669, "concavity_mean": 0.03299, "concave points_mean": 0.03323,
        "symmetry_mean": 0.1528, "fractal_dimension_mean": 0.05697,
        "radius_se": 0.3795, "texture_se": 1.187, "perimeter_se": 2.466, "area_se": 40.51,
        "smoothness_se": 0.004029, "compactness_se": 0.009269, "concavity_se": 0.01101, "concave points_se": 0.007591,
        "symmetry_se": 0.0146, "fractal_dimension_se": 0.003042,
        "radius_worst": 19.19, "texture_worst": 33.88, "perimeter_worst": 123.8, "area_worst": 1150.0,
        "smoothness_worst": 0.1181, "compactness_worst": 0.1551, "concavity_worst": 0.1459, "concave points_worst": 0.09975,
        "symmetry_worst": 0.2948, "fractal_dimension_worst": 0.08452
    },
    "High-Risk Malignant Sample": {
        "radius_mean": 20.57, "texture_mean": 17.77, "perimeter_mean": 132.9, "area_mean": 1326.0,
        "smoothness_mean": 0.08474, "compactness_mean": 0.07864, "concavity_mean": 0.0869, "concave points_mean": 0.07017,
        "symmetry_mean": 0.1812, "fractal_dimension_mean": 0.05667,
        "radius_se": 0.5435, "texture_se": 0.7339, "perimeter_se": 3.398, "area_se": 74.08,
        "smoothness_se": 0.005225, "compactness_se": 0.01308, "concavity_se": 0.0186, "concave points_se": 0.0134,
        "symmetry_se": 0.01389, "fractal_dimension_se": 0.003532,
        "radius_worst": 24.99, "texture_worst": 23.41, "perimeter_worst": 158.8, "area_worst": 1956.0,
        "smoothness_worst": 0.1238, "compactness_worst": 0.1866, "concavity_worst": 0.2416, "concave points_worst": 0.186,
        "symmetry_worst": 0.275, "fractal_dimension_worst": 0.08902
    }
}


class ModelPredictor:
    """Manages loading the pre-trained machine learning model and scaler."""

    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        self._load_artifacts()

    def _locate_file(self, possible_paths: List[str]) -> Path:
        for p in possible_paths:
            path_obj = Path(p)
            if path_obj.exists() and path_obj.is_file():
                return path_obj
        raise FileNotFoundError(f"Could not find required artifact in: {possible_paths}")

    def _load_artifacts(self) -> None:
        """Finds and loads the existing trained model and scaler files."""
        model_candidates = [
            self.model_path or "",
            "models/breast_cancer_model.pkl",
            "model/trained_model.pkl",
            "model/breast_cancer_model.pkl",
            "breast_cancer_model.pkl",
        ]
        scaler_candidates = [
            self.scaler_path or "",
            "models/scaler.pkl",
            "model/scaler.pkl",
            "scaler.pkl",
        ]

        resolved_model_path = self._locate_file([p for p in model_candidates if p])
        resolved_scaler_path = self._locate_file([p for p in scaler_candidates if p])

        self.model = joblib.load(str(resolved_model_path))
        self.scaler = joblib.load(str(resolved_scaler_path))
        self.resolved_model_path = str(resolved_model_path)
        self.resolved_scaler_path = str(resolved_scaler_path)

    def get_model_info(self) -> Dict[str, Any]:
        """Returns metadata about the loaded model."""
        info = {
            "model_type": type(self.model).__name__,
            "scaler_type": type(self.scaler).__name__,
            "model_path": self.resolved_model_path,
            "scaler_path": self.resolved_scaler_path,
            "num_features": len(FEATURE_NAMES),
            "supports_proba": hasattr(self.model, "predict_proba"),
            "classes": getattr(self.model, "classes_", ["Benign (0)", "Malignant (1)"]).tolist()
            if hasattr(self.model, "classes_") else [0, 1]
        }
        return info

    def validate_inputs(self, raw_inputs: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validates user input against expected feature names and non-negative constraints.
        """
        for feature in FEATURE_NAMES:
            if feature not in raw_inputs:
                return False, f"Missing required feature: '{feature}'"
            val = raw_inputs[feature]
            if val is None or not isinstance(val, (int, float, np.number)):
                return False, f"Feature '{feature}' must be a valid number, got: {val}"
            if np.isnan(val) or np.isinf(val):
                return False, f"Feature '{feature}' cannot be NaN or Infinite."
            if val < 0:
                return False, f"Feature '{feature}' cannot be negative in biological measurements."
        return True, None

    def prepare_input(self, raw_inputs: Dict[str, Any]) -> pd.DataFrame:
        """
        Formats the inputs into a single-row DataFrame with the exact columns and order.
        """
        ordered_data = {feature: [float(raw_inputs[feature])] for feature in FEATURE_NAMES}
        return pd.DataFrame(ordered_data)

    def predict(self, raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the full inference pipeline:
        1. Validates inputs
        2. Formats to exact feature order
        3. Scales with pre-trained StandardScaler
        4. Predicts with pre-trained model
        5. Computes probabilities and feature importances
        """
        is_valid, error_msg = self.validate_inputs(raw_inputs)
        if not is_valid:
            raise ValueError(error_msg)

        input_df = self.prepare_input(raw_inputs)

        # Preprocessing: Scale using the existing scaler
        scaled_features = self.scaler.transform(input_df)

        # Prediction
        prediction_raw = self.model.predict(scaled_features)[0]

        # Determine label (in WDBC: 1 = Malignant, 0 = Benign or 'M'/'B')
        if prediction_raw in [1, "1", "M", "Malignant"]:
            label = "Malignant"
            code = 1
        else:
            label = "Benign"
            code = 0

        result: Dict[str, Any] = {
            "prediction": label,
            "prediction_code": code,
            "raw_output": prediction_raw,
            "probabilities": None,
            "benign_prob": None,
            "malignant_prob": None,
            "confidence": None,
            "top_risk_factors": []
        }

        # Check probability support
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(scaled_features)[0]
            # Classes are typically [0, 1] or ['B', 'M']
            classes = getattr(self.model, "classes_", [0, 1])
            benign_idx = 0
            malignant_idx = 1
            for idx, c in enumerate(classes):
                if str(c) in ["0", "B", "Benign"]:
                    benign_idx = idx
                elif str(c) in ["1", "M", "Malignant"]:
                    malignant_idx = idx

            benign_prob = float(proba[benign_idx])
            malignant_prob = float(proba[malignant_idx])
            result["probabilities"] = {
                "Benign": benign_prob,
                "Malignant": malignant_prob
            }
            result["benign_prob"] = benign_prob
            result["malignant_prob"] = malignant_prob
            result["confidence"] = malignant_prob if code == 1 else benign_prob

        # Feature contribution analysis (for linear models with coef_)
        if hasattr(self.model, "coef_"):
            coefs = self.model.coef_[0]
            scaled_vals = scaled_features[0]
            contributions = []
            for i, feat_name in enumerate(FEATURE_NAMES):
                impact = coefs[i] * scaled_vals[i]
                contributions.append({
                    "feature": feat_name,
                    "coefficient": float(coefs[i]),
                    "scaled_value": float(scaled_vals[i]),
                    "raw_value": float(raw_inputs[feat_name]),
                    "impact": float(impact),
                    "pushes_malignant": bool(impact > 0)
                })
            # Sort by absolute impact
            contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
            result["top_risk_factors"] = contributions[:8]

        return result


# Singleton instance for easy reuse in Streamlit with caching
_global_predictor: Optional[ModelPredictor] = None

def get_predictor() -> ModelPredictor:
    """Returns a cached instance of the ModelPredictor."""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = ModelPredictor()
    return _global_predictor
