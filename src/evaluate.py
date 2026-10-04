import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Dict, Any


class WildfireRiskEvaluator:
    """Provides SHAP model interpretability, feature attribution, and underwriting risk scores."""

    def __init__(
        self, 
        model_path: str = "models/xgboost_wildfire_model.joblib",
        data_path: str = "data/processed/engineered_wildfire_features.parquet",
        outputs_dir: str = "data/outputs"
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.outputs_dir = outputs_dir
        os.makedirs(self.outputs_dir, exist_ok=True)
        
        # Load production model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        self.model = joblib.load(self.model_path)
        
        # Initialize SHAP TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)

    def load_feature_data(self) -> pd.DataFrame:
        """Loads dataset and extracts feature matrix."""
        df = pd.read_parquet(self.data_path)
        feature_cols = [
            "elevation_m", "slope_pct", "dist_nearest_fire_m", 
            "fires_within_5km", "fires_within_10km", "years_since_last_fire",
            "max_vpd", "max_wind_speed", "total_precip_annual"
        ]
        return df[feature_cols]

    def generate_global_shap_summary(self, save_filename: str = "shap_summary_plot.png") -> str:
        """Generates and saves a global SHAP summary feature importance plot."""
        X = self.load_feature_data()
        shap_values = self.explainer(X)
        
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X, show=False)
        output_path = os.path.join(self.outputs_dir, save_filename)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        
        print(f"[OK] Global SHAP summary plot saved to: {output_path}")
        return output_path

    def explain_property_prediction(self, property_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates property-specific SHAP values and computes a 0-100 continuous Risk Score.
        """
        feature_cols = [
            "elevation_m", "slope_pct", "dist_nearest_fire_m", 
            "fires_within_5km", "fires_within_10km", "years_since_last_fire",
            "max_vpd", "max_wind_speed", "total_precip_annual"
        ]
        
        input_df = pd.DataFrame([property_features])[feature_cols]
        
        # Predict High Risk Probability (0.0 to 1.0)
        prob_high_risk = float(self.model.predict_proba(input_df)[0][1])
        
        # Calibrated 0-100 Risk Score
        risk_score_100 = float(np.round(prob_high_risk * 100, 1))
        
        # SHAP calculation for single instance
        shap_vals = self.explainer(input_df)
        contributions = dict(zip(feature_cols, shap_vals.values[0]))
        
        # Determine underwriting tier
        if risk_score_100 >= 70.0:
            underwriting_action = "DECLINE / HIGH RISK"
        elif risk_score_100 >= 35.0:
            underwriting_action = "MANUAL REVIEW REQUIRED"
        else:
            underwriting_action = "APPROVE / LOW RISK"

        return {
            "risk_score_100": risk_score_100,
            "underwriting_action": underwriting_action,
            "high_risk_probability": np.round(prob_high_risk, 4),
            "feature_shap_contributions": contributions
        }


if __name__ == "__main__":
    evaluator = WildfireRiskEvaluator()
    
    print("\n--- Testing Model Explainability Engine (SHAP) ---")
    
    # 1. Generate Global Summary Plot
    evaluator.generate_global_shap_summary()
    
    # 2. Test Single Property Evaluation
    sample_property = {
        "elevation_m": 120.5,
        "slope_pct": 18.5,
        "dist_nearest_fire_m": 1200.0,
        "fires_within_5km": 2,
        "fires_within_10km": 3,
        "years_since_last_fire": 4,
        "max_vpd": 4.8,
        "max_wind_speed": 38.0,
        "total_precip_annual": 450.0
    }
    
    assessment = evaluator.explain_property_prediction(sample_property)
    
    print("\nUnderwriting Assessment Result:")
    print(f"  - Risk Score (0-100):  {assessment['risk_score_100']}")
    print(f"  - Underwriting Action: {assessment['underwriting_action']}")
    print(f"  - Probability:         {assessment['high_risk_probability']}")
    print("\nTop Feature Contributions (SHAP):")
    for feat, shap_val in sorted(assessment['feature_shap_contributions'].items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  - {feat:22s}: {shap_val:+.4f}")