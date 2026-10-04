import joblib
import os
import numpy as np
import pandas as pd
import shap


class UnderwritingEngine:
    def __init__(self, model_path: str = "models/xgboost_wildfire_model.joblib", base_premium: float = 2200.0):
        """
        Initializes the Underwriting Engine with a trained XGBoost model and base policy premium.
        """
        self.base_premium = base_premium
        self.model = None
        self.explainer = None

        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self.explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                print(f"[!] Warning: Could not load trained model from {model_path}: {e}")

    def predict_risk_probability(self, features: dict) -> float:
        """Calculates raw wildfire risk probability (0.0 to 1.0) using the loaded model or heuristic fallback."""
        feature_order = [
            "elevation_m", "slope_pct", "dist_nearest_fire_m",
            "fires_within_5km", "fires_within_10km", "years_since_last_fire",
            "max_vpd", "max_wind_speed", "total_precip_annual"
        ]

        if self.model is not None:
            df = pd.DataFrame([features])[feature_order]
            prob = float(self.model.predict_proba(df)[0, 1])
        else:
            # Fallback risk calculation if model isn't present
            logit = (
                    -2.5
                    + (0.1 * features.get("slope_pct", 10.0))
                    - (0.00015 * features.get("dist_nearest_fire_m", 5000.0))
                    + (0.5 * features.get("fires_within_5km", 1))
            )
            prob = 1.0 / (1.0 + np.exp(-logit))

        return float(np.clip(prob, 0.01, 0.99))

    def compute_shap_values(self, features: dict) -> dict:
        """Computes individual SHAP feature contributions for a property feature dictionary."""
        feature_order = [
            "elevation_m", "slope_pct", "dist_nearest_fire_m",
            "fires_within_5km", "fires_within_10km", "years_since_last_fire",
            "max_vpd", "max_wind_speed", "total_precip_annual"
        ]

        df = pd.DataFrame([features])[feature_order]

        if self.explainer is not None:
            shap_vals = self.explainer.shap_values(df)[0]
            return {col: float(val) for col, val in zip(feature_order, shap_vals)}

        # Heuristic fallback for SHAP contributions
        return {
            "slope_pct": round(features.get("slope_pct", 10.0) * 0.08, 4),
            "dist_nearest_fire_m": round((10000.0 - features.get("dist_nearest_fire_m", 5000.0)) * 0.0001, 4),
            "total_precip_annual": -0.15,
            "max_vpd": 0.12,
            "max_wind_speed": round(features.get("max_wind_speed", 30.0) * 0.003, 4),
            "years_since_last_fire": 0.05,
            "fires_within_10km": 0.04,
            "fires_within_5km": 0.02,
            "elevation_m": -0.01
        }

    def calculate_mitigation_adjustments(
            self,
            features: dict,
            defensible_space_ft: float = 0,
            roof_material: str = "Standard",
            ember_vents: bool = False,
            climate_scenario: str = "Baseline"
    ) -> dict:
        """
        Evaluates risk score, applies property hardening credits, models climate stress scenarios,
        and computes actuarial premium pricing.
        """
        # 1. Base Model Probability & Risk Score
        raw_prob = self.predict_risk_probability(features)

        # 2. Climate Stress Testing Factor
        climate_multiplier = 1.25 if climate_scenario == "Extreme Drought" else 1.00
        stressed_prob = min(0.99, raw_prob * climate_multiplier)
        base_risk_score = stressed_prob * 100.0

        # 3. Calculate Hardening Mitigation Discounts
        defensible_space_credit = min(18.0, (defensible_space_ft / 100.0) * 18.0)
        roof_credit = 15.0 if roof_material in ["Metal / Tile", "Class A Fire-Resistant"] else 0.0
        vent_credit = 7.0 if ember_vents else 0.0

        total_mitigation_pts = float(defensible_space_credit + roof_credit + vent_credit)
        final_risk_score = float(np.round(max(0.0, base_risk_score - total_mitigation_pts), 1))

        # 4. Actuarial Premium Calculation
        # Risk Multiplier Scale based on final risk score
        risk_multiplier = 1.0 + ((final_risk_score / 100.0) ** 1.8) * 2.2
        recommended_premium = float(np.round(self.base_premium * risk_multiplier, 2))

        # 5. Underwriting Decision Thresholds
        if final_risk_score < 30.0:
            status = "APPROVED / LOW RISK"
        elif final_risk_score < 60.0:
            status = "APPROVED / CONDITIONAL"
        else:
            status = "DECLINED / HIGH RISK"

        # 6. Get SHAP Feature Attribution
        shap_contribs = self.compute_shap_values(features)

        return {
            "base_risk_score": float(np.round(base_risk_score, 1)),
            "final_risk_score": final_risk_score,
            "mitigation_credit_pts": float(np.round(total_mitigation_pts, 1)),
            "recommended_annual_premium": recommended_premium,
            "underwriting_status": status,
            "shap_contributions": shap_contribs
        }


if __name__ == "__main__":
    print("\n--- Testing Underwriting & Mitigation Simulation Engine ---")
    engine = UnderwritingEngine(base_premium=2200.0)

    sample_props = {
        "elevation_m": 285.0,
        "slope_pct": 18.5,
        "dist_nearest_fire_m": 916.0,
        "fires_within_5km": 2,
        "fires_within_10km": 5,
        "years_since_last_fire": 6,
        "max_vpd": 4.2,
        "max_wind_speed": 42.0,
        "total_precip_annual": 380.0
    }

    # Scenario A: Unmitigated Property
    res_a = engine.calculate_mitigation_adjustments(sample_props, defensible_space_ft=0, roof_material="Standard")
    print("\n[Scenario A: Unmitigated Property]")
    print(f"  - Risk Score:       {res_a['final_risk_score']} / 100")
    print(f"  - Decision:         {res_a['underwriting_status']}")
    print(f"  - Annual Premium:   ${res_a['recommended_annual_premium']:,.2f}")

    # Scenario B: Fully Hardened Property
    res_b = engine.calculate_mitigation_adjustments(
        sample_props, defensible_space_ft=100, roof_material="Class A Fire-Resistant", ember_vents=True
    )
    print("\n[Scenario B: Fully Hardened Property]")
    print(f"  - Risk Score:       {res_b['final_risk_score']} / 100")
    print(f"  - Decision:         {res_b['underwriting_status']}")
    print(f"  - Annual Premium:   ${res_b['recommended_annual_premium']:,.2f}")

    savings = res_a['recommended_annual_premium'] - res_b['recommended_annual_premium']
    print(f"\n[OK] Annual Premium Savings achieved via mitigation: ${savings:,.2f}\n")