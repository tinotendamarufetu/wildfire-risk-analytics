import pytest
from src.underwriting_engine import UnderwritingEngine


@pytest.fixture
def engine():
    """Fixture initializing UnderwritingEngine with base premium $2,000."""
    return UnderwritingEngine(base_premium=2000.0)


@pytest.fixture
def sample_property():
    """Fixture defining base property feature profile."""
    return {
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


def test_model_inference_and_score_range(engine, sample_property):
    """Verify underwriting engine returns valid risk score between 0 and 100."""
    result = engine.calculate_mitigation_adjustments(sample_property)
    
    assert "final_risk_score" in result
    assert 0.0 <= result["final_risk_score"] <= 100.0
    assert result["recommended_annual_premium"] > 0.0


def test_mitigation_discount_impact(engine, sample_property):
    """Verify property hardening reduces risk score and premium cost."""
    unmitigated = engine.calculate_mitigation_adjustments(
        sample_property, defensible_space_ft=0, roof_material="Standard", ember_vents=False
    )
    
    hardened = engine.calculate_mitigation_adjustments(
        sample_property, defensible_space_ft=100, roof_material="Class A Fire-Resistant", ember_vents=True
    )
    
    assert hardened["final_risk_score"] < unmitigated["final_risk_score"], "Hardening failed to reduce risk score."
    assert hardened["recommended_annual_premium"] < unmitigated["recommended_annual_premium"], "Hardening failed to reduce premium."


def test_climate_stress_testing(engine, sample_property):
    """Verify extreme drought scenario increases risk score output."""
    baseline = engine.calculate_mitigation_adjustments(sample_property, climate_scenario="Baseline")
    drought = engine.calculate_mitigation_adjustments(sample_property, climate_scenario="Extreme Drought")
    
    assert drought["raw_model_score"] >= baseline["raw_model_score"], "Climate stress test failed to elevate risk."