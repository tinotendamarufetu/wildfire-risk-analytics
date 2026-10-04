import pytest
import pandas as pd
import numpy as np
import os
from src.feature_engineering import WildfireFeatureEngineer


@pytest.fixture
def feature_engineer():
    """Fixture providing an instance of WildfireFeatureEngineer."""
    return WildfireFeatureEngineer()


def test_processed_dataset_exists():
    """Verify processed feature files are generated in target path."""
    parquet_path = "data/processed/engineered_wildfire_features.parquet"
    csv_path = "data/processed/engineered_wildfire_features.csv"
    
    assert os.path.exists(parquet_path), "Parquet feature dataset missing."
    assert os.path.exists(csv_path), "CSV feature dataset missing."


def test_engineered_features_schema(feature_engineer):
    """Verify processed dataset contains all required model features."""
    df = pd.read_parquet("data/processed/engineered_wildfire_features.parquet")
    
    required_cols = [
        "latitude", "longitude", "elevation_m", "slope_pct",
        "dist_nearest_fire_m", "fires_within_5km", "fires_within_10km",
        "years_since_last_fire", "max_vpd", "max_wind_speed",
        "total_precip_annual", "is_high_risk"
    ]
    
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"


def test_haversine_distance_calculation(feature_engineer):
    """Verify spatial distance calculation logic between two known coordinates."""
    # Distance between Los Angeles (34.0522, -118.2437) and Glendale (34.1425, -118.2551) ~10.1 km
    lat1, lon1 = 34.0522, -118.2437
    lat2, lon2 = 34.1425, -118.2551
    
    dist_m = feature_engineer.haversine_distance(lat1, lon1, lat2, lon2)
    assert 9500.0 <= dist_m <= 10800.0, f"Unexpected calculated distance: {dist_m}m"