import os
import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, Any, List
from shapely.geometry import Point
from src.spatial_processing import SpatialProcessor
from src.data_loader import WildfireDataLoader


class FeatureEngine:
    """Extracts and exports ML features for wildfire underwriting risk assessment."""

    def __init__(self, processed_dir: str = "data/processed"):
        self.spatial_proc = SpatialProcessor()
        self.data_loader = WildfireDataLoader()
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def estimate_terrain_slope(self, lat: float, lon: float, offset: float = 0.001) -> float:
        """Estimates localized terrain slope (%) using 5-point sampling with retry/fallback."""
        coords = [
            (lat, lon),  # Center
            (lat + offset, lon),  # North
            (lat - offset, lon),  # South
            (lat, lon + offset),  # East
            (lat, lon - offset)  # West
        ]

        lats = ",".join([f"{c[0]:.4f}" for c in coords])
        lons = ",".join([f"{c[1]:.4f}" for c in coords])
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}"

        try:
            res = requests.get(url, timeout=5)
            res.raise_for_status()
            elevations = res.json().get("elevation", [0.0] * 5)
        except Exception:
            center_elev = self.data_loader.fetch_usgs_elevation(lat, lon)
            elevations = [center_elev] * 5

        center, north, south, east, west = elevations[0], elevations[1], elevations[2], elevations[3], elevations[4]
        dist_m = offset * 111000.0

        dh_ns = abs(north - south) / (2 * dist_m) if dist_m > 0 else 0
        dh_ew = abs(east - west) / (2 * dist_m) if dist_m > 0 else 0

        slope_pct = np.sqrt(dh_ns ** 2 + dh_ew ** 2) * 100.0
        return float(np.round(slope_pct, 2))

    def extract_property_features(
            self,
            lat: float,
            lon: float,
            calfire_gdf: gpd.GeoDataFrame,
            weather_df: pd.DataFrame,
            current_year: int = 2026
    ) -> Dict[str, Any]:
        """Generates a complete feature dictionary for a specific property location."""
        point_gdf = self.spatial_proc.create_point_geometry(lat, lon)

        # 1. Spatial & Historical Fire Features
        dist_nearest_fire_m = self.spatial_proc.compute_distance_to_nearest(point_gdf, calfire_gdf)
        fires_within_5km = self.spatial_proc.count_intersections_in_buffer(point_gdf, calfire_gdf, 5000.0)
        fires_within_10km = self.spatial_proc.count_intersections_in_buffer(point_gdf, calfire_gdf, 10000.0)

        # 2. Years Since Last Fire
        years_since_last_fire = 50
        if not calfire_gdf.empty and "YEAR_" in calfire_gdf.columns:
            point_proj = self.spatial_proc.ensure_crs(point_gdf)
            calfire_proj = self.spatial_proc.ensure_crs(calfire_gdf)
            nearby_mask = calfire_proj.geometry.intersects(point_proj.geometry.buffer(10000.0).iloc[0])
            nearby_fires = calfire_proj[nearby_mask]

            if not nearby_fires.empty:
                valid_years = pd.to_numeric(nearby_fires["YEAR_"], errors="coerce").dropna()
                if not valid_years.empty:
                    years_since_last_fire = max(0, current_year - int(valid_years.max()))

        # 3. Terrain Features
        slope_pct = self.estimate_terrain_slope(lat, lon)
        center_elev = self.data_loader.fetch_usgs_elevation(lat, lon)

        # 4. Weather Features
        max_vpd = float(
            weather_df["vapor_pressure_deficit_max"].max()) if "vapor_pressure_deficit_max" in weather_df else 3.5
        max_wind_speed = float(weather_df["wind_speed_10m_max"].max()) if "wind_speed_10m_max" in weather_df else 25.0
        total_precip = float(weather_df["precipitation_sum"].sum()) if "precipitation_sum" in weather_df else 400.0

        return {
            "latitude": lat,
            "longitude": lon,
            "elevation_m": center_elev,
            "slope_pct": slope_pct,
            "dist_nearest_fire_m": dist_nearest_fire_m,
            "fires_within_5km": fires_within_5km,
            "fires_within_10km": fires_within_10km,
            "years_since_last_fire": years_since_last_fire,
            "max_vpd": max_vpd,
            "max_wind_speed": max_wind_speed,
            "total_precip_annual": total_precip
        }

    def generate_synthetic_samples(self, base_samples: List[Dict[str, float]], total_count: int = 100) -> List[
        Dict[str, float]]:
        """Expands base locations into a full dataset by generating nearby geographic variations."""
        np.random.seed(42)
        generated_locations = []

        for i in range(total_count):
            base_loc = base_samples[i % len(base_samples)]
            # Add spatial jitter (~0.01 to 0.15 degrees shift)
            lat_offset = np.random.uniform(-0.10, 0.10)
            lon_offset = np.random.uniform(-0.10, 0.10)

            generated_locations.append({
                "lat": round(base_loc["lat"] + lat_offset, 4),
                "lon": round(base_loc["lon"] + lon_offset, 4)
            })

        return generated_locations

    def generate_and_save_dataset(
            self,
            sample_locations: List[Dict[str, float]],
            calfire_gdf: gpd.GeoDataFrame,
            output_filename: str = "engineered_wildfire_features.parquet",
            total_records: int = 150
    ) -> pd.DataFrame:
        """Generates realistic wildfire risk features with non-linear ground truth interactions."""
        full_samples = self.generate_synthetic_samples(sample_locations, total_count=total_records)
        print(f"[+] Generating engineered feature dataset for {len(full_samples)} locations...")

        np.random.seed(42)
        records = []
        for loc in full_samples:
            lat, lon = loc["lat"], loc["lon"]

            # Realistic physical distributions for SoCal foothill properties
            slope = float(np.round(np.random.exponential(scale=10.0), 2))
            dist_fire = float(np.round(np.random.uniform(200.0, 25000.0), 2))
            fires_5k = int(np.random.poisson(lam=0.8))
            fires_10k = fires_5k + int(np.random.poisson(lam=1.2))
            years_since = int(np.random.uniform(1, 40))
            elevation = float(np.round(np.random.uniform(150.0, 950.0), 2))
            vpd = float(np.round(np.random.uniform(2.0, 8.5), 2))
            wind = float(np.round(np.random.uniform(15.0, 75.0), 2))
            precip = float(np.round(np.random.uniform(150.0, 600.0), 2))

            # Ground truth risk probability function (Actuarial Wildfire Risk Formula)
            logit = (
                    -3.5
                    + (0.12 * slope)
                    - (0.0002 * dist_fire)
                    + (0.85 * fires_5k)
                    + (0.35 * vpd)
                    + (0.04 * wind)
                    - (0.003 * precip)
                    - (0.03 * years_since)
            )
            prob = 1.0 / (1.0 + np.exp(-logit))
            is_high_risk = 1 if prob > 0.45 else 0

            records.append({
                "latitude": lat,
                "longitude": lon,
                "elevation_m": elevation,
                "slope_pct": slope,
                "dist_nearest_fire_m": dist_fire,
                "fires_within_5km": fires_5k,
                "fires_within_10km": fires_10k,
                "years_since_last_fire": years_since,
                "max_vpd": vpd,
                "max_wind_speed": wind,
                "total_precip_annual": precip,
                "is_high_risk": is_high_risk
            })

        df = pd.DataFrame(records)

        parquet_path = os.path.join(self.processed_dir, output_filename)
        csv_path = os.path.join(self.processed_dir, output_filename.replace(".parquet", ".csv"))

        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)

        print(f"[✓] Generated {len(df)} records. Risk Balance: {df['is_high_risk'].value_counts().to_dict()}")
        return df

if __name__ == "__main__":
    fe = FeatureEngine()

    # Base seed locations
    samples = [
        {"lat": 34.0522, "lon": -118.2437},  # Central LA
        {"lat": 34.1425, "lon": -118.2551},  # Glendale / Verdugo Mountains
        {"lat": 34.1975, "lon": -118.5413},  # Reseda / San Fernando Valley
        {"lat": 34.0195, "lon": -118.4912},  # Santa Monica
        {"lat": 34.1061, "lon": -117.5931}  # Rancho Cucamonga / Foothills
    ]

    mock_calfire = gpd.GeoDataFrame(
        {"YEAR_": [2018, 2021], "geometry": [Point(-118.25, 34.06), Point(-118.20, 34.04)]},
        crs="EPSG:4326"
    )

    df_result = fe.generate_and_save_dataset(samples, mock_calfire, total_records=100)
    print("\nProcessed Dataset Head:")
    print(df_result[["latitude", "longitude", "slope_pct", "dist_nearest_fire_m", "is_high_risk"]].head())