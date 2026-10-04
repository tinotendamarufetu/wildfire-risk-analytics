import os
import duckdb
import pandas as pd


class DatabaseManager:
    """Manages DuckDB spatial database queries over processed GeoParquet datasets."""

    def __init__(self, db_path: str = "data/processed/wildfire_analytics.duckdb"):
        self.db_path = db_path
        self.conn = duckdb.connect(self.db_path)
        self._init_spatial_extension()

    def _init_spatial_extension(self):
        """Installs and loads the DuckDB spatial extension."""
        self.conn.execute("INSTALL spatial; LOAD spatial;")

    def load_parquet_to_table(
        self, 
        parquet_path: str = "data/processed/engineered_wildfire_features.parquet", 
        table_name: str = "property_risk_features"
    ):
        """Loads a GeoParquet/Parquet file directly into a DuckDB table."""
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found at: {parquet_path}")

        query = f"""
        CREATE OR REPLACE TABLE {table_name} AS 
        SELECT * FROM read_parquet('{parquet_path}');
        """
        self.conn.execute(query)
        print(f"[✓] DuckDB table '{table_name}' initialized from {parquet_path}.")

    def query_high_risk_properties(self, min_slope: float = 5.0) -> pd.DataFrame:
        """SQL query example retrieving high-risk properties with slope above threshold."""
        query = f"""
        SELECT 
            latitude, 
            longitude, 
            slope_pct, 
            dist_nearest_fire_m, 
            is_high_risk 
        FROM property_risk_features 
        WHERE slope_pct >= {min_slope} 
        ORDER BY dist_nearest_fire_m ASC;
        """
        return self.conn.execute(query).df()

    def close(self):
        """Closes the DuckDB database connection."""
        self.conn.close()


if __name__ == "__main__":
    db = DatabaseManager()
    db.load_parquet_to_table()
    
    print("\n--- Testing DuckDB Spatial Query Engine ---")
    df_high_risk = db.query_high_risk_properties(min_slope=3.0)
    print(df_high_risk)
    db.close()