import numpy as np
import geopandas as gpd
from shapely.geometry import Point


class SpatialProcessor:
    """Handles CRS transformations, spatial buffering, and distance calculations."""

    # UTM Zone 11N (EPSG:32611) is optimal for accurate metric calculations in California
    PROJECTED_CRS = "EPSG:32611"
    WGS84_CRS = "EPSG:4326"

    @staticmethod
    def ensure_crs(gdf: gpd.GeoDataFrame, target_crs: str = PROJECTED_CRS) -> gpd.GeoDataFrame:
        """Reprojects a GeoDataFrame to a target CRS for metric spatial calculations."""
        if gdf.crs is None:
            gdf = gdf.set_crs(SpatialProcessor.WGS84_CRS)
        if gdf.crs.to_string() != target_crs:
            gdf = gdf.to_crs(target_crs)
        return gdf

    @staticmethod
    def create_point_geometry(lat: float, lon: float) -> gpd.GeoDataFrame:
        """Creates a single-point GeoDataFrame from lat/lon coordinates."""
        point = Point(lon, lat)
        gdf = gpd.GeoDataFrame(geometry=[point], crs=SpatialProcessor.WGS84_CRS)
        return SpatialProcessor.ensure_crs(gdf)

    @staticmethod
    def compute_distance_to_nearest(
        target_point_gdf: gpd.GeoDataFrame, 
        polygons_gdf: gpd.GeoDataFrame
    ) -> float:
        """
        Calculates the minimum distance in meters from a target property point 
        to the nearest perimeter polygon in a dataset.
        """
        if polygons_gdf.empty:
            return 999999.0  # Default large distance if no perimeters exist
            
        target_proj = SpatialProcessor.ensure_crs(target_point_gdf)
        polygons_proj = SpatialProcessor.ensure_crs(polygons_gdf)
        
        # Calculate distances to all geometries
        distances = polygons_proj.geometry.distance(target_proj.geometry.iloc[0])
        return float(distances.min())

    @staticmethod
    def count_intersections_in_buffer(
        target_point_gdf: gpd.GeoDataFrame, 
        polygons_gdf: gpd.GeoDataFrame, 
        buffer_radius_meters: float = 5000.0
    ) -> int:
        """Counts how many historical fire polygons fall within a given buffer radius."""
        if polygons_gdf.empty:
            return 0
            
        target_proj = SpatialProcessor.ensure_crs(target_point_gdf)
        polygons_proj = SpatialProcessor.ensure_crs(polygons_gdf)
        
        buffered_point = target_proj.geometry.buffer(buffer_radius_meters).iloc[0]
        intersections = polygons_proj[polygons_proj.geometry.intersects(buffered_point)]
        return len(intersections)