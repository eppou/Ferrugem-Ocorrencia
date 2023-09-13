import geopandas as gpd


def contains(gdf_shape: gpd.GeoDataFrame, geometry) -> bool:
    intersect_series = gdf_shape.contains(geometry)

    return intersect_series[intersect_series].index.shape[0] > 0
