import geopandas as gpd
from shapely.geometry import Point

from testlab.constants import INPUT_PATH


def run():
    path = INPUT_PATH / "shapefile_non_soybean_areas_parana_2021.gpkg"

    gdf_non_soybean_areas_parana = gpd.read_file(path, crs="epsg:4674")
    # print(gdf_non_soybean_areas_parana.head(20))
    # print(gdf_non_soybean_areas_parana.shape[0])

    probe_1_out = Point(-52.128577901747235, -24.524442964105837)
    probe_2_out = Point(-52.181023810244035, -24.432884852662266)
    probe_3_in = Point(-52.09568809472382, -24.481775106345726)

    print(contains(gdf_non_soybean_areas_parana, probe_1_out))
    print(contains(gdf_non_soybean_areas_parana, probe_2_out))
    print(contains(gdf_non_soybean_areas_parana, probe_3_in))


def contains(gdf_shape: gpd.GeoDataFrame, geometry) -> bool:
    intersect_series = gdf_shape.contains(geometry)

    return intersect_series[intersect_series].index.shape[0] > 0
