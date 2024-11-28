from pprint import pprint

import geopandas as gpd
from sqlalchemy import create_engine

from config import Config
from constants import INPUT_PATH


def run(cfg: Config):
    # Import NON-soybean areas in state of Paraná (PR)
    path = INPUT_PATH / "shapefiles_soybean_areas/shapefile_non_soybean_areas_parana_2021.gpkg"

    gdf_non_soybean_areas_parana = gpd.read_file(path, crs="epsg:4674")

    # Ajusting df and adding additional information
    gdf_non_soybean_areas_parana = gdf_non_soybean_areas_parana[["id", "geometry"]]
    gdf_non_soybean_areas_parana["state"] = "PR"
    gdf_non_soybean_areas_parana["has_soybean"] = False

    # Forcing crs update
    gdf_non_soybean_areas_parana.set_crs("epsg:4674", inplace=True, allow_override=True)

    # Checking out
    print("gdf_non_soybean_areas_parana: How it looks like:")
    pprint(gdf_non_soybean_areas_parana.head(20))
    print("gdf_non_soybean_areas_parana: What about CRS?")
    pprint(gdf_non_soybean_areas_parana.crs)

    # Persisting into new DB table
    db_con_engine = create_engine(cfg.database_config.dbstring)
    gdf_non_soybean_areas_parana.to_postgis("soybean_areas", db_con_engine, if_exists="replace")
