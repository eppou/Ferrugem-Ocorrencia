from pprint import pprint

import geopandas as gpd
import seaborn as sns
from sqlalchemy import create_engine

from constants import DB_STRING, INPUT_PATH


def import_postgres():
    sns.set(style="whitegrid", palette="pastel", color_codes=True)
    sns.mpl.rc("figure", figsize=(10, 6))

    path = INPUT_PATH / "shapefiles_LimitesMunicipaisPR/LimitesMunicipaisPR.shp"
    # sf = shp.Reader(path)

    geodf = gpd.read_file(path, crs="epsg:4674")

    geodf = geodf[["nome", "geometry"]]
    geodf.columns = ["city_name", "geometry"]
    geodf.set_crs("epsg:4674", inplace=True, allow_override=True)
    pprint(geodf.columns)
    pprint(geodf.crs)

    db_con_engine = create_engine(DB_STRING)
    geodf.to_postgis("city_shapefiles", db_con_engine, if_exists="replace")

    pprint(geodf.sample(5))
