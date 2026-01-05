import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine

from config import Config
from constants import INPUT_PATH


def run(cfg: Config):
    path = INPUT_PATH / "cptec_precipitation_2022.csv"

    df = pd.read_csv(path, sep=",")
    df["source"] = "CPTEC"

    geodf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.latitude, df.longitude), crs="epsg:4674"
    )

    db_con_engine = create_engine(cfg.database_config.dbstring)
    geodf.to_postgis("precipitation", db_con_engine, if_exists="replace")

    print(geodf.sample(10))
