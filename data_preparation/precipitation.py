from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text

from config import Config
from data_preparation.constants import QUERY_PRECIPITATION
from helpers.input_output import output_file

"""
This just fetch precipitation from database, for min and max days for harvest start/end dates and writes in a file.

Used for analysis only. Model feature extraction takes directly from DB.
"""


def run(execution_started_at: datetime, cfg: Config):
    db_con_engine = create_engine(cfg.database_config.dbstring)
    conn = db_con_engine.connect()

    precipitation_df = pd.read_sql_query(sql=text(QUERY_PRECIPITATION), con=conn)

    precipitation_df.to_csv(output_file(execution_started_at, "precipitation", "precipitation.csv"))
