from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text

from constants import DB_STRING
from data_preparation.constants import QUERY_PRECIPITATION
from helpers.input_output import output_file

"""
This just fetch precipitation from database, for min and max days for harvest start/end dates and writes in a file.
"""


def run():
    db_con_engine = create_engine(DB_STRING)
    conn = db_con_engine.connect()
    execution_start = datetime.now()

    precipitation_df = pd.read_sql_query(sql=text(QUERY_PRECIPITATION), con=conn)

    precipitation_df.to_csv(output_file(execution_start, "precipitation", "precipitation_all.csv"))
