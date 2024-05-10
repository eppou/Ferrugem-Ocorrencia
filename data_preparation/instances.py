from datetime import datetime, date, timedelta

import pandas as pd
from sqlalchemy import create_engine

from calculation.coordinates import find_nearest_segment_id
from constants import DB_STRING
from data_preparation.constants import QUERY_OCORRENCIAS
from helpers.input_output import output_file
from source.occurrence import get_safras

import statistics as s

# 1. Coletar todas as ocorrências por safra. Calcular features por data de ocorrência.
# 2. Contar ocorrências e gerar não-ocorrencias por safra. Método: Sorteio. Definir distancia = 2 graus em duas direções
# 3. Para cada não-ocorrência, usar dados da safra para gerar features. Chutar um intervalo entre o começo e fim da
# safra como "data de verificado como não ocorrência". Calcular features.

def run():
    db_con_engine = create_engine(DB_STRING)
    conn = db_con_engine.connect()
    execution_started = datetime.now()

    instances_df_all = pd.DataFrame()

    safras = get_safras(conn)

    for safra in safras:
        # INSTANCE DETERMINATION
        print(f"=====> Processing safra {safra['safra']}...")

        # Fetching occurrences from consorcio_antiferrugem database, per harvest
        df = pd.read_sql_query(
            QUERY_OCORRENCIAS.replace(":safra", safra["safra"]),
            con=db_con_engine,
            parse_dates=["data_ocorrencia"],
        )

        instances_df_all = pd.concat([instances_df_all, df])

    # Assigning a segment_id_precipitation - a match for a position for the nearest precipitation data
    print("=====> Assigning a segment_id_precipitation (for precipitation data)")
    instances_df_all = instances_df_all.reset_index()

    for index, instance in instances_df_all.iterrows():
        latitude = instance["ocorrencia_latitude"]
        longitude = instance["ocorrencia_longitude"]
        occurrence_id = instance["ocorrencia_id"]

        print(f"Finding nearest segment for (lat/long) {latitude} {longitude}, index {index}")

        segment_id_precipitation = find_nearest_segment_id(conn, occurrence_id, latitude, longitude)
        print(f"Segment found: {segment_id_precipitation}, index {index}")

        # New fields
        instances_df_all.at[index, "segment_id_precipitation"] = segment_id_precipitation
        instances_df_all.at[index, "planting_start_date"] = calculate_planting_start_date(instance)

    print(f"=====> Size of instances dataset: {instances_df_all.shape[0]}")

    # Output full dataset (possible contain extra information for debugging and visualization)
    instances_df_all.to_csv(
        output_file(execution_started, "instances", "instances_all.csv"), index=False
    )

    instances_df = instances_df_all
    [[
        "safra", "ocorrencia_latitude", "ocorrencia_longitude", "ocorrencia"
    ]].copy()
    instances_df.to_csv(
        output_file(execution_started, "instances", "instance.csv"), index=False
    )

    conn.close()
    db_con_engine.dispose()


def calculate_planting_start_date(instance: pd.Series) -> datetime.date:
    occurrence_date = instance["data_ocorrencia"]

    emergence_days_min = instance["emergence_days_min"]
    emergence_days_max = instance["emergence_days_max"]
    days_after_emergence_min = instance["days_after_emergence_min"]
    days_after_emergence_max = instance["days_after_emergence_max"]

    emergence_days = s.mean([emergence_days_max, emergence_days_min])
    days_after_emergence = s.mean([days_after_emergence_max, days_after_emergence_min])

    planting_relative_day = round(emergence_days + days_after_emergence, 0)

    return occurrence_date - timedelta(days=planting_relative_day)
