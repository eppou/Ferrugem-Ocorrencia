import statistics as s
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import create_engine

from calculation.coordinates import find_nearest_segment_id
from config import Config
from data_preparation.constants import QUERY_OCORRENCIAS
from helpers.input_output import output_file
from source.occurrence import get_safras


# 1. Coletar todas as ocorrências por safra. Calcular features por data de ocorrência.
# 2. Contar ocorrências e gerar não-ocorrencias por safra. Método: Sorteio. Definir distancia = 2 graus em duas direções
# 3. Para cada não-ocorrência, usar dados da safra para gerar features. Chutar um intervalo entre o começo e fim da
# safra como "data de verificado como não ocorrência". Calcular features.

def run(execution_started_at: datetime, cfg: Config):
    db_con_engine = create_engine(cfg.database_config.dbstring)
    conn = db_con_engine.connect()

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

        print(f"Finding nearest precipitation segment for (lat/long) {latitude} {longitude}, index {index}")

        segment_id_precipitation = find_nearest_segment_id(conn, occurrence_id, latitude, longitude)
        print(f"Segment found: {segment_id_precipitation}, index {index}")

        # New fields
        instances_df_all.at[index, "segment_id_precipitation"] = segment_id_precipitation
        instances_df_all.at[index, "planting_start_date"] = calculate_planting_start_date(instance)

    print(f"=====> Size of instances dataset: {instances_df_all.shape[0]}")

    # Output full dataset (possible contain extra information for debugging and visualization)
    instances_df_all.to_csv(
        output_file(execution_started_at, "instances", "instances_all.csv"), index=False
    )

    instances_df = instances_df_all
    [[
        "safra", "ocorrencia_latitude", "ocorrencia_longitude", "ocorrencia"
    ]].copy()
    instances_df.to_csv(
        output_file(execution_started_at, "instances", "instance.csv"), index=False
    )

    conn.close()
    db_con_engine.dispose()


def calculate_planting_start_date(instance: pd.Series) -> datetime.date:
    occurrence_date = instance["data_ocorrencia"]

    emergence_days_min = instance["emergence_days_min"]
    emergence_days_max = instance["emergence_days_max"]
    days_after_emergence_min = instance["days_after_emergence_min"]
    days_after_emergence_max = instance["days_after_emergence_max"]

    emergence_days_mean = s.mean([emergence_days_max, emergence_days_min])
    days_after_emergence_mean = s.mean([days_after_emergence_max, days_after_emergence_min])

    interval = round(emergence_days_mean + days_after_emergence_mean, 0)

    return occurrence_date - timedelta(days=interval)

def calculate_phenological_stage(Planting_date, current_date):
    """
    Calculate the phenological stage of a plant based on the planting date and the current date.
    The function uses a aproximation of how much time is necessary for a plant advance in phenological state.
    credits(Karla Braga)
    """
    #diference in days
    delta = current_date - Planting_date
    
    if delta.days < 0:
        return None
    if delta.days == 0: 
        return 0 #plantio
    if delta.days > 5 and delta.days <= 8:
        return 1 #emergencia
    if delta.days > 8 and delta.days <= 13:
        return 2 #v1
    if delta.days > 13 and delta.days <= 24:
        return 3 #v3
    if delta.days > 24 and delta.days <= 29:
        return 4 #v4
    if delta.days > 29 and delta.days <= 38:
        return 5 #v8
    if delta.days > 38 and delta.days <= 58:
        return 6 #r1
    if delta.days > 58 and delta.days <= 68:
        return 7 #r3
    if delta.days > 68 and delta.days <= 93:
        return 8 #r5
    if delta.days > 93 and delta.days <= 113:
        return 9 #r6
    if delta.days > 113 and delta.days <= 128:
        return 10 #r7
    if delta.days > 128 and delta.days <= 158:
        return 11 #r8   
    # baseado no email da Karla para o kemmer com varias aproximações feitas para poder calcular o estagio fenologico de forma 'exata'.
    #0 = plantio, 1= emergencia, 2= v1, 3= v3, 4= v4, 5= v8, 6= r1, 7= r3, 8= r5, 9= r6, 10= r7, 11= r8
    