from datetime import datetime, date

import pandas as pd
from sqlalchemy import create_engine

from calculation.coordinates import find_nearest_segment_id
from constants import DB_STRING
from data_preparation.constants import QUERY_OCORRENCIAS
from helpers.input_output import output_file
from source.occurrence import get_safras


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

        df = df[df["ocorrencia_latitude"].notna()]
        df = df[df["ocorrencia_longitude"].notna()]
        df = df[df["data_ocorrencia"].dt.year >= 2000]  # Filtrando resultados incorretos no formato '0001-XX-XX'

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

        instances_df_all.at[index, "segment_id_precipitation"] = segment_id_precipitation

        harvest_start_date = get_harvest_start_date(safras, instance["safra"])
        print(f"Add harvest_start_date: {harvest_start_date.strftime("%Y-%m-%d")=}, {index=}")
        instances_df_all.at[index, "harvest_start_date"] = harvest_start_date

    print(f"=====> Size of instances dataset: {instances_df_all.shape[0]}")

    # Output full dataset (possible contain extra information for debugging and visualization)
    instances_df_all.to_csv(output_file(
        execution_started,
        "prepare_occurrence_instances",
        "instances_dataset_all.csv"
    ), index=False)

    instances_df = instances_df_all
    [[
        "safra", "ocorrencia_latitude", "ocorrencia_longitude", "ocorrencia"
    ]].copy()
    instances_df.to_csv(output_file(
        execution_started,
        "prepare_occurrence_instances",
        "instances_dataset.csv"
    ), index=False)

    conn.close()
    db_con_engine.dispose()


def get_harvest_start_date(safras: list, safra_current: str) -> date:
    safras_indexed = {}
    for safra in safras:
        safras_indexed[safra["safra"]] = safra

    if safra_current not in safras_indexed.keys():
        raise RuntimeError(f"Safra is not present in safra information from database {safra_current=}")

    return safras_indexed[safra_current]["planting_start_date"]
