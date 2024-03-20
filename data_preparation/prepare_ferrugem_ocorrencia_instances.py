from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine

from calculation.coordinates import find_nearest_segment_id, determine_random_coordinate
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
        instances_ocorrencia_df = pd.read_sql_query(
            QUERY_OCORRENCIAS.replace(":safra", safra["safra"]),
            con=db_con_engine
        )

        instances_df_all = pd.concat([instances_df_all, instances_ocorrencia_df])

    # Assigning a segment_id_precipitation - a match for a position for the nearest precipitation data
    print("=====> Assigning a segment_id_precipitation (for precipitation data)")
    for index, ocorrencia in instances_df_all.iterrows():
        latitude = ocorrencia["ocorrencia_latitude"]
        longitude = ocorrencia["ocorrencia_longitude"]
        print(f"Finding nearest segment for (lat/long) {latitude} {longitude}, index {index}")

        segment_id_precipitation = find_nearest_segment_id(conn, latitude, longitude)
        print(f"Segment found: {segment_id_precipitation}, index {index}")

        instances_df_all.at[index, "segment_id_precipitation"] = segment_id_precipitation

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
