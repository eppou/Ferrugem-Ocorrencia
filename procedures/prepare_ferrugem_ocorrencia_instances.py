import pandas as pd
from datetime import date
from sqlalchemy import create_engine, Connection
from calculation.precipitation import calculate_precipitation_acc, calculate_precipitation_count
from calculation.coordinates import find_nearest_segment_id, determine_random_coordinate

from constants import DB_STRING, OUTPUT_PATH
from procedures.constants import (
    QUERY_OCORRENCIAS,
    MIN_DISTANCE_FOR_NON_OCCURRENCES
)
from source.occurrence import get_safras

"""
Main pipeline to create the dataset with Soybean rust occurrences.
"""


# 1. Coletar todas as ocorrências por safra. Calcular features por data de ocorrência.
# 2. Contar ocorrências e gerar não-ocorrencias por safra. Método: Sorteio. Definir distancia = 2 graus em duas direções
# 3. Para cada não-ocorrência, usar dados da safra para gerar features. Chutar um intervalo entre o começo e fim da
# safra como "data de verificado como não ocorrência". Calcular features.

def run():
    db_con_engine = create_engine(DB_STRING)
    conn = db_con_engine.connect()

    instances_df_all = pd.DataFrame()

    safras = get_safras(conn)

    for safra in safras:
        # INSTANCE DETERMINATION
        print(f"=====> Processsing safra {safra['safra']}...")

        # Fetching occurrences from consorcio_antiferrugem database, per harverst
        instances_ocorrencia_df = pd.read_sql_query(QUERY_OCORRENCIAS.replace(":safra", safra["safra"]), con=db_con_engine)

        # Randomly generating non-occurrences
        used_coordinates = [
            (x[0], x[1]) for x in instances_ocorrencia_df[["ocorrencia_longitude", "ocorrencia_latitude"]].values.tolist()]
        instances_nao_ocorrencia_df = create_random_non_occurrences(
            MIN_DISTANCE_FOR_NON_OCCURRENCES,
            len(instances_ocorrencia_df.index),
            safra["safra"],
            used_coordinates,
        )
        instances_ocorrencia_df = pd.concat([instances_ocorrencia_df, instances_nao_ocorrencia_df]).reset_index()

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
    instances_df_all.to_csv(OUTPUT_PATH / "instances_dataset_all.csv", index=False)

    instances_df = instances_df_all
    [[
        "safra", "ocorrencia_latitude", "ocorrencia_longitude", "ocorrencia"
    ]].copy()
    instances_df.to_csv(OUTPUT_PATH / "instances_dataset.csv", index=False)

    conn.close()
    db_con_engine.dispose()


def create_random_non_occurrences(
        min_distance: float,
        count: int,
        safra: str,
        used_coordinates: list[tuple[float, float]]  # x, y => long, lat
):
    # Calcular ocorrências aleatórias. Quantidade especificada no parâmetro. Distância especificada no parâmetro.
    # Chutar um dia dentro da safra para "data de não ocorrencia"

    # NOTA: É importante calcular coordenadas aleatórias por safra e não global, pois estamos analizando os eventos
    # de cada safra separadamente. Portanto, used_coordinates é resetado para cada invocação desta função.

    data: list[dict] = []

    for x in range(count):
        coordinate = determine_random_coordinate(used_coordinates, min_distance)
        used_coordinates.append(coordinate)

        data.append({
            'ocorrencia_id': '',
            'data': '',
            'safra': safra,
            'cidade_nome': '',
            'estado_nome': '',
            'ocorrencia_localizacao': '',
            'ocorrencia_latitude': coordinate[1],   # y => Latitude
            'ocorrencia_longitude': coordinate[0],  # x => Longitude
            'ocorrencia': False,
        })
        print(f"=====> Coordinate found! Value: (long/lat) (x/y) {coordinate}")

    return pd.DataFrame(data)


def calculate_dsv_30d(conn: Connection, segment_id, target_date: date) -> float:
    pass

