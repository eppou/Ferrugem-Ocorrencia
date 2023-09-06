import pandas as pd
from datetime import date, datetime
from sqlalchemy import create_engine, Connection, text

from constants import DB_STRING, OUTPUT_PATH
from procedures.constants import QUERY_OCORRENCIAS, QUERY_PRECIPITATION_SEGMENTS, QUERY_PRECIPITATION_ACC
from calculation import coordinates, dates

"""
Main pipeline to create the dataset with Soybean rust occurrences.
"""

SAFRAS = ["2017/2018", "2018/2019", "2019/2020"]


# 1. Coletar todas as ocorrências por safra. Calcular features por data de ocorrência.
# 2. Contar ocorrências e gerar não-ocorrencias por safra. Método: Sorteio. Definir distancia = 2 graus em duas direções
# 3. Para cada não-ocorrência, usar dados da safra para gerar features. Chutar um intervalo entre o começo e fim da
# safra como "data de verificado como não ocorrência". Calcular features.
def run():
    db_con_engine = create_engine(DB_STRING)
    conn = db_con_engine.connect()

    ocorrencias_df_full_all = pd.DataFrame()

    for safra in SAFRAS:
        # Fetching all occurrences from consorcio_antiferrugem database
        ocorrencias_df_full = pd.read_sql_query(QUERY_OCORRENCIAS.replace(":safra", safra), con=db_con_engine)

        # Randomly generating non-occurrences
        nao_ocorrencias_df_full = create_random_non_occurrences(1.5, len(ocorrencias_df_full.index))
        ocorrencias_df_full = pd.concat([ocorrencias_df_full, nao_ocorrencias_df_full])

        # Assigning a segment_id - a match for a position for the nearest precipitation data
        for index, ocorrencia in ocorrencias_df_full.iterrows():
            latitude = ocorrencia["ocorrencia_latitude"]
            longitude = ocorrencia["ocorrencia_longitude"]

            ocorrencias_df_full.at[index, "segment_id"] = find_nearest_segment_id(conn, latitude, longitude)

        # Calculating and storing accumulated precipitation
        segment_id_data_list = ocorrencias_df_full[["segment_id", "data"]]
        precipitation_7d_list, precipitation_14d_list, precipitation_30d_list = [], [], []
        for seg_data in segment_id_data_list.values.tolist():
            segment_id, data = seg_data
            p7d, p14d, p30d = calculate_precipitation_acc(conn, segment_id, data)
            precipitation_7d_list.append(p7d)
            precipitation_14d_list.append(p14d)
            precipitation_30d_list.append(p30d)

        ocorrencias_df_full["precipitation_7d"] = precipitation_7d_list
        ocorrencias_df_full["precipitation_14d"] = precipitation_14d_list
        ocorrencias_df_full["precipitation_30d"] = precipitation_30d_list

        ocorrencias_df_full_all = pd.concat([ocorrencias_df_full_all, ocorrencias_df_full])

    # Output full dataset (possible contain extra information for debugging and visualization)
    ocorrencias_df_full_all.to_csv(OUTPUT_PATH / "ferrugem_ocorrencia_dataset_full.csv", index=False)

    ocorrencias_df = ocorrencias_df_full_all[
        ["data", "ocorrencia_latitude", "ocorrencia_longitude", "ocorrencia"]].copy()
    ocorrencias_df.to_csv(OUTPUT_PATH / "ferrugem_ocorrencia_dataset.csv", index=False)

    conn.close()
    db_con_engine.dispose()


def create_random_non_occurrences(distance: float, count: int):
    # Calcular ocorrências aleatórias. Quantidade especificada no parâmetro. Distância especificada no parâmetro.
    # Chutar um dia dentro da safra para "data de não ocorrencia"

    # NOTA: É importante calcular coordenadas aleatórias por safra e não global, pois estamos analizando os eventos
    # de cada safra separadamente. Portanto, used_coordinates é resetado para cada invocação desta função.

    # TODO: Qual é a data da safra?

    used_coordinates = []
    data = []
    for x in range(count):
        coordinate = coordinates.determine_random_coordinates(used_coordinates, distance)
        used_coordinates.append(coordinate)

        data.append({
            'ocorrencia_id': '',
            'data': dates.determine_random_date(date(2023, 5, 23), date(2023, 8, 11)),
            'cidade_nome': '',
            'estado_nome': '',
            'ocorrencia_localizacao': '',
            'ocorrencia_latitude': coordinate[0],
            'ocorrencia_longitude': coordinate[1],
            'ocorrencia': False,
        })

    return pd.DataFrame(data)


def find_nearest_segment_id(conn: Connection, lat, long) -> int:
    result = conn.execute(
        text(QUERY_PRECIPITATION_SEGMENTS.replace(":latitude", str(lat)).replace(":longitude", str(long)))
    ).fetchone()

    return int(result.t[0])


def calculate_precipitation_acc(conn: Connection, segment_id, target_date: date) -> tuple:
    target_date_str = target_date.strftime("%Y-%m-%d")
    result = conn.execute(
        text(QUERY_PRECIPITATION_ACC.replace(":segment_id", str(segment_id)).replace(":target_date", target_date_str))
    ).fetchone()

    return result.t
