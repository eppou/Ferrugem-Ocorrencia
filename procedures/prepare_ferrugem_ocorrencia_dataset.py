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

# DONE: Criar view para query de ocorrências
# DONE: Determinar datas das safras
# DONE: Calcular precipitação para 15, 30, 45, etc para toda a safra
# DONE: Adicionar contagem de dias de chuva para 15, 30, 45, etc para toda a safra
# DONE: Melhorar algoritmo do chute com distância reduzida
# DONE: Fazer recorte do mapa do Paraná (primeiro filtro)
# DONE: Melhorar algoritmo do chute com mapas da plantação de soja no Paraná (segundo filtro)
# TODO: Ajustar para intervalos de 15 dias
# TODO: Verificar datas mais precisas das safras?
# TODO: Separar geração das instâncias do cálculo dos features
# TODO: Adicionar índice na busca do vizinho mais próximo no banco
def run():
    db_con_engine = create_engine(DB_STRING)
    conn = db_con_engine.connect()

    ocorrencias_df_full_all = pd.DataFrame()

    safras = get_safras(conn)

    for safra in safras:
        print(f"=> Processsing safra {safra['safra']}...")

        # Fetching occurrences from consorcio_antiferrugem database, per harverst
        ocorrencias_df_full = pd.read_sql_query(QUERY_OCORRENCIAS.replace(":safra", safra["safra"]), con=db_con_engine)

        # Randomly generating non-occurrences
        used_coordinates = [
            (x[0], x[1]) for x in ocorrencias_df_full[["ocorrencia_longitude", "ocorrencia_latitude"]].values.tolist()]
        nao_ocorrencias_df_full = create_random_non_occurrences(
            MIN_DISTANCE_FOR_NON_OCCURRENCES,
            len(ocorrencias_df_full.index),
            safra["safra"],
            used_coordinates,
        )
        ocorrencias_df_full = pd.concat([ocorrencias_df_full, nao_ocorrencias_df_full]).reset_index()

        print(f"Size of occurrences dataset: {ocorrencias_df_full.shape[0]}")

        # Assigning a segment_id - a match for a position for the nearest precipitation data
        for index, ocorrencia in ocorrencias_df_full.iterrows():
            latitude = ocorrencia["ocorrencia_latitude"]
            longitude = ocorrencia["ocorrencia_longitude"]
            print(f"Finding nearest segment for (lat/long) {latitude} {longitude}, index {index}")

            segment_id = find_nearest_segment_id(conn, latitude, longitude)
            ocorrencias_df_full.at[index, "segment_id"] = segment_id
            print(f"Segment found: {segment_id}, index {index}")

        # Calculating and storing accumulated precipitation
        # Calculating number of days of precipitation
        # Calculating DSV severity indicator
        segment_id_list = ocorrencias_df_full[["segment_id"]]
        p15d_list, p30d_list, p45d_list, p60d_list, p75d_list, p90d_list = [], [], [], [], [], []
        pc15d_list, pc30d_list, pc45d_list, pc60d_list, pc75d_list, pc90d_list = [], [], [], [], [], []

        for seg_data in segment_id_list.values.tolist():
            segment_id = int(seg_data[0])

            p15, p30, p45, p60, p75, p90 = calculate_precipitation_acc(
                conn,
                segment_id,
                safra["planting_start_date"],
                safra["planting_end_date"],
            )
            pc15, pc30, pc45, pc60, pc75, pc90 = calculate_precipitation_count(
                conn,
                segment_id,
                safra["planting_start_date"],
                safra["planting_end_date"],
            )
            # dsv_30d = calculate_dsv_30d(conn, segment_id, data)

            p15d_list.append(p15)
            p30d_list.append(p30)
            p45d_list.append(p45)
            p60d_list.append(p60)
            p75d_list.append(p75)
            p90d_list.append(p90)

            pc15d_list.append(pc15)
            pc30d_list.append(pc30)
            pc45d_list.append(pc45)
            pc60d_list.append(pc60)
            pc75d_list.append(pc75)
            pc90d_list.append(pc90)

        ocorrencias_df_full["precipitation_15d"] = p15d_list
        ocorrencias_df_full["precipitation_30d"] = p30d_list
        ocorrencias_df_full["precipitation_45d"] = p45d_list
        ocorrencias_df_full["precipitation_60d"] = p60d_list
        ocorrencias_df_full["precipitation_75d"] = p75d_list
        ocorrencias_df_full["precipitation_90d"] = p90d_list

        ocorrencias_df_full["precipitation_15d_count"] = pc15d_list
        ocorrencias_df_full["precipitation_30d_count"] = pc30d_list
        ocorrencias_df_full["precipitation_45d_count"] = pc45d_list
        ocorrencias_df_full["precipitation_60d_count"] = pc60d_list
        ocorrencias_df_full["precipitation_75d_count"] = pc75d_list
        ocorrencias_df_full["precipitation_90d_count"] = pc90d_list

        ocorrencias_df_full_all = pd.concat([ocorrencias_df_full_all, ocorrencias_df_full])

    # Output full dataset (possible contain extra information for debugging and visualization)
    ocorrencias_df_full_all.to_csv(OUTPUT_PATH / "ferrugem_ocorrencia_dataset_full.csv", index=False)

    ocorrencias_df = ocorrencias_df_full_all
    [[
        "safra", "ocorrencia_latitude", "ocorrencia_longitude",
        "precipitation_15d", "precipitation_30d", "precipitation_45d",
        "precipitation_60d", "precipitation_75d", "precipitation_90d",
        "precipitation_15d_count", "precipitation_30d_count", "precipitation_45d_count",
        "precipitation_60d_count", "precipitation_75d_count", "precipitation_90d_count",
        "ocorrencia"
    ]].copy()
    ocorrencias_df.to_csv(OUTPUT_PATH / "ferrugem_ocorrencia_dataset.csv", index=False)

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
        print(f"Coordinate found! Value: (long/lat) (x/y) {coordinate}")

    return pd.DataFrame(data)


def calculate_dsv_30d(conn: Connection, segment_id, target_date: date) -> float:
    pass
