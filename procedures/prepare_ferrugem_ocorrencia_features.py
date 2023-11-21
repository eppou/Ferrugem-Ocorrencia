import numpy as np
import pandas as pd
from datetime import date, datetime
from sqlalchemy import create_engine, Connection
from calculation.precipitation import calculate_precipitation_acc, calculate_precipitation_count
from calculation.severity import calculate_dsv_30d, calculate_dsv_safra
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
# DONE: Ajustar para intervalos de 15 dias
# TODO: Verificar datas mais precisas das safras?
# DONE: Separar geração das instâncias do cálculo dos features
# DONE: Adicionar índice na busca do vizinho mais próximo no banco
def run(count_limit: int | None = None):
    db_con_engine = create_engine(DB_STRING)
    conn = db_con_engine.connect()

    instances_df = pd.read_csv(OUTPUT_PATH / "instances_dataset.csv")
    safras = get_safras(conn)
    ocorrencias_df = pd.DataFrame()

    count = 0

    for safra in safras:
        if processing_limit_reached(count_limit, count):
            break

        safra_nome = safra["safra"]
        print(f"=====> Processing features for safra {safra_nome}")

        ocorrencias_df_safra = instances_df[instances_df["safra"] == safra_nome].copy()

        # FEATURE CALCULATION
        ocorrencias_df_safra_filtrado = ocorrencias_df_safra[["segment_id_precipitation", "data_ocorrencia"]]
        p15d_list, p30d_list, p45d_list, p60d_list, p75d_list, p90d_list = [], [], [], [], [], []
        pc15d_list, pc30d_list, pc45d_list, pc60d_list, pc75d_list, pc90d_list = [], [], [], [], [], []
        severity_acc_30d_list, severity_acc_safra_list = [], []

        for segment_id_precipitation, data_ocorrencia_str in ocorrencias_df_safra_filtrado.values.tolist():
            segment_id_precipitation = int(segment_id_precipitation)
            data_ocorrencia = None
            if data_ocorrencia_str is not None and isinstance(data_ocorrencia_str, str):
                data_ocorrencia = datetime.strptime(data_ocorrencia_str, "%Y-%m-%d")

            p15, p30, p45, p60, p75, p90 = calculate_precipitation_acc(
                conn,
                segment_id_precipitation,
                safra["planting_start_date"],
                safra["planting_end_date"],
            )
            pc15, pc30, pc45, pc60, pc75, pc90 = calculate_precipitation_count(
                conn,
                segment_id_precipitation,
                safra["planting_start_date"],
                safra["planting_end_date"],
            )

            severity_acc_30d = 0
            severity_acc_safra = 0

            if data_ocorrencia is not None:
                severity_acc_30d = calculate_dsv_30d(conn, segment_id_precipitation, data_ocorrencia)
                severity_acc_safra = calculate_dsv_safra(
                    conn,
                    segment_id_precipitation,
                    safra["planting_start_date"],
                    data_ocorrencia,
                )

            # Storing features
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

            severity_acc_30d_list.append(severity_acc_30d)
            severity_acc_safra_list.append(severity_acc_safra)

        ocorrencias_df_safra["precipitation_15d"] = p15d_list
        ocorrencias_df_safra["precipitation_30d"] = p30d_list
        ocorrencias_df_safra["precipitation_45d"] = p45d_list
        ocorrencias_df_safra["precipitation_60d"] = p60d_list
        ocorrencias_df_safra["precipitation_75d"] = p75d_list
        ocorrencias_df_safra["precipitation_90d"] = p90d_list

        ocorrencias_df_safra["precipitation_15d_count"] = pc15d_list
        ocorrencias_df_safra["precipitation_30d_count"] = pc30d_list
        ocorrencias_df_safra["precipitation_45d_count"] = pc45d_list
        ocorrencias_df_safra["precipitation_60d_count"] = pc60d_list
        ocorrencias_df_safra["precipitation_75d_count"] = pc75d_list
        ocorrencias_df_safra["precipitation_90d_count"] = pc90d_list

        ocorrencias_df_safra["severity_acc_30d"] = severity_acc_30d_list
        ocorrencias_df_safra["severity_acc_safra"] = severity_acc_safra_list

        ocorrencias_df = pd.concat([ocorrencias_df, ocorrencias_df_safra])

        count += 1

    # Output full dataset (possible contain extra information for debugging and visualization)
    ocorrencias_df.reset_index(inplace=True)
    ocorrencias_df.to_csv(OUTPUT_PATH / "instances_features_dataset_all.csv")

    ocorrencias_df = ocorrencias_df
    [[
        "safra", "ocorrencia_latitude", "ocorrencia_longitude",
        "precipitation_15d", "precipitation_30d", "precipitation_45d",
        "precipitation_60d", "precipitation_75d", "precipitation_90d",
        "precipitation_15d_count", "precipitation_30d_count", "precipitation_45d_count",
        "precipitation_60d_count", "precipitation_75d_count", "precipitation_90d_count",
        "ocorrencia"
    ]].copy()
    ocorrencias_df.to_csv(OUTPUT_PATH / "instances_features_dataset.csv")

    conn.close()
    db_con_engine.dispose()


def processing_limit_reached(count_limit, count) -> bool:
    if count_limit is not None:
        if count == count_limit:
            return True

    return False
