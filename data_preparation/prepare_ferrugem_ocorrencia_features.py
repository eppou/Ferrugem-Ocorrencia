from datetime import datetime, date, timedelta

import pandas as pd
from sqlalchemy import create_engine, text

from constants import DB_STRING, MAX_HARVEST_RELATIVE_DAY
from data_preparation.constants import QUERY_PRECIPITATION_FOR_ALL_HARVESTS
from helpers.input_output import get_latest_file, output_file
from source.occurrence import get_safras


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
# TODO: Verificar datas mais precisas das safras? Se não, usar as mesmas para todas as safras
# TODO: Fazer o treinamento e processamento dos datasets com todas as ocorrencias até 2005
# TODO: Fazer uma análise para verificar se o resultado com partes do dataset é melhor do que com todo dataset
# TODO: Corrigir cálculo do erro
# DONE: Separar geração das instâncias do cálculo dos features
# DONE: Adicionar índice na busca do vizinho mais próximo no banco
def run(count_limit: int | None = None):
    execution_start = datetime.now()

    db_con_engine = create_engine(DB_STRING)
    conn = db_con_engine.connect()

    precipitation_df = pd.read_sql_query(sql=text(QUERY_PRECIPITATION_FOR_ALL_HARVESTS), con=conn, parse_dates=["date_precipitation"])
    severity_df = pd.read_csv(get_latest_file("prepare_severity_per_occurrence", "severity_per_occurrence.csv"))
    instances_df = pd.read_csv(get_latest_file("prepare_occurrence_instances", "instances_dataset.csv"), parse_dates=["data_ocorrencia"])
    safras = get_safras(conn)
    ocorrencias_df = pd.DataFrame()

    count = 0

    for safra in safras:
        if processing_limit_reached(count_limit, count):
            break

        safra_nome = safra["safra"]
        print(f"=====> Processing features for safra {safra_nome}")

        ocorrencias_df_safra = instances_df[instances_df["safra"] == safra_nome].copy()
        ocorrencias_df_safra = ocorrencias_df_safra[ocorrencias_df_safra["ocorrencia_id"].notnull()]

        instances_count = 0
        for index, instance in ocorrencias_df_safra.iterrows():
            instances_count += 1
            ocorrencias_df_safra_generated = instance.copy().to_frame().T
            print(f"=====> Progress [{instances_count}/{ocorrencias_df_safra.shape[0]}]")

            harvest_start_date = safra["planting_start_date"]
            segment_id_precipitation = instance["segment_id_precipitation"]
            occurrence_id = instance["ocorrencia_id"]
            occurrence_date = instance["data_ocorrencia"]

            precipitation_features = calculate_precipitation_all_harvest_days(
                precipitation_df,
                segment_id_precipitation,
                harvest_start_date,
            )

            precipitation_features_df = pd.DataFrame(precipitation_features, index=[index])
            ocorrencias_df_safra_generated = pd.merge(
                ocorrencias_df_safra_generated, precipitation_features_df, how="outer", left_index=True, right_index=True)

            harvest_relative_day = (occurrence_date.date() - safra["planting_start_date"]).days
            ocorrencias_df_safra_generated["harvest_relative_day"] = harvest_relative_day
            ocorrencias_df_safra_generated["harvest_start_date"] = harvest_start_date

            severity_acc_5d_before_occurrence = calculate_severity(
                severity_df,
                occurrence_id,
                harvest_relative_day - 5,
            )
            severity_acc_10d_before_occurrence = calculate_severity(
                severity_df,
                occurrence_id,
                harvest_relative_day - 10,
            )
            severity_acc_15d_before_occurrence = calculate_severity(
                severity_df,
                occurrence_id,
                harvest_relative_day - 15,
            )

            ocorrencias_df_safra_generated["severity_acc_5d_before_occurrence"] = severity_acc_5d_before_occurrence
            ocorrencias_df_safra_generated["severity_acc_10d_before_occurrence"] = severity_acc_10d_before_occurrence
            ocorrencias_df_safra_generated["severity_acc_15d_before_occurrence"] = severity_acc_15d_before_occurrence

            ocorrencias_df = pd.concat([ocorrencias_df, ocorrencias_df_safra_generated])

    # Output full dataset (possible contain extra information for debugging and visualization)
    ocorrencias_df.reset_index(inplace=True)
    ocorrencias_df.to_csv(
        output_file(execution_start, "prepare_occurrence_features", "instances_features_dataset_all.csv"))

    ocorrencias_df = ocorrencias_df.filter(axis=1, regex="(safra|ocorrencia|precipitation)").copy()
    ocorrencias_df.to_csv(
        output_file(execution_start, "prepare_occurrence_features", "instances_features_dataset.csv"))

    conn.close()
    db_con_engine.dispose()


def processing_limit_reached(count_limit, count) -> bool:
    if count_limit is not None:
        if count == count_limit:
            return True

    return False


def calculate_precipitation_all_harvest_days(
        precipitation_df: pd.DataFrame,
        segment_id_precipitation,
        harvest_start_date: date,
) -> dict:
    df = precipitation_df
    df = df[df["segment_id"] == segment_id_precipitation]
    df = df[df["prec"] > 0.5]

    current_harvest_relative_day = 7
    harvest_start_date = pd.to_datetime(harvest_start_date)
    current_date = pd.to_datetime(harvest_start_date + timedelta(days=current_harvest_relative_day))
    precipitation_features = {}

    while current_harvest_relative_day <= MAX_HARVEST_RELATIVE_DAY:
        filtered_df = df[(df["date_precipitation"] >= harvest_start_date) & (df["date_precipitation"] <= current_date)]

        precipitation_acc = filtered_df["prec"].sum()
        precipitation_count = filtered_df["prec"].count()

        precipitation_features[f"precipitation_acc_{current_harvest_relative_day}d"] = precipitation_acc
        precipitation_features[f"precipitation_count_{current_harvest_relative_day}d"] = precipitation_count

        current_harvest_relative_day += 7
        current_date = pd.to_datetime(harvest_start_date + timedelta(days=current_harvest_relative_day))

    return precipitation_features


def calculate_severity(
        severity_df: pd.DataFrame,
        occurrence_id,
        harvest_relative_day,
) -> dict:
    df = severity_df
    df = df[df["occurrence_id"] == occurrence_id]
    df = df[df["harvest_relative_day"] == harvest_relative_day]

    return df["severity_acc"].array[0]
