from datetime import datetime, date, timedelta
from time import sleep

import pandas as pd
from sqlalchemy import create_engine, text

from constants import DB_STRING, MAX_PLANTING_RELATIVE_DAY, FEATURE_DAY_INTERVAL
from data_preparation.constants import QUERY_PRECIPITATION
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
# DONE: Verificar datas mais precisas das safras? Se não, usar as mesmas para todas as safras (como: Utilizando dia relativo da safra para estádios fenológicos)
# DONE: Fazer o treinamento e processamento dos datasets com todas as ocorrencias até 2005
# TODO: Fazer uma análise para verificar se o resultado com partes do dataset é melhor do que com todo dataset
# DONE: Corrigir cálculo do erro
# DONE: Separar geração das instâncias do cálculo dos features
# DONE: Adicionar índice na busca do vizinho mais próximo no banco
# TODO: Verificar se dataset de precipitacao nao contem duplicados
# TODO: Verificar se os dados para as ocorrências estão presentes no intervalo de datas das safras (harvest_start_date e harvest_end_date)
def run(execution_started_at: datetime, harvest: list = None):
    db_con_engine = create_engine(DB_STRING)
    conn = db_con_engine.connect()

    chunk_size = 10000  # Número de linhas por bloco

    # Inicializar uma lista para armazenar os blocos
    chunks = []

    # Usar cursor server-side para streaming de resultados sem fechar a conexão
    result = conn.execution_options(stream_results=True).execute(text(QUERY_PRECIPITATION))
    columns = result.keys()

    # Ler dados em blocos
    while True:
        rows = result.fetchmany(chunk_size)
        if not rows:
            break
        chunks.append(pd.DataFrame(rows, columns=columns))

    # Concatenar os blocos em um único DataFrame
    precipitation_df = pd.concat(chunks, ignore_index=True)
        
    print("=====> Processing features")
    severity_df = pd.read_csv(get_latest_file("severity", "severity.csv"))
    instances_df = pd.read_csv(
        get_latest_file("instances", "instances_all.csv"),
        parse_dates=["data_ocorrencia", "planting_start_date"],
    )
    
    if harvest is None:
        harvest = [s['safra'] for s in get_safras(conn)]
        # harvest = harvest[1:]  # Removendo a última safra, pois está muito incompleta

    ocorrencias_df = pd.DataFrame()

    for harvest in harvest:
        print(f"=====> Processing features for harvest {harvest}")

        occurrences_df_safra = instances_df[instances_df["safra"] == harvest].copy()
        occurrences_df_safra = occurrences_df_safra[occurrences_df_safra["ocorrencia_id"].notnull()]

        instances_count = 0
        for index, instance in occurrences_df_safra.iterrows():
            instances_count += 1
            ocorrencias_df_safra_generated = instance.copy().to_frame().T
            print(f"=====> Progress {harvest} [{instances_count}/{occurrences_df_safra.shape[0]}]")

            planting_start_date = instance["planting_start_date"]
            segment_id_precipitation = instance["segment_id_precipitation"]
            occurrence_id = instance["ocorrencia_id"]
            occurrence_date = instance["data_ocorrencia"]

            precipitation_features = calculate_precipitation_all_planting_days(
                precipitation_df,
                segment_id_precipitation,
                planting_start_date,
                occurrence_date,
            )

            precipitation_features_df = pd.DataFrame(precipitation_features, index=[index])
            ocorrencias_df_safra_generated = pd.merge(
                ocorrencias_df_safra_generated, precipitation_features_df, how="outer", left_index=True,
                right_index=True)

            planting_relative_day = calculate_planting_relative_day(instance)
            ocorrencias_df_safra_generated["planting_relative_day"] = planting_relative_day
            ocorrencias_df_safra_generated["planting_start_date"] = planting_start_date

            severity_acc_5d_before_occurrence = calculate_severity(
                severity_df,
                occurrence_id,
                planting_relative_day - 5,
            )
            severity_acc_10d_before_occurrence = calculate_severity(
                severity_df,
                occurrence_id,
                planting_relative_day - 10,
            )
            severity_acc_15d_before_occurrence = calculate_severity(
                severity_df,
                occurrence_id,
                planting_relative_day - 15,
            )

            ocorrencias_df_safra_generated["severity_acc_5d_before_occurrence"] = severity_acc_5d_before_occurrence
            ocorrencias_df_safra_generated["severity_acc_10d_before_occurrence"] = severity_acc_10d_before_occurrence
            ocorrencias_df_safra_generated["severity_acc_15d_before_occurrence"] = severity_acc_15d_before_occurrence

            # TODO: Feature: Média de severidade diária até o dia da ocorrência (severidade_acc_d / harvest_relative_day)
            # TODO: Feature: Média de severidade diária até o dia da ocorrência (severidade_acc_d / harvest_relative_day) para todas as instâncias da mesma safra
            # TODO: Feature: Mediana da severidade diária até o dia da ocorrência
            # TODO: Feature: Desvio-padrão da severidade diária até o dia da ocorrência
            # TODO: Feature: Desvio-padrão da severidade diária até 30 dias antes da ocorrência

            # TODO: Feature: Média do harvest_relative_day para a safra

            # TODO: Feature: Percentual de variacão de chuva nos últimos 30 dias, normalizado para (0,1) (para todas as instâncias da mesma safra)

            # TODO: Zerar severidade acumuladas para dias após o dia da ocorrência?
            # DONE: Zerar precipitação acumuladas para dias após o dia da ocorrência?

            # TODO?: Features: Features para estádios fenológico (V1, V2... R1...). Cada um seria uma coluna. Agora, incluir precipitation_acc e count.
            # DONE: Calcular features baseados em planting_start_date ao invés de harvest_relative_day
            # DONE: Feature: planting_relative_day: Dia relativo ao inicio da safra quando foi plantado a safra para aquela ocorrência. Calcular pelo planting_start_date (que é calculado pelo estadio fenológico).
            # DONE: Renomear: harvest_relative_day para planting_relative_day

            ocorrencias_df = pd.concat([ocorrencias_df, ocorrencias_df_safra_generated])

    # Output full dataset (possible contain extra information for debugging and visualization)
    ocorrencias_df.reset_index(inplace=True)
    ocorrencias_df = ocorrencias_df.drop(columns=["level_0"])
    ocorrencias_df.to_csv(
        output_file(execution_started_at, "features", "features_all.csv")
    )

    ocorrencias_df = ocorrencias_df.filter(axis=1, regex="(safra|ocorrencia|precipitation)").copy()
    ocorrencias_df.to_csv(
        output_file(execution_started_at, "features", "features.csv")
    )

    conn.close()
    db_con_engine.dispose()


def processing_limit_reached(count_limit, count) -> bool:
    if count_limit is not None:
        if count == count_limit:
            return True

    return False


# TODO: Aprimorar chute, ao invés de usar média, usar uma relação entre os dias da safra esperados por grupo relativo
# DONE: Calcular assim: occurrence_date - planting_start_date
def calculate_planting_relative_day(instance: pd.Series) -> int:
    occurrence_date = instance["data_ocorrencia"]
    planting_start_date = instance["planting_start_date"]

    return (occurrence_date - planting_start_date).days


def calculate_precipitation_all_planting_days(
        precipitation_df: pd.DataFrame,
        segment_id_precipitation,
        planting_start_date: date,
        occurrence_date: date,
) -> dict:
    df = precipitation_df
    df = df[df["segment_id"] == segment_id_precipitation]
    df = df[df["prec"] > 0.5]

    # Converter a coluna 'date_precipitation' para datetime
    df["date_precipitation"] = pd.to_datetime(df["date_precipitation"])

    current_planting_relative_day = FEATURE_DAY_INTERVAL
    planting_start_date = pd.to_datetime(planting_start_date)
    current_date = pd.to_datetime(planting_start_date + timedelta(days=current_planting_relative_day))
    precipitation_features = {}

    while current_planting_relative_day <= MAX_PLANTING_RELATIVE_DAY:
        filtered_df = df[
            (df["date_precipitation"] >= planting_start_date) & 
            (df["date_precipitation"] <= current_date)
        ]

        precipitation_acc = filtered_df["prec"].sum()
        precipitation_count = filtered_df["prec"].count()

        # TODO: adicionar _diff
        precipitation_features[f"precipitation_acc_{current_planting_relative_day}d"] = precipitation_acc
        precipitation_features[f"precipitation_count_{current_planting_relative_day}d"] = precipitation_count

        current_planting_relative_day += FEATURE_DAY_INTERVAL
        current_date = pd.to_datetime(planting_start_date + timedelta(days=current_planting_relative_day))

    return precipitation_features


def calculate_severity(
        severity_df: pd.DataFrame,
        occurrence_id,
        planting_relative_day,
) -> dict:
    if planting_relative_day < 0:
        print(
            f"=====> calculate_severity for {occurrence_id=} and {planting_relative_day=}: "
            f"planting_relative_day is NEGATIVE. Fallback value is 0."
        )
        sleep(0.5)
        planting_relative_day = 0

    df = severity_df
    df = df[df["occurrence_id"] == occurrence_id]
    df = df[df["planting_relative_day"] == planting_relative_day]

    if len(df["severity_acc"].array) == 0:
        raise RuntimeError(
            f"Severity Accumulated (severity_acc) not found for {occurrence_id=} and {planting_relative_day=}")

    severity = df["severity_acc"].array[0]

    return severity
