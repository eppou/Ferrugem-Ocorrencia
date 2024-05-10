import random as r
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from calculation.threshold import calculate_threshold_for_baseline_model
from constants import DB_STRING
from source.occurrence import get_safras
from helpers.input_output import output_file, get_latest_file

K_FOLDS = 5
SEED = 492848

"""
Prepara dataset de saída, com as instâncias e features calculadas da severidade acumulada para o threshold determinado
de 5, 10 e 15 dias. O threshold é calculado pela severidade média em 5, 10 e 15 dias antes da data_ocorrencia.
"""


def run(safras: list = None):
    db_con_engine = create_engine(DB_STRING)
    conn = db_con_engine.connect()
    execution_started = datetime.now()

    severity_df = pd.read_csv(get_latest_file("severity", "severity.csv"))
    all_instances_df = pd.read_csv(get_latest_file("features", "features_all.csv"))

    all_instances_df = all_instances_df[all_instances_df["ocorrencia_id"].notnull()]
    all_instances_df = all_instances_df.drop(columns=["level_0", "index", "Unnamed: 0"])

    instances_all_safras_df = pd.DataFrame()

    if safras is None:
        safras = [s['safra'] for s in get_safras(conn)]
        safras = safras[1:]  # Removendo a última safra, pois está muito incompleta

    print("=====> Resultados para CADA safra")
    result_df_all_safras = pd.DataFrame()

    for safra in safras:
        instances_df = all_instances_df.copy()

        print(f"- Filtrando por Safra: {safra}")
        instances_safra_df = instances_df[instances_df["safra"] == safra]
        instances_all_safras_df = pd.concat([instances_all_safras_df, instances_safra_df])

        print("- Preparando folds (k=5)")
        folds = prepare_folds(instances_safra_df, K_FOLDS)

        k_num = 0
        result_df_all_folds = pd.DataFrame()
        for k in folds:
            k_num += 1
            print(f"- Determinando resultados para fold {k_num}/{K_FOLDS}")

            train_indices = k[0]
            test_indices = k[1]
            train_x, train_y, test_x, test_y = prepare_train_test_for_fold(instances_safra_df, train_indices,
                                                                           test_indices)

            result_df = prepare_severity_model_results(train_x, train_y, test_x, test_y, severity_df, safra, k_num)

            result_df_all_folds = pd.concat([result_df_all_folds, result_df])

        write_result(execution_started, result_df_all_folds, safra)

        result_df_all_safras = pd.concat([result_df_all_safras, result_df_all_folds])
        print("\n\n")

    write_result(execution_started, result_df_all_safras, "all")

    print("=====> Resultados considerando TODAS as safras juntas")
    instances_df = instances_all_safras_df.copy()

    print("- Preparando folds (k=5)")
    folds = prepare_folds(instances_df, K_FOLDS)

    k_num = 0
    result_df_all_folds = pd.DataFrame()
    for k in folds:
        k_num += 1
        print(f"- Determinando resultados para fold {k_num}/{K_FOLDS}")

        train_indices = k[0]
        test_indices = k[1]
        train_x, train_y, test_x, test_y = prepare_train_test_for_fold(instances_df, train_indices, test_indices)

        result_df = prepare_severity_model_results(train_x, train_y, test_x, test_y, severity_df, None, k_num)

        result_df_all_folds = pd.concat([result_df_all_folds, result_df])

    write_result(execution_started, result_df_all_folds, None)


def prepare_severity_model_results(
        train_x: pd.DataFrame,
        train_y: pd.DataFrame,
        test_x: pd.DataFrame,
        test_y: pd.DataFrame,
        severity_df: pd.DataFrame,
        safra: str | None,
        fold_number: int,
):
    # O train_y não é utilizado, pois o "treino" dos resultados é apenas o cálculo destes thresholds
    threshold_5d, threshold_10d, threshold_15d = calculate_threshold_for_baseline_model(train_x)

    test_x.drop(columns=["severity_acc_5d_before_occurrence", "severity_acc_10d_before_occurrence",
                         "severity_acc_15d_before_occurrence"], inplace=True)

    total_values_for_testing = test_x.shape[0]
    value_test_count = 0
    print(f"=====> Testando com {total_values_for_testing} valores")

    severity_acc_5d_list, severity_acc_10d_list, severity_acc_15d_list = [], [], []
    predicted_planting_relative_day_5d_list, predicted_planting_relative_day_10d_list, predicted_planting_relative_day_15d_list = [], [], []

    # determining the planting_relative_day values
    for index, instance in test_x.iterrows():
        value_test_count += 1

        occurrence_id = instance["ocorrencia_id"]
        print(
            f"=====> [{value_test_count}/{total_values_for_testing}] {safra if safra is not None else ""} PARA OCORRÊNCIA COM: occurrence_id: {occurrence_id}")

        severity_acc_5d, predicted_planting_relative_day_5d = determine_planting_relative_day_from_threshold(
            severity_df, occurrence_id, threshold_5d, 5)
        severity_acc_10d, predicted_planting_relative_day_10d = determine_planting_relative_day_from_threshold(
            severity_df, occurrence_id, threshold_10d, 10)
        severity_acc_15d, predicted_planting_relative_day_15d = determine_planting_relative_day_from_threshold(
            severity_df, occurrence_id, threshold_15d, 15)

        print(
            f"\t- For threshold_5d: {threshold_5d} "
            f"| Severity (accumulated): {severity_acc_5d} "
            f"| Predicted planting_relative_day: {predicted_planting_relative_day_5d}"
        )

        print(
            f"\t- For threshold_10d: {threshold_10d} "
            f"| Severity (accumulated): {severity_acc_10d} "
            f"| Predicted planting_relative_day: {predicted_planting_relative_day_10d}"
        )

        print(
            f"\t- For threshold_15d: {threshold_15d} "
            f"| Severity (accumulated): {severity_acc_15d} "
            f"| Predicted planting_relative_day: {predicted_planting_relative_day_15d}"
        )

        severity_acc_5d_list.append(severity_acc_5d)
        predicted_planting_relative_day_5d_list.append(predicted_planting_relative_day_5d)

        severity_acc_10d_list.append(severity_acc_10d)
        predicted_planting_relative_day_10d_list.append(predicted_planting_relative_day_10d)

        severity_acc_15d_list.append(severity_acc_15d)
        predicted_planting_relative_day_15d_list.append(predicted_planting_relative_day_15d)

    result_df = test_x.copy()
    result_df['planting_relative_day'] = test_y['planting_relative_day']

    result_df["threshold_5d"] = threshold_5d
    result_df["severity_acc_5d"] = severity_acc_5d_list
    result_df["predicted_planting_relative_day_5d"] = predicted_planting_relative_day_5d_list

    result_df["threshold_10d"] = threshold_10d
    result_df["severity_acc_10d"] = severity_acc_10d_list
    result_df["predicted_planting_relative_day_10d"] = predicted_planting_relative_day_10d_list

    result_df["threshold_15d"] = threshold_15d
    result_df["severity_acc_15d"] = severity_acc_15d_list
    result_df["predicted_planting_relative_day_15d"] = predicted_planting_relative_day_15d_list

    if safra is not None:
        result_df["safra"] = [safra] * result_df.shape[0]

    result_df["fold"] = fold_number

    result_df.reset_index(inplace=True)

    return result_df


def determine_planting_relative_day_from_threshold(severity_df: pd.DataFrame, occurrence_id, threshold,
                                                  threshold_days: int) -> tuple:
    df = severity_df

    df = df[df["occurrence_id"] == occurrence_id]
    df = df[df["severity_acc"] <= threshold]
    df = df.loc[df["severity_acc"].idxmax()]

    return df["severity_acc"], (df["planting_relative_day"] + threshold_days)


def write_result(execution_started: datetime, result_df: pd.DataFrame, safra: str | None):
    base_filename = "concorrente"
    filename = ""
    if safra is None:
        filename = f"{base_filename}_results_all.csv"

    if safra is not None and safra.lower() == "all":
        filename = f"{base_filename}_results_harvest_all.csv"
    elif safra is not None:
        filename = f"{base_filename}_results_harvest_{safra.replace("/", "_")}.csv"

    result_df.to_csv(output_file(execution_started, "concorrente", filename))


def prepare_folds(df: pd.DataFrame, k: int) -> list[tuple]:
    data_index = df.index.to_list()
    r.Random(SEED).shuffle(data_index)

    fold_size = len(data_index) // k
    folds = []
    for i in range(k):
        train_indices = np.concatenate([data_index[:i * fold_size], data_index[(i + 1) * fold_size:]])
        test_indices = data_index[i * fold_size: (i + 1) * fold_size]

        folds.append((train_indices, test_indices))

    return folds


def prepare_train_test_for_fold(df: pd.DataFrame, train_indices, test_indices) -> tuple:
    """Prepare train and test datasets, x and y. Return: train_x, train_y, test_x, test_y"""

    train_df = df.filter(items=train_indices, axis=0)
    test_df = df.filter(items=test_indices, axis=0)

    train_x = train_df[[
        "ocorrencia_id", "segment_id_precipitation", "planting_start_date",
        "severity_acc_5d_before_occurrence", "severity_acc_10d_before_occurrence", "severity_acc_15d_before_occurrence",
    ]]
    train_y = train_df[["planting_relative_day"]].astype(int)

    test_x = test_df[[
        "ocorrencia_id", "segment_id_precipitation", "planting_start_date",
        "severity_acc_5d_before_occurrence", "severity_acc_10d_before_occurrence", "severity_acc_15d_before_occurrence",
    ]]
    test_y = test_df[["planting_relative_day"]].astype(int)

    return train_x, train_y, test_x, test_y
