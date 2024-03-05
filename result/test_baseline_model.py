from random import shuffle

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from calculation.threshold import calculate_threshold_for_baseline_model
from constants import DB_STRING, OUTPUT_PATH
from source.occurrence import get_safras

K_FOLDS = 5

"""
Prepara dataset de saída, com as instâncias e features calculadas da severidade acumulada para o threshold determinado
de 5, 10 e 15 dias. O threshold é calculado pela severidade média em 5, 10 e 15 dias antes da data_ocorrencia.
"""


def run():
    db_con_engine = create_engine(DB_STRING)
    conn = db_con_engine.connect()

    severity_df = pd.read_csv(OUTPUT_PATH / "severity_per_occurrence.csv")
    all_instances_df = pd.read_csv(OUTPUT_PATH / "instances_features_dataset.csv")
    all_instances_df = all_instances_df[all_instances_df["ocorrencia_id"].notnull()]
    all_instances_df = all_instances_df.drop(columns=["level_0", "index", "Unnamed: 0"])

    safras = [s['safra'] for s in get_safras(conn)]
    safras = safras[1:]  # Removendo a última safra, pois está muito incompleta

    print("=====> Resultados para CADA safra")
    result_df_all_safras = pd.DataFrame()

    for safra in safras:
        instances_df = all_instances_df.copy()

        print(f"- Filtrando por Safra: {safra}")
        instances_safra_df = instances_df[instances_df["safra"] == safra]

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

        write_result(result_df_all_folds, safra)

        result_df_all_safras = pd.concat([result_df_all_safras, result_df_all_folds])
        print("\n\n")

    write_result(result_df_all_safras, "all")

    print("=====> Resultados considerando TODAS as safras juntas")
    instances_df = all_instances_df.copy()

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

    write_result(result_df_all_folds, None)


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

    test_x.drop(columns=["severity_acc_safra_5d", "severity_acc_safra_10d", "severity_acc_safra_15d"], inplace=True)

    total_values_for_testing = test_x.shape[0]
    value_test_count = 0
    print(f"=====> Testando com {total_values_for_testing} valores")

    severity_acc_5d_list, severity_acc_10d_list, severity_acc_15d_list = [], [], []
    predicted_harvest_relative_day_5d_list, predicted_harvest_relative_day_10d_list, predicted_harvest_relative_day_15d_list = [], [], []

    # determining the day_in_harvest values
    for index, instance in test_x.iterrows():
        value_test_count += 1

        occurrence_id = instance["ocorrencia_id"]
        print(
            f"=====> [{value_test_count}/{total_values_for_testing}] {safra if safra is not None else ""} PARA OCORRÊNCIA COM: occurrence_id: {occurrence_id}")

        severity_acc_5d, predicted_harvest_relative_day_5d = determine_harvest_relative_day_from_threshold(
            severity_df, occurrence_id, threshold_5d)
        severity_acc_10d, predicted_harvest_relative_day_10d = determine_harvest_relative_day_from_threshold(
            severity_df, occurrence_id, threshold_10d)
        severity_acc_15d, predicted_harvest_relative_day_15d = determine_harvest_relative_day_from_threshold(
            severity_df, occurrence_id, threshold_15d)

        print(
            f"\t- For threshold_5d: {threshold_5d} "
            f"| Severity (accumulated): {severity_acc_5d} "
            f"| Predicted harvest_relative_day: {predicted_harvest_relative_day_5d}"
        )

        print(
            f"\t- For threshold_10d: {threshold_10d} "
            f"| Severity (accumulated): {severity_acc_10d} "
            f"| Predicted harvest_relative_day: {predicted_harvest_relative_day_10d}"
        )

        print(
            f"\t- For threshold_15d: {threshold_15d} "
            f"| Severity (accumulated): {severity_acc_15d} "
            f"| Predicted harvest_relative_day: {predicted_harvest_relative_day_15d}"
        )

        severity_acc_5d_list.append(severity_acc_5d)
        predicted_harvest_relative_day_5d_list.append(predicted_harvest_relative_day_5d)

        severity_acc_10d_list.append(severity_acc_10d)
        predicted_harvest_relative_day_10d_list.append(predicted_harvest_relative_day_10d)

        severity_acc_15d_list.append(severity_acc_15d)
        predicted_harvest_relative_day_15d_list.append(predicted_harvest_relative_day_15d)

    result_df = test_x.copy()
    result_df['harvest_relative_day'] = test_y['day_in_harvest']

    result_df["threshold_5d"] = threshold_5d
    result_df["severity_acc_5d"] = severity_acc_5d_list
    result_df["predicted_harvest_relative_day_5d"] = predicted_harvest_relative_day_5d_list

    result_df["threshold_10d"] = threshold_10d
    result_df["severity_acc_10d"] = severity_acc_10d_list
    result_df["predicted_harvest_relative_day_10d"] = predicted_harvest_relative_day_10d_list

    result_df["threshold_15d"] = threshold_15d
    result_df["severity_acc_15d"] = severity_acc_15d_list
    result_df["predicted_harvest_relative_day_15d"] = predicted_harvest_relative_day_15d_list

    if safra is not None:
        result_df["safra"] = [safra] * result_df.shape[0]

    result_df["fold"] = fold_number

    result_df.reset_index(inplace=True)

    return result_df


def determine_harvest_relative_day_from_threshold(severity_df: pd.DataFrame, occurrence_id, threshold) -> tuple:
    df = severity_df

    df = df[df["ocorrencia_id"] == occurrence_id]
    df = df[df["severity_acc"] <= threshold]
    df = df.loc[df["severity_acc"].idxmax()]

    return df["severity_acc"], df["harvest_relative_day"]


def write_result(result_df: pd.DataFrame, safra: str | None):
    filename = ""
    if safra is None:
        filename = "test_severity_model_results_all.csv"

    if safra is not None and safra.lower() == "all":
        filename = "test_severity_model_results_safra_all.csv"
    elif safra is not None:
        filename = f"test_severity_model_results_safra_{safra.replace("/", "_")}.csv"

    result_df.to_csv(OUTPUT_PATH / filename)


def prepare_folds(df: pd.DataFrame, k: int) -> list[tuple]:
    data_index = df.index.to_list()
    shuffle(data_index)

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
        "severity_acc_safra_5d", "severity_acc_safra_10d", "severity_acc_safra_15d",
    ]]
    train_y = train_df[["day_in_harvest"]].astype(int)

    test_x = test_df[[
        "ocorrencia_id", "segment_id_precipitation", "planting_start_date",
        "severity_acc_safra_5d", "severity_acc_safra_10d", "severity_acc_safra_15d",
    ]]
    test_y = test_df[["day_in_harvest"]].astype(int)

    return train_x, train_y, test_x, test_y
