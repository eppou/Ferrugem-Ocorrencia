from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

from calculation.precipitation import collect_precipitation_safra
from calculation.severity import calculate_dsv_acc_with_df
from constants import DB_STRING, OUTPUT_PATH
from source.occurrence import get_safras

SEED = 5
TEST_SIZE = 0.20

"""
Prepara dataset de saída, com as instâncias e features calculadas da severidade acumulada para o threshold determinado
de 5, 10 e 15 dias. O threshold é calculado pela severidade média em 5, 10 e 15 dias antes da data_ocorrencia.
"""


# TODO: Criar CSV com severidade x dias, ao invés de calcular toda hora
def run():
    db_con_engine = create_engine(DB_STRING)
    conn = db_con_engine.connect()

    severity_df = pd.read_csv(OUTPUT_PATH / "severity_per_occurrence.csv")
    all_instances_df = pd.read_csv(OUTPUT_PATH / "instances_features_dataset.csv")

    safras = [s['safra'] for s in get_safras(conn)]
    safras = safras[1:]  # Removendo a última safra, pois está muito incompleta

    for safra in safras:
        print(f"=====> Safra: {safra}")

        instances_df = all_instances_df.copy()
        instances_df = instances_df[instances_df["safra"] == safra]
        instances_df = instances_df[instances_df["ocorrencia_id"].notnull()]
        print(f"=====> Instâncias totais: {instances_df.shape[0]}")

        x = instances_df[[
            "ocorrencia_id", "segment_id_precipitation", "planting_start_date",
            "severity_acc_safra_5d", "severity_acc_safra_10d", "severity_acc_safra_15d",
        ]]
        y = instances_df[["day_in_harvest"]].astype(int)

        # segmenting the dataset for training
        np.random.seed(SEED)
        train_x, test_x, train_y, test_y = train_test_split(
            x, y,
            test_size=TEST_SIZE,
            # stratify=y,
        )

        # finding the thresholds for 5, 10 and 15 days (average)
        threshold_5d = train_x["severity_acc_safra_5d"].mean()
        threshold_10d = train_x["severity_acc_safra_10d"].mean()
        threshold_15d = train_x["severity_acc_safra_15d"].mean()

        print(f"=====> threshold_5d: {threshold_5d} | threshold_10d: {threshold_10d} | threshold_15d: {threshold_15d}")

        # determining the day_in_harvest values (calculated)
        test_x.drop(columns=["severity_acc_safra_5d", "severity_acc_safra_10d", "severity_acc_safra_15d"], inplace=True)

        total_values_for_testing = test_x.shape[0]
        value_test_count = 0
        print(f"=====> Testing with {total_values_for_testing} values")

        severity_acc_5d_list, severity_acc_10d_list, severity_acc_15d_list = [], [], []
        predicted_harvest_relative_day_5d_list, predicted_harvest_relative_day_10d_list, predicted_harvest_relative_day_15d_list = [], [], []

        for index, instance in test_x.iterrows():
            value_test_count += 1

            occurrence_id = instance["ocorrencia_id"]
            print(
                f"=====> [{value_test_count}/{total_values_for_testing}] {safra} FOR OCCURRENCE IN: occurrence_id: {occurrence_id}")

            severity_acc_5d, predicted_harvest_relative_day_5d = determine_harvest_relative_day_from_threshold(severity_df, occurrence_id, threshold_5d)
            severity_acc_10d, predicted_harvest_relative_day_10d = determine_harvest_relative_day_from_threshold(severity_df, occurrence_id, threshold_10d)
            severity_acc_15d, predicted_harvest_relative_day_15d = determine_harvest_relative_day_from_threshold(severity_df, occurrence_id, threshold_15d)

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


        # for segment_id_precipitation, planting_start_date_str in test_x.values.tolist():
        #     value_test_count += 1
        #     print(
        #         f"=====> [{value_test_count}/{total_values_for_testing}] FOR OCCURRENCE IN: "
        #         f"segment_id_precipitation: {segment_id_precipitation} "
        #         f"| planting_start_date: {planting_start_date_str}"
        #     )
        #     planting_start_date = datetime.strptime(planting_start_date_str, "%Y-%m-%d").date()
        #     planting_presumed_end_date = planting_start_date + timedelta(days=200)
        #
        #     precipitation_per_segment_id_df = collect_precipitation_safra(conn, planting_start_date,
        #                                                                   planting_presumed_end_date)
        #
        #     severity = 0
        #     target_date = planting_start_date + timedelta(days=1)
        #     day_in_harvest = 1
        #     while severity < threshold_5d:
        #         severity = calculate_dsv_acc_with_df(
        #             precipitation_per_segment_id_df,
        #             segment_id_precipitation,
        #             planting_start_date,
        #             target_date
        #         )
        #         target_date += timedelta(days=1)
        #         day_in_harvest += 1
        #
        #     day_in_harvest += 5
        #     print(
        #         f"\t- Calculated: threshold_5d: {threshold_5d} | severity: {severity} | Calculated day_in_harvest: {day_in_harvest}")
        #     severity_acc_5d_list.append(severity)
        #     predicted_harvest_relative_day_5d_list.append(day_in_harvest)
        #
        #     severity = 0
        #     target_date = planting_start_date + timedelta(days=1)
        #     day_in_harvest = 1
        #     while severity < threshold_10d:
        #         severity = calculate_dsv_acc_with_df(
        #             precipitation_per_segment_id_df,
        #             segment_id_precipitation,
        #             planting_start_date,
        #             target_date
        #         )
        #         target_date += timedelta(days=1)
        #         day_in_harvest += 1
        #
        #     day_in_harvest += 10
        #     print(
        #         f"\t- Calculated: threshold_10d: {threshold_10d} | severity: {severity} | Calculated day_in_harvest: {day_in_harvest}")
        #     severity_acc_10d_list.append(severity)
        #     predicted_harvest_relative_day_10d_list.append(day_in_harvest)
        #
        #     severity = 0
        #     target_date = planting_start_date + timedelta(days=1)
        #     day_in_harvest = 1
        #     while severity < threshold_15d:
        #         severity = calculate_dsv_acc_with_df(
        #             precipitation_per_segment_id_df,
        #             segment_id_precipitation,
        #             planting_start_date,
        #             target_date
        #         )
        #         target_date += timedelta(days=1)
        #         day_in_harvest += 1
        #
        #     day_in_harvest += 15
        #     print(
        #         f"\t- Calculated: threshold_15d: {threshold_15d} | severity: {severity} | Calculated day_in_harvest: {day_in_harvest}")
        #     severity_acc_15d_list.append(severity)
        #     predicted_harvest_relative_day_15d_list.append(day_in_harvest)

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

        result_df["safra"] = [safra] * result_df.shape[0]

        result_df.reset_index(inplace=True)
        result_df.to_csv(OUTPUT_PATH / f"test_severity_model_results_{safra.replace("/", "_")}.csv")



def determine_harvest_relative_day_from_threshold(severity_df: pd.DataFrame, occurrence_id, threshold) -> tuple:
    df = severity_df

    df = df[df["ocorrencia_id"] == occurrence_id]
    df = df[df["severity_acc"] <= threshold]
    df = df.loc[df["severity_acc"].idxmax()]

    return df["severity_acc"], df["harvest_relative_day"]
