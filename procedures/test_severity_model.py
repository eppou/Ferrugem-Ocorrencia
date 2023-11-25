from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, Connection

from calculation.severity import calculate_dsv_safra
from constants import DB_STRING, OUTPUT_PATH

SEED = 5
TEST_SIZE = 0.10


def run():
    data = pd.read_csv(OUTPUT_PATH / "instances_features_dataset.csv", sep=",")
    data_filtered = data.copy()
    data_filtered = data_filtered[data_filtered["data_ocorrencia"].notnull()]

    x = data_filtered[[
        "segment_id_precipitation", "planting_start_date",
        "severity_acc_safra_5d", "severity_acc_safra_10d", "severity_acc_safra_15d",
    ]]
    y = data_filtered[["day_in_harvest"]].astype(int)

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

    print(f"threshold_5d: {threshold_5d}")
    print(f"threshold_10d: {threshold_10d}")
    print(f"threshold_15d: {threshold_15d}")

    # determining the day_in_harvest values (calculated)
    test_x.drop(columns=["severity_acc_safra_5d", "severity_acc_safra_10d", "severity_acc_safra_15d"], inplace=True)

    db_con_engine = create_engine(DB_STRING)
    conn = db_con_engine.connect()

    severity_5d_list, severity_10d_list, severity_15d_list = [], [], []
    day_in_harvest_5d_list, day_in_harvest_10d_list, day_in_harvest_15d_list = [], [], []
    for segment_id_precipitation, planting_start_date_str in test_x.values.tolist():
        print(f"=====> segment_id_precipitation: {segment_id_precipitation} | planting_start_date: {planting_start_date_str}")
        planting_start_date = datetime.strptime(planting_start_date_str, "%Y-%m-%d")

        severity = 0
        target_date = planting_start_date + timedelta(days=1)
        day_in_harvest = 1
        while severity < threshold_5d:
            severity = calculate_dsv_safra(
                conn,
                segment_id_precipitation,
                planting_start_date,
                target_date
            )
            target_date += timedelta(days=1)
            day_in_harvest += 1

        print(f"Calculated: threshold_5d: {threshold_5d} | severity: {severity} | Calculated day_in_harvest: {day_in_harvest}")
        severity_5d_list.append(severity)
        day_in_harvest_5d_list.append(day_in_harvest)

        severity = 0
        target_date = planting_start_date + timedelta(days=1)
        day_in_harvest = 1
        while severity < threshold_10d:
            # TODO: Optimize to fetch all data to memory instead of individual queries
            severity = calculate_dsv_safra(
                conn,
                segment_id_precipitation,
                planting_start_date,
                target_date
            )
            target_date += timedelta(days=1)
            day_in_harvest += 1

        print(f"Calculated: threshold_10d: {threshold_10d} | severity: {severity} | Calculated day_in_harvest: {day_in_harvest}")
        severity_10d_list.append(severity)
        day_in_harvest_10d_list.append(day_in_harvest)

        severity = 0
        target_date = planting_start_date + timedelta(days=1)
        day_in_harvest = 1
        while severity < threshold_15d:
            severity = calculate_dsv_safra(
                conn,
                segment_id_precipitation,
                planting_start_date,
                target_date
            )
            target_date += timedelta(days=1)
            day_in_harvest += 1

        print(f"Calculated: threshold_15d: {threshold_15d} | severity: {severity} | Calculated day_in_harvest: {day_in_harvest}")
        severity_15d_list.append(severity)
        day_in_harvest_15d_list.append(day_in_harvest)

    result_df = pd.concat([test_x, test_y])
    result_df["threshold_5d"] = threshold_5d
    result_df["severity_5d"] = severity_5d_list
    result_df["day_in_harvest_5d"] = day_in_harvest_5d_list
    result_df["threshold_10d"] = threshold_10d
    result_df["severity_10d"] = severity_10d_list
    result_df["day_in_harvest_10d"] = day_in_harvest_10d_list
    result_df["threshold_15d"] = threshold_15d
    result_df["severity_15d"] = severity_15d_list
    result_df["day_in_harvest_15d"] = day_in_harvest_15d_list

    print(result_df.head(10))

    result_df.reset_index(inplace=True)
    result_df.to_csv(OUTPUT_PATH / "test_severity_model_results.csv")
