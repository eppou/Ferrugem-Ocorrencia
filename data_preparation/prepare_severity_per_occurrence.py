from datetime import timedelta, datetime, date

import pandas as pd
from sqlalchemy import create_engine

from calculation.precipitation import collect_precipitation_safra
from calculation.severity import calculate_dsv_acc_with_df
from constants import DB_STRING, MAX_HARVEST_RELATIVE_DAY
from helpers.input_output import get_latest_file, output_file
from source.occurrence import get_safras


def calculate_severity_all_harvest_days(
        occurrence_id,
        segment_id_precipitation,
        harvest_start_date,
        precipitation_per_segment_id_df
) -> list:
    severities = []
    current_date = harvest_start_date
    end_date = harvest_start_date + timedelta(days=MAX_HARVEST_RELATIVE_DAY)
    current_harvest_relative_day = 0

    while current_date <= end_date:
        severity_acc = calculate_dsv_acc_with_df(
            precipitation_per_segment_id_df,
            segment_id_precipitation,
            harvest_start_date,
            current_date,
        )

        severities.append({
            "occurrence_id": occurrence_id,
            "harvest_relative_day": current_harvest_relative_day,
            "date": current_date,
            "severity_acc": severity_acc,
        })
        print(
            f"\t- Calculated severity (accumulated): {severity_acc} "
            f"| harvest_relative_day: {current_harvest_relative_day}"
            f"| date: {current_date}"
        )

        current_date = current_date + timedelta(days=1)
        current_harvest_relative_day += 1

    return severities


def run():
    db_con_engine = create_engine(DB_STRING)
    conn = db_con_engine.connect()
    execution_started = datetime.now()

    df = pd.read_csv(get_latest_file("prepare_occurrence_instances", "instances_dataset_all.csv"), parse_dates=["data_ocorrencia"])

    instances_df = df[df["ocorrencia_id"].notnull()]
    instances_df = instances_df[
        ["ocorrencia_id", "data_ocorrencia", "segment_id_precipitation", "safra", "harvest_start_date"]]

    severity_list = []
    instances_count = 0
    for index, instance in instances_df.iterrows():
        instances_count += 1
        print(f"=====> Progress [{instances_count}/{instances_df.shape[0]}]")

        occurrence_id = instance["ocorrencia_id"]
        segment_id_precipitation = instance["segment_id_precipitation"]
        occurrence_date = instance["data_ocorrencia"].date()
        harvest_start_date = instance["harvest_start_date"]

        print("\t- Calculating precipitation for instance's harvest dates")
        precipitation_per_segment_id_df = collect_precipitation_safra(
            conn,
            harvest_start_date,
            occurrence_date,
        )

        print(
            f"\t- Calculating severity (accumulated) for each day of the harvest until {MAX_HARVEST_RELATIVE_DAY} days"
        )
        severity_list_instance = calculate_severity_all_harvest_days(
            occurrence_id,
            segment_id_precipitation,
            harvest_start_date,
            precipitation_per_segment_id_df,
        )

        severity_list = [*severity_list, *severity_list_instance]
        print("\t- Done")

    print(f"=====> Severity (accumulated) calculation finalized for all instances. Writing results.")
    severity_df = pd.DataFrame(severity_list)
    severity_df.to_csv(output_file(execution_started, "prepare_severity_per_occurrence", "severity_per_occurrence.csv"))
