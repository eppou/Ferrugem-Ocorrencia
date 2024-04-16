from datetime import timedelta, datetime

import pandas as pd
from sqlalchemy import create_engine

from calculation.precipitation import collect_precipitation_safra
from calculation.severity import calculate_dsv_acc_with_df
from constants import DB_STRING, MAX_HARVEST_RELATIVE_DAY
from helpers.input_output import get_latest_file, output_file


def calculate_severity_all_harvest_days(
        occurrence_id,
        segment_id_precipitation,
        harvest_start_date,
        precipitation_per_occurrence_harvest_df,
        silent=True
) -> list:
    severities = []
    current_date = harvest_start_date
    end_date = harvest_start_date + timedelta(days=MAX_HARVEST_RELATIVE_DAY)  # Desnecessário, já calculado
    current_harvest_relative_day = 0

    df = precipitation_per_occurrence_harvest_df.copy()
    precipitation_filtered_df = df[
        (df["segment_id"] == segment_id_precipitation) & (df["date_precipitation"] >= harvest_start_date)]

    while current_date <= end_date:
        severity_acc = calculate_dsv_acc_with_df(
            precipitation_filtered_df,
            current_date,
        )

        severities.append({
            "occurrence_id": occurrence_id,
            "harvest_relative_day": current_harvest_relative_day,
            "date": current_date,
            "severity_acc": severity_acc,
        })

        if not silent:
            print(
                f"\t- Calculated severity (accumulated): {severity_acc} "
                f"| harvest_start_date: {harvest_start_date}"
                f"| harvest_relative_day: {current_harvest_relative_day}"
                f"| date: {current_date}"
            )

        current_date = current_date + timedelta(days=1)
        current_harvest_relative_day += 1

    return severities


def run(silent=True):
    db_con_engine = create_engine(DB_STRING)
    conn = db_con_engine.connect()
    execution_started = datetime.now()

    df = pd.read_csv(get_latest_file("prepare_occurrence_instances", "instances_dataset_all.csv"),
                     parse_dates=["data_ocorrencia", "harvest_start_date"])

    instances_df = df[df["ocorrencia_id"].notnull()]
    instances_df = instances_df[
        ["ocorrencia_id", "data_ocorrencia", "segment_id_precipitation", "safra", "harvest_start_date"]]

    precipitation_per_safra = {}

    severity_list = []
    instances_count = 0
    for index, instance in instances_df.iterrows():
        instances_count += 1

        end_line = ''
        if not silent:
            end_line = '\n'

        print(f"\r=====> Progress [{instances_count}/{instances_df.shape[0]}]", end=end_line)

        occurrence_id = instance["ocorrencia_id"]
        segment_id_precipitation = instance["segment_id_precipitation"]
        harvest_start_date = instance["harvest_start_date"].date()
        harvest = instance["safra"]

        if harvest not in precipitation_per_safra:
            end_date = harvest_start_date + timedelta(days=MAX_HARVEST_RELATIVE_DAY)

            if not silent:
                print(
                    f"\t- Calculating precipitation for instance's harvest dates {harvest=}"
                    f", start_date={harvest_start_date}, end_date={end_date}"
                )

            # Todas as entradas de precipitação para a safra inteira da ocorrência atual
            df = collect_precipitation_safra(
                conn,
                harvest_start_date,
                end_date,
            )
            precipitation_per_safra[harvest] = df

        if not silent:
            print(
                f"\t- Calculating severity (accumulated) for each day of the harvest until {MAX_HARVEST_RELATIVE_DAY} days"
            )
        severity_list_instance = calculate_severity_all_harvest_days(
            occurrence_id,
            segment_id_precipitation,
            harvest_start_date,
            precipitation_per_safra[harvest],
            silent,
        )

        severity_list = [*severity_list, *severity_list_instance]

    print(f"=====> Severity (accumulated) calculation finalized for all instances. Writing results.")
    severity_df = pd.DataFrame(severity_list)
    severity_df.to_csv(output_file(execution_started, "prepare_severity_per_occurrence", "severity_per_occurrence.csv"))
