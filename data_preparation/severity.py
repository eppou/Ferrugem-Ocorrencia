from datetime import timedelta, datetime

import pandas as pd
from sqlalchemy import create_engine

from calculation.precipitation import collect_precipitation_planting
from calculation.severity import calculate_dsv_acc_with_df
from constants import DB_STRING, MAX_PLANTING_RELATIVE_DAY
from helpers.input_output import get_latest_file, output_file
from source.occurrence import get_safras

"""
Calculate accumulated severity for every occurrence, for all days until MAX_PLANTING_RELATIVE_DAY
"""


def run(execution_started_at: datetime, silent=True):
    db_con_engine = create_engine(DB_STRING)
    conn = db_con_engine.connect()

    df = pd.read_csv(
        get_latest_file("instances", "instances_all.csv"),
        parse_dates=["data_ocorrencia", "planting_start_date"],
    )

    instances_df = df[df["ocorrencia_id"].notnull()]
    instances_df = instances_df[
        ["ocorrencia_id", "data_ocorrencia", "segment_id_precipitation", "safra", "planting_start_date"]]

    precipitation_per_harvest_df = {}
    harvests = [s["safra"] for s in get_safras(conn)]

    for harvest in harvests:
        instances_harvest_df = instances_df[instances_df["safra"] == harvest].copy()
        earlier_planting_start_date = instances_harvest_df["planting_start_date"].min()
        latest_planting_start_date = instances_harvest_df["planting_start_date"].max()
        end_date = latest_planting_start_date + timedelta(days=MAX_PLANTING_RELATIVE_DAY)

        if not silent:
            print(
                f"\t- Calculating precipitation for {harvest=}"
                f", start_date (earlier_planting_start_date)={earlier_planting_start_date}"
                f", end_date (latest_planting_start_date + MAX_PLANTING_RELATIVE_DAY)={end_date}"
            )

        precipitation_per_harvest_df[harvest] = collect_precipitation_planting(
            conn,
            earlier_planting_start_date,
            end_date,
        )

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
        planting_start_date = instance["planting_start_date"].date()
        harvest = instance["safra"]

        if not silent:
            print(f"=====> {segment_id_precipitation=}")
            print(
                f"\t- Calculating severity (accumulated) for each day of the planting until {MAX_PLANTING_RELATIVE_DAY} days"
            )
        severity_list_instance = calculate_severity_all_planting_days(
            occurrence_id,
            segment_id_precipitation,
            planting_start_date,
            precipitation_per_harvest_df[harvest],
            silent,
        )

        severity_list = [*severity_list, *severity_list_instance]

    print()
    print(f"=====> Severity (accumulated) calculation finalized for all instances. Writing results.")
    severity_df = pd.DataFrame(severity_list)
    severity_df.to_csv(output_file(execution_started_at, "severity", "severity.csv"))


def calculate_severity_all_planting_days(
        occurrence_id,
        segment_id_precipitation,
        planting_start_date,
        precipitation_df,
        silent=True
) -> list:
    severities = []
    current_date = planting_start_date
    end_date = planting_start_date + timedelta(days=MAX_PLANTING_RELATIVE_DAY)
    current_planting_relative_day = 0

    precipitation_filtered_df = precipitation_df[
        (precipitation_df["segment_id"] == segment_id_precipitation) & (
                precipitation_df["date_precipitation"] >= planting_start_date)
        ]

    while current_date <= end_date:
        severity_acc = calculate_dsv_acc_with_df(
            precipitation_filtered_df,
            current_date,
        )

        severities.append({
            "occurrence_id": occurrence_id,
            "planting_relative_day": current_planting_relative_day,
            "date": current_date,
            "severity_acc": severity_acc,
        })

        if not silent:
            print(
                f"\t- Calculated severity (accumulated): {severity_acc} "
                f"| planting_start_date: {planting_start_date}"
                f"| planting_relative_day: {current_planting_relative_day}"
                f"| date: {current_date}"
            )

        current_date = current_date + timedelta(days=1)
        current_planting_relative_day += 1

    return severities
