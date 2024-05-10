from datetime import datetime, timedelta

import pandas as pd

from constants import FEATURE_DAY_INTERVAL, MAX_PLANTING_RELATIVE_DAY
from helpers.input_output import get_latest_file, output_file


def run():
    execution_start = datetime.now()
    features_df = pd.read_csv(
        get_latest_file("features", "features_all.csv"),
        parse_dates=["data_ocorrencia", "planting_start_date"]
    )
    features_df = features_df.drop(columns=["level_0", "index", "Unnamed: 0"])

    for index, instance in features_df.iterrows():
        planting_start_date = instance["planting_start_date"]
        occurrence_date = instance["data_ocorrencia"]

        current_planting_relative_day = FEATURE_DAY_INTERVAL
        current_date = pd.to_datetime(planting_start_date + timedelta(days=current_planting_relative_day))

        while current_planting_relative_day <= MAX_PLANTING_RELATIVE_DAY:
            if current_date > occurrence_date:
                features_df.at[index, f"precipitation_acc_{current_planting_relative_day}d"] = 0
                features_df.at[index, f"precipitation_count_{current_planting_relative_day}d"] = 0

            current_planting_relative_day += FEATURE_DAY_INTERVAL
            current_date = pd.to_datetime(planting_start_date + timedelta(days=current_planting_relative_day))

    features_df.reset_index(inplace=True)
    features_df.to_csv(output_file(execution_start, "features_with_zero", "features_with_zero_all.csv"))

    features_df = features_df.filter(axis=1, regex="(safra|ocorrencia|precipitation)").copy()
    features_df.to_csv(output_file(execution_start, "features_with_zero", "features_with_zero.csv"))
