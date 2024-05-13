from datetime import datetime

import pandas as pd

from helpers.input_output import output_file


def write_result(
        base_filename: str,
        description: str,
        execution_started: datetime,
        result_df: pd.DataFrame,
        safra: str | None
):
    filename_description = description.strip()
    if description == "":
        filename_description = "_"
    else:
        filename_description = f"_{filename_description}_"

    filename = ""

    if safra is None:
        filename = f"{base_filename}_results{filename_description}all.csv"

    if safra is not None and safra.lower() == "all":
        filename = f"{base_filename}_results{filename_description}harvest_all.csv"
    elif safra is not None:
        filename = f"{base_filename}_results{filename_description}harvest_{safra.replace("/", "_")}.csv"

    result_df.to_csv(output_file(execution_started, base_filename, filename))
