from datetime import datetime

import pandas as pd

from helpers.input_output import output_file


def write_result(
        base_filename: str,
        description: str,
        execution_started: datetime,
        result_df: pd.DataFrame,
        safra: str | None,
        description_extra: str = "",
):
    result_df.to_csv(
        compose_filename(base_filename, description, execution_started, safra, description_extra)
    )


def read_result(
        base_filename: str,
        description: str,
        execution_started: datetime,
        safra: str | None,
        description_extra: str = "",
) -> pd.DataFrame:
    return pd.read_csv(
        compose_filename(base_filename, description, execution_started, safra, description_extra)
    )


def compose_filename(
        base_filename: str,
        description: str,
        execution_started: datetime,
        safra: str | None,
        description_extra: str = "",
):
    filename_description = description.strip()
    if filename_description == "":
        filename_description = "_"
    else:
        filename_description = f"_{filename_description}_"

    filename_description_extra = description_extra.strip()
    if filename_description_extra == "":
        filename_description_extra = ""
    else:
        filename_description_extra = f"_{filename_description_extra}_"

    filename_description += filename_description_extra

    filename = ""

    if safra is None:
        filename = f"{base_filename}_results{filename_description}all.csv"

    if safra is not None and safra.lower() == "all":
        filename = f"{base_filename}_results{filename_description}harvest_all.csv"
    elif safra is not None:
        filename = f"{base_filename}_results{filename_description}harvest_{safra.replace("/", "_")}.csv"

    return output_file(execution_started, base_filename, filename)
