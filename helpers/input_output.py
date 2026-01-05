import os
from datetime import datetime

from constants import OUTPUT_PATH


def output_file(execution_started: datetime, basedir: str, filename: str):
    """
    Generates the output file path and filename for a basedir, considering current execution time.
    """
    output_dir = OUTPUT_PATH / f"{basedir}_{execution_started.strftime('%Y-%m-%d_%H-%M-%S')}"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    return os.path.join(output_dir, filename)

def output_path(execution_started: datetime, basedir: str):
    """
    Generates the output file path for a basedir, considering current execution time.
    """
    output_dir = OUTPUT_PATH / f"{basedir}_{execution_started.strftime('%Y-%m-%d_%H-%M-%S')}"
    #adiciona uma barra no final do diretório
    output_dir = str(output_dir) + os.sep

    print(f"Output path: {output_dir}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    return output_dir


def get_latest_file(basedir: str, filename: str) -> str:
    """
    Get latest file given a basedir. Latest consider most recent directory based on datetime
    """
    basedir_list_executions = list(OUTPUT_PATH.glob(f"{basedir}_2*"))

    if len(basedir_list_executions) == 0:
        raise RuntimeError(f"Basedir directories not found for {basedir=}")

    basedir_list_executions.sort()
    latest_basedir = basedir_list_executions[-1]

    latest_file = latest_basedir / filename

    if not latest_file.exists():
        raise RuntimeError(f"Filename not found in basedir, {basedir=}, {filename=}")

    return latest_file


def get_latest_file_all(basedir: str) -> str:
    basedir_list_executions = list(OUTPUT_PATH.glob(f"{basedir}_2*"))

    if len(basedir_list_executions) == 0:
        raise RuntimeError(f"Directories not found for {basedir=}")

    basedir_list_executions.sort()
    latest_basedir = basedir_list_executions[-1]

    filename_list = latest_basedir.glob("*.csv")

    return filename_list
