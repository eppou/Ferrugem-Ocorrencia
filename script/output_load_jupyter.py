from shutil import copy

from constants import OUTPUT_PATH_JUPYTER
from helpers.input_output import get_latest_file_all


def run(execution_started_at):
    try:
        source_path_list = [
            *list(get_latest_file_all("instances")),
            *list(get_latest_file_all("severity")),
            *list(get_latest_file_all("features")),
            *list(get_latest_file_all("concorrente")),
            *list(get_latest_file_all("proposta")),
            *list(get_latest_file_all("precipitation")),
        ]
    except (RuntimeError, OSError) as e:
        print(f"Error when getting output files: {e}")
        return

    destination_path = OUTPUT_PATH_JUPYTER

    print(f"Destination path: {destination_path}")
    for source_path in source_path_list:
        try:
            print(f"- Copying {source_path}... ", end="")
            copy(source_path, destination_path)
            print("Done")
        except Exception as e:
            print(f"\nError when copying file {source_path}: {e}")
            break
