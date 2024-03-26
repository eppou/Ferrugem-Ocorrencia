import os
from datetime import datetime, timedelta

import requests

from constants import OUTPUT_PATH

"""
Download script for grib files from INPE (CPTEC). Precipitation data.
Does not import the data, only downloads it.
"""


def run():
    download("2000-06-01", "2024-02-29")


def download(start_date_str: str, end_date_str: str, force_redownload=False):
    base_url = "http://ftp.cptec.inpe.br/"

    for paths in get_paths(start_date_str, end_date_str):
        if os.path.isfile(OUTPUT_PATH / paths[2]) and force_redownload is False:
            print(f"Skipping download for {paths[0]}. File is already present.")
            continue

        url = base_url + paths[0]
        r = requests.get(url, allow_redirects=True)

        if not os.path.exists(OUTPUT_PATH / paths[1]):
            os.makedirs(OUTPUT_PATH / paths[1])
        open(OUTPUT_PATH / paths[2], 'wb').write(r.content)

        print(f"Downloaded file {paths[0]} to {OUTPUT_PATH / paths[2]}")


def get_paths(target_start_date: str, target_end_date: str) -> list[tuple[str, str, str]]:
    source_destination_paths = []

    start_date = datetime.strptime(str(target_start_date), "%Y-%m-%d").date()
    end_date = datetime.strptime(str(target_end_date), "%Y-%m-%d").date()

    while start_date <= end_date:
        if int(start_date.month) < 10 and int(start_date.day) < 10:
            file_name = f"MERGE_CPTEC_{start_date.year}0{start_date.month}0{start_date.day}.grib2"
            path = f"modelos/tempo/MERGE/GPM/DAILY/{start_date.year}/0{start_date.month}/{file_name}"
            directory_destination = f"fileCPTEC/{start_date.year}/0{start_date.month}/"
            path_destination = f"fileCPTEC/{start_date.year}/0{start_date.month}/{file_name}"
        elif int(start_date.month) < 10:
            file_name = f"MERGE_CPTEC_{start_date.year}0{start_date.month}{start_date.day}.grib2"
            path = f"modelos/tempo/MERGE/GPM/DAILY/{start_date.year}/0{start_date.month}/{file_name}"
            directory_destination = f"fileCPTEC/{start_date.year}/0{start_date.month}/"
            path_destination = f"fileCPTEC/{start_date.year}/0{start_date.month}/{file_name}"
        elif int(start_date.day) < 10:
            file_name = f"MERGE_CPTEC_{start_date.year}{start_date.month}0{start_date.day}.grib2"
            path = f"modelos/tempo/MERGE/GPM/DAILY/{start_date.year}/{start_date.month}/{file_name}"
            directory_destination = f"fileCPTEC/{start_date.year}/{start_date.month}/"
            path_destination = f"fileCPTEC/{start_date.year}/{start_date.month}/{file_name}"
        else:
            file_name = f"MERGE_CPTEC_{start_date.year}{start_date.month}{start_date.day}.grib2"
            path = f"modelos/tempo/MERGE/GPM/DAILY/{start_date.year}/{start_date.month}/{file_name}"
            directory_destination = f"fileCPTEC/{start_date.year}/{start_date.month}/"
            path_destination = f"fileCPTEC/{start_date.year}/{start_date.month}/{file_name}"

        source_destination_paths.append((path, directory_destination, path_destination))

        start_date = start_date + timedelta(1)

    return source_destination_paths
