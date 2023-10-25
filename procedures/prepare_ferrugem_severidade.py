import pandas as pd
from datetime import date
from sqlalchemy import create_engine, Connection
from calculation.precipitation import calculate_precipitation_acc, calculate_precipitation_count
from calculation.coordinates import find_nearest_segment_id, determine_random_coordinate

from constants import DB_STRING, OUTPUT_PATH
from source.occurrence import get_safras

"""
Main pipeline to create the dataset with Soybean rust occurrences.
"""


def run():
    db_con_engine = create_engine(DB_STRING)
    conn = db_con_engine.connect()

    ocorrencias_df = pd.read_csv(OUTPUT_PATH / "instances_df.csv")

    print("=====> Assigning a segment_id (for precipitation data)")
    # Assigning a segment_id - a match for a position for the nearest precipitation data
    for index, ocorrencia in ocorrencias_df.iterrows():
        latitude = ocorrencia["ocorrencia_latitude"]
        longitude = ocorrencia["ocorrencia_longitude"]
        print(f"Finding nearest segment for (lat/long) {latitude} {longitude}, index {index}")

        segment_id = find_nearest_segment_id(conn, latitude, longitude)
        print(f"Segment found: {segment_id}, index {index}")

        ocorrencias_df.at[index, "segment_id"] = segment_id

    safras = get_safras(conn)

    for safra in safras:
        safra_nome = safra["safra"]
        ocorrencias_df_safra = ocorrencias_df[ocorrencias_df["safra"] == safra_nome]

        # FEATURE CALCULATION

        # Calculating and storing accumulated precipitation
        # Calculating number of days of precipitation
        # Calculating DSV severity indicator
        segment_id_list = ocorrencias_df_safra[["segment_id"]]
        p15d_list, p30d_list, p45d_list, p60d_list, p75d_list, p90d_list = [], [], [], [], [], []
        pc15d_list, pc30d_list, pc45d_list, pc60d_list, pc75d_list, pc90d_list = [], [], [], [], [], []

        for seg_data in segment_id_list.values.tolist():
            segment_id = int(seg_data[0])

            p15, p30, p45, p60, p75, p90 = calculate_precipitation_acc(
                conn,
                segment_id,
                safra["planting_start_date"],
                safra["planting_end_date"],
            )
            pc15, pc30, pc45, pc60, pc75, pc90 = calculate_precipitation_count(
                conn,
                segment_id,
                safra["planting_start_date"],
                safra["planting_end_date"],
            )
            # dsv_30d = calculate_dsv_30d(conn, segment_id, data)

            p15d_list.append(p15)
            p30d_list.append(p30)
            p45d_list.append(p45)
            p60d_list.append(p60)
            p75d_list.append(p75)
            p90d_list.append(p90)

            pc15d_list.append(pc15)
            pc30d_list.append(pc30)
            pc45d_list.append(pc45)
            pc60d_list.append(pc60)
            pc75d_list.append(pc75)
            pc90d_list.append(pc90)

        ocorrencias_df_safra["precipitation_15d"] = p15d_list
        ocorrencias_df_safra["precipitation_30d"] = p30d_list
        ocorrencias_df_safra["precipitation_45d"] = p45d_list
        ocorrencias_df_safra["precipitation_60d"] = p60d_list
        ocorrencias_df_safra["precipitation_75d"] = p75d_list
        ocorrencias_df_safra["precipitation_90d"] = p90d_list

        ocorrencias_df_safra["precipitation_15d_count"] = pc15d_list
        ocorrencias_df_safra["precipitation_30d_count"] = pc30d_list
        ocorrencias_df_safra["precipitation_45d_count"] = pc45d_list
        ocorrencias_df_safra["precipitation_60d_count"] = pc60d_list
        ocorrencias_df_safra["precipitation_75d_count"] = pc75d_list
        ocorrencias_df_safra["precipitation_90d_count"] = pc90d_list

        ocorrencias_df = pd.concat([ocorrencias_df, ocorrencias_df_safra])

    # Output full dataset (possible contain extra information for debugging and visualization)
    ocorrencias_df.to_csv(OUTPUT_PATH / "instances_features_dataset_all.csv", index=False)

    ocorrencias_df = ocorrencias_df
    [[
        "safra", "ocorrencia_latitude", "ocorrencia_longitude",
        "precipitation_15d", "precipitation_30d", "precipitation_45d",
        "precipitation_60d", "precipitation_75d", "precipitation_90d",
        "precipitation_15d_count", "precipitation_30d_count", "precipitation_45d_count",
        "precipitation_60d_count", "precipitation_75d_count", "precipitation_90d_count",
        "ocorrencia"
    ]].copy()
    ocorrencias_df.to_csv(OUTPUT_PATH / "instances_features_dataset.csv", index=False)

    conn.close()
    db_con_engine.dispose()
