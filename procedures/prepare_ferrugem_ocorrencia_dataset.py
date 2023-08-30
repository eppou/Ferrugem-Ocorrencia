import pandas as pd
from datetime import date
from sqlalchemy import create_engine, Connection, text

from constants import DB_STRING, OUTPUT_PATH
from procedures.constants import QUERY_OCORRENCIAS, QUERY_PRECIPITATION_SEGMENTS, QUERY_PRECIPITATION_ACC


def run():
    db_con_engine = create_engine(DB_STRING)
    conn = db_con_engine.connect()

    # Fetching all occurrences from consorcio_antiferrugem database
    ocorrencias_df_full = pd.read_sql_query(QUERY_OCORRENCIAS, con=db_con_engine)

    # Assigning a segment_id - a match for a position for the nearest precipitation data
    for index, ocorrencia in ocorrencias_df_full.iterrows():
        latitude = ocorrencia["ocorrencia_latitude"]
        longitude = ocorrencia["ocorrencia_longitude"]

        ocorrencias_df_full.at[index, "segment_id"] = find_nearest_segment_id(conn, latitude, longitude)

    # Calculating and storing accumulated precipitation
    segment_id_data_list = ocorrencias_df_full[["segment_id", "data"]]
    precipitation_7d_list, precipitation_14d_list, precipitation_30d_list = [], [], []
    for seg_data in segment_id_data_list.values.tolist():
        segment_id, data = seg_data
        p7d, p14d, p30d = calculate_precipitation_acc(conn, segment_id, data)
        precipitation_7d_list.append(p7d)
        precipitation_14d_list.append(p14d)
        precipitation_30d_list.append(p30d)

    ocorrencias_df_full["precipitation_7d"] = precipitation_7d_list
    ocorrencias_df_full["precipitation_14d"] = precipitation_14d_list
    ocorrencias_df_full["precipitation_30d"] = precipitation_30d_list

    # Output full dataset (possible contain extra information for debugging and visualization)
    ocorrencias_df_full.to_csv(OUTPUT_PATH / "ferrugem_ocorrencia_dataset_full.csv", index=False)

    ocorrencias_df = ocorrencias_df_full[["data", "ocorrencia_latitude", "ocorrencia_longitude", "ocorrencia"]].copy()
    ocorrencias_df.to_csv(OUTPUT_PATH / "ferrugem_ocorrencia_dataset.csv", index=False)

    conn.close()
    db_con_engine.dispose()


def find_nearest_segment_id(conn: Connection, lat, long) -> int:
    result = conn.execute(
        text(QUERY_PRECIPITATION_SEGMENTS.replace(":latitude", str(lat)).replace(":longitude", str(long)))
    ).fetchone()

    return int(result.t[0])


def calculate_precipitation_acc(conn: Connection, segment_id, target_date: date) -> tuple:
    target_date_str = target_date.strftime("%Y-%m-%d")
    result = conn.execute(
        text(QUERY_PRECIPITATION_ACC.replace(":segment_id", str(segment_id)).replace(":target_date", target_date_str))
    ).fetchone()

    return result.t