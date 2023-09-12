from sqlalchemy import Connection, text
from datetime import date
from procedures.constants import QUERY_PRECIPITATION_ACC, QUERY_PRECIPITATION_COUNT


def calculate_precipitation_acc(conn: Connection, segment_id, start_date: date, end_date: date) -> tuple:
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    result = conn.execute(
        text(
            QUERY_PRECIPITATION_ACC
            .replace(":segment_id", str(segment_id))
            .replace(":start_date", start_date_str)
            .replace(":end_date", end_date_str)
        )
    ).fetchone()

    return result.t


def calculate_precipitation_count(conn: Connection, segment_id, start_date: date, end_date: date) -> tuple:
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    result = conn.execute(
        text(
            QUERY_PRECIPITATION_COUNT
            .replace(":segment_id", str(segment_id))
            .replace(":start_date", start_date_str)
            .replace(":end_date", end_date_str)
        )
    ).fetchone()

    return result.t
