from datetime import date

from sqlalchemy import Connection, text

from procedures.constants import QUERY_PRECIPITATION_ACC, QUERY_PRECIPITATION_COUNT, QUERY_PRECIPITATION_ACC_30D, \
    QUERY_PRECIPITATION_COUNT_30D, QUERY_PRECIPITATION_ACC_SAFRA, QUERY_PRECIPITATION_COUNT_SAFRA


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


def calculate_precipitation_acc_30d(conn: Connection, segment_id, target_date: date) -> tuple:
    target_date_str = target_date.strftime("%Y-%m-%d")

    result = conn.execute(
        text(
            QUERY_PRECIPITATION_ACC_30D
            .replace(":segment_id", str(segment_id))
            .replace(":target_date", target_date_str)
        )
    ).fetchone()

    return result.t


def calculate_precipitation_count_30d(conn: Connection, segment_id, target_date: date) -> tuple:
    target_date_str = target_date.strftime("%Y-%m-%d")

    result = conn.execute(
        text(
            QUERY_PRECIPITATION_COUNT_30D
            .replace(":segment_id", str(segment_id))
            .replace(":target_date", target_date_str)
        )
    ).fetchone()

    return result.t


def calculate_precipitation_acc_safra(conn: Connection, segment_id, start_date: date, target_date: date) -> tuple:
    start_date_str = start_date.strftime("%Y-%m-%d")
    target_date_str = target_date.strftime("%Y-%m-%d")

    result = conn.execute(
        text(
            QUERY_PRECIPITATION_ACC_SAFRA
            .replace(":segment_id", str(segment_id))
            .replace(":target_date", target_date_str)
            .replace(":start_date", start_date_str)
        )
    ).fetchone()

    return result.t


def calculate_precipitation_count_safra(conn: Connection, segment_id, start_date: date, target_date: date) -> tuple:
    start_date_str = start_date.strftime("%Y-%m-%d")
    target_date_str = target_date.strftime("%Y-%m-%d")

    result = conn.execute(
        text(
            QUERY_PRECIPITATION_COUNT_SAFRA
            .replace(":segment_id", str(segment_id))
            .replace(":target_date", target_date_str)
            .replace(":start_date", start_date_str)
        )
    ).fetchone()

    return result.t
