from datetime import date

import pandas as pd
from sqlalchemy import Connection

from calculation.precipitation import calculate_precipitation_acc_30d, calculate_precipitation_count_30d, \
    calculate_precipitation_acc_safra, calculate_precipitation_count_safra


def calculate_dsv_30d(conn: Connection, segment_id, target_date: date):
    p15d_b, p15d_a = calculate_precipitation_acc_30d(conn, segment_id, target_date)
    pc15d_b, pc15d_a = calculate_precipitation_count_30d(conn, segment_id, target_date)

    dsv = pdsv_a(p15d_a, pc15d_a) + pdsv_b(p15d_b, pc15d_b)
    dsv = round(dsv, 4)

    return dsv


def calculate_dsv_safra(conn: Connection, segment_id, safra_start_date: date, target_date: date):
    p, = calculate_precipitation_acc_safra(conn, segment_id, safra_start_date, target_date)
    pc, = calculate_precipitation_count_safra(conn, segment_id, safra_start_date, target_date)

    dsv = pdsv_generic(p, pc)
    dsv = round(dsv, 4)

    return dsv


def calculate_dsv_safra_df(df: pd.DataFrame, segment_id, safra_start_date: date, target_date: date):
    safra_start_date_pd = safra_start_date
    target_date_pd = target_date
    df_filtered = df[(df["segment_id"] == segment_id) & (df["date_precipitation"] >= safra_start_date_pd) & (df["date_precipitation"] <= target_date_pd)]

    p = df_filtered["prec"].sum()
    pc = df_filtered.loc[(df["prec"] > 0.5)].shape[0]

    dsv = pdsv_generic(p, pc)
    dsv = round(dsv, 4)

    return dsv


def pdsv_a(rain_acc: float, rain_nod: int) -> float:
    return round(0.7 * pdsv_generic(rain_acc, rain_nod), 4)


def pdsv_b(rain_acc: float, rain_nod: int) -> float:
    return round(0.3 * pdsv_generic(rain_acc, rain_nod), 4)


def pdsv_generic(rain_acc: float, rain_nod: int) -> float:
    if rain_acc == 0 or rain_nod == 0:
        return 0.0

    return -2.1433 + 0.3622 * rain_acc + 2.573 * rain_nod
