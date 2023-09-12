from sqlalchemy import Connection, text
from datetime import date
from procedures.constants import QUERY_SAFRAS

def get_safras(conn: Connection) -> list[dict[str, str | date]]:
    result = conn.execute(text(QUERY_SAFRAS)).fetchall()

    safras = []
    for row in result:
        safras.append({
            'safra': row[0],
            'planting_start_date': row[1],
            'planting_end_date': row[2],
        })

    return safras
