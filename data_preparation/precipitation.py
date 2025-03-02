from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text

from config import Config
from data_preparation.constants import QUERY_PRECIPITATION
from helpers.input_output import output_file

"""
This just fetch precipitation from database, for min and max days for harvest start/end dates and writes in a file.

Used for analysis only. Model feature extraction takes directly from DB.
"""


def run(execution_started_at: datetime, cfg: Config):
   # Configurar a conexão com o banco de dados
    db_con_engine = create_engine(cfg.database_config.dbstring)
    conn = db_con_engine.connect()

    # Tamanho do bloco
    chunk_size = 5000  # Ajuste conforme necessário

    # Caminho de saída do arquivo CSV
    output_path = output_file(execution_started_at, "precipitation", "precipitation.csv")

    # Inicializar o arquivo CSV com o cabeçalho
    first_chunk = True

    # Processar em chunks
    for chunk in pd.read_sql_query(sql=text(QUERY_PRECIPITATION), con=conn, chunksize=chunk_size):
        # Salvar cada chunk no arquivo CSV
        chunk.to_csv(output_path, mode='a', index=False, header=first_chunk)
        first_chunk = False  # Apenas o primeiro chunk adiciona o cabeçalho
