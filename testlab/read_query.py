import pandas as pd
from sqlalchemy import create_engine
from constants import DB_STRING


def run():
    db_con_engine = create_engine(DB_STRING)

    ocorrencias = pd.read_sql_query(
        """
        SELECT
            ocorrencia_id,
            data,
            cidade_nome,
            estado_nome,
            st_astext(st_setsrid(st_makepoint(ocorrencia_longitude, ocorrencia_latitude), 4674)) AS ocorrencia_localizacao,
            st_x(st_setsrid(st_makepoint(ocorrencia_longitude, ocorrencia_latitude), 4674)) AS longitude,
            st_y(st_setsrid(st_makepoint(ocorrencia_longitude, ocorrencia_latitude), 4674)) AS latitude
        
        FROM (
                 SELECT
                     o.id AS ocorrencia_id,
                     o.dia AS data,
                     c.nome                                                                     AS cidade_nome,
                     uf.nome                                                                    AS estado_nome,
                     CASE WHEN o.latitude IS NULL THEN st_y(shp.centroid) ELSE o.latitude END   AS ocorrencia_latitude,
                     CASE WHEN o.longitude IS NULL THEN st_x(shp.centroid) ELSE o.longitude END AS ocorrencia_longitude
        
                 FROM consorcio_antiferrugem.ocorrencia o
                          JOIN consorcio_antiferrugem.cidade c ON c.id = cidade_id
                          JOIN consorcio_antiferrugem.estado uf ON uf.id = c.estado_id
                          JOIN consorcio_antiferrugem.estadio e ON e.id = o.estadio_id
                          JOIN consorcio_antiferrugem.safra s ON s.id = o.safra_id
                          LEFT JOIN public.city_shapefiles shp ON lower(unaccent(c.nome)) = lower(shp.city_name)
                 WHERE 1 = 1
                   AND o.doenca_id = 1
                   AND uf.nome = 'PARANÁ'
                   AND EXTRACT(YEAR FROM o.dia) BETWEEN 2012 AND 2022
                   AND o.tipodearea_id = 1
                 ORDER BY dia DESC
             ) as occurrence_location_data;
        ;
""",
        con=db_con_engine,
    )

    print(ocorrencias.shape)
    print(ocorrencias.head(n=30))
