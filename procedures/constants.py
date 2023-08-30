QUERY_OCORRENCIAS = """
SELECT
    ocorrencia_id,
    data,
    cidade_nome,
    estado_nome,
    st_astext(st_setsrid(st_makepoint(ocorrencia_latitude, ocorrencia_longitude), 4674)) AS ocorrencia_localizacao,
    st_x(st_setsrid(st_makepoint(ocorrencia_latitude, ocorrencia_longitude), 4674)) AS ocorrencia_latitude,
    st_y(st_setsrid(st_makepoint(ocorrencia_latitude, ocorrencia_longitude), 4674)) AS ocorrencia_longitude,
    true as ocorrencia

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
         WHERE 1=1
           AND o.doenca_id = 1
           AND uf.nome = 'PARANÁ'
           -- AND EXTRACT(YEAR FROM o.dia) BETWEEN 2020 AND 2021
           AND EXTRACT(YEAR FROM o.dia) = 2022
           AND o.tipodearea_id = 1
         ORDER BY o.dia DESC
     ) as occurrence_location_data
;
"""

QUERY_PRECIPITATION_SEGMENTS = """
SELECT DISTINCT 
    segment_id, 
    st_distance('SRID=4674;POINT(:latitude :longitude)'::geometry, p.geometry) AS distance
FROM precipitation p
ORDER BY distance
LIMIT 1;
"""

QUERY_PRECIPITATION_ACC = """
SELECT
(SELECT SUM(p.prec) AS precipitation_7d FROM precipitation p
WHERE p.segment_id = :segment_id::int
    AND p.date_precipitation::date < ':target_date'
    AND p.date_precipitation::date >= (':target_date'::date - '7 days'::interval)),
(SELECT SUM(p.prec) AS precipitation_14d FROM precipitation p
WHERE p.segment_id = :segment_id::int
    AND p.date_precipitation::date < ':target_date'
    AND p.date_precipitation::date >= (':target_date'::date - '14 days'::interval)),
(SELECT SUM(p.prec) AS precipitation_30d
FROM precipitation p
WHERE p.segment_id = :segment_id::int
    AND p.date_precipitation::date < ':target_date'
    AND p.date_precipitation::date >= (':target_date'::date - '30 days'::interval));
"""
