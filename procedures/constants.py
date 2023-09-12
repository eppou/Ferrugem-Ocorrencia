MIN_DISTANCE_FOR_NON_OCCURRENCES = 0.2

QUERY_OCORRENCIAS = """
SELECT *
FROM ferrugem_asiatica_occurrences o
WHERE o.safra = ':safra';
"""

QUERY_PRECIPITATION_SEGMENTS = """
SELECT DISTINCT 
    segment_id, 
    st_distance('SRID=4674;POINT(:longitude :latitude)'::geometry, p.geometry) AS distance
FROM precipitation p
ORDER BY distance
LIMIT 1;
"""

QUERY_PRECIPITATION_ACC = """
SELECT
(SELECT SUM(p.prec) AS precipitation_14d FROM precipitation p
WHERE p.segment_id = :segment_id::int
    AND p.date_precipitation::date >= ':start_date'
    AND p.date_precipitation::date <= (':start_date'::date + '14 days'::interval)
    AND p.date_precipitation::date <= ':end_date'::date),
(SELECT SUM(p.prec) AS precipitation_30d FROM precipitation p
 WHERE p.segment_id = :segment_id::int
   AND p.date_precipitation::date >= ':start_date'
   AND p.date_precipitation::date <= (':start_date'::date + '30 days'::interval)
   AND p.date_precipitation::date <= ':end_date'::date),
(SELECT SUM(p.prec) AS precipitation_60d FROM precipitation p
 WHERE p.segment_id = :segment_id::int
   AND p.date_precipitation::date >= ':start_date'
   AND p.date_precipitation::date <= (':start_date'::date + '60 days'::interval)
   AND p.date_precipitation::date <= ':end_date'::date),
(SELECT SUM(p.prec) AS precipitation_90d FROM precipitation p
 WHERE p.segment_id = :segment_id::int
   AND p.date_precipitation::date >= ':start_date'
   AND p.date_precipitation::date <= (':start_date'::date + '90 days'::interval)
   AND p.date_precipitation::date <= ':end_date'::date)
;
"""

QUERY_PRECIPITATION_COUNT = """
SELECT
    (SELECT COUNT(p.prec) AS precipitation_14d FROM precipitation p
     WHERE p.segment_id = :segment_id::int
       AND p.date_precipitation::date >= ':start_date'
       AND p.date_precipitation::date <= (':start_date'::date + '14 days'::interval)
       AND p.date_precipitation::date <= ':end_date'::date
       AND p.prec >= 0.5),
    (SELECT COUNT(p.prec) AS precipitation_30d FROM precipitation p
     WHERE p.segment_id = :segment_id::int
       AND p.date_precipitation::date >= ':start_date'
       AND p.date_precipitation::date <= (':start_date'::date + '30 days'::interval)
       AND p.date_precipitation::date <= ':end_date'::date
       AND p.prec >= 0.5),
    (SELECT COUNT(p.prec) AS precipitation_60d FROM precipitation p
     WHERE p.segment_id = :segment_id::int
       AND p.date_precipitation::date >= ':start_date'
       AND p.date_precipitation::date <= (':start_date'::date + '60 days'::interval)
       AND p.date_precipitation::date <= ':end_date'::date
       AND p.prec >= 0.5),
    (SELECT COUNT(p.prec) AS precipitation_90d FROM precipitation p
     WHERE p.segment_id = :segment_id::int
       AND p.date_precipitation::date >= ':start_date'
       AND p.date_precipitation::date <= (':start_date'::date + '90 days'::interval)
       AND p.date_precipitation::date <= ':end_date'::date
       AND p.prec >= 0.5)
;

"""

# TODO: Remove filter from query
QUERY_SAFRAS = """
SELECT safra, planting_start_date, planting_end_date FROM safra_date WHERE state = 'PR' AND safra = '2021/2022'
"""