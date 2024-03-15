MIN_DISTANCE_FOR_NON_OCCURRENCES = 0.2

QUERY_OCORRENCIAS = """
SELECT *
FROM ferrugem_asiatica_occurrences o
WHERE o.safra = ':safra';
"""

# TODO: Usar ST_DWithin ao inves deste https://postgis.net/docs/ST_DWithin.html
# TODO: Usar índice na coluna geometry
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
(SELECT SUM(p.prec) AS precipitation_15d FROM precipitation p
WHERE p.segment_id = :segment_id::int
    AND p.date_precipitation::date >= ':start_date'
    AND p.date_precipitation::date <= (':start_date'::date + '15 days'::interval)
    AND p.date_precipitation::date <= ':end_date'::date),
(SELECT SUM(p.prec) AS precipitation_30d FROM precipitation p
 WHERE p.segment_id = :segment_id::int
   AND p.date_precipitation::date >= ':start_date'
   AND p.date_precipitation::date <= (':start_date'::date + '30 days'::interval)
   AND p.date_precipitation::date <= ':end_date'::date),
(SELECT SUM(p.prec) AS precipitation_45d FROM precipitation p
 WHERE p.segment_id = :segment_id::int
   AND p.date_precipitation::date >= ':start_date'
   AND p.date_precipitation::date <= (':start_date'::date + '45 days'::interval)
   AND p.date_precipitation::date <= ':end_date'::date),
(SELECT SUM(p.prec) AS precipitation_60d FROM precipitation p
 WHERE p.segment_id = :segment_id::int
   AND p.date_precipitation::date >= ':start_date'
   AND p.date_precipitation::date <= (':start_date'::date + '60 days'::interval)
   AND p.date_precipitation::date <= ':end_date'::date),
(SELECT SUM(p.prec) AS precipitation_75d FROM precipitation p
 WHERE p.segment_id = :segment_id::int
   AND p.date_precipitation::date >= ':start_date'
   AND p.date_precipitation::date <= (':start_date'::date + '75 days'::interval)
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
    (SELECT COUNT(p.prec) AS precipitation_15d FROM precipitation p
     WHERE p.segment_id = :segment_id::int
       AND p.date_precipitation::date >= ':start_date'
       AND p.date_precipitation::date <= (':start_date'::date + '15 days'::interval)
       AND p.date_precipitation::date <= ':end_date'::date
       AND p.prec >= 0.5),
    (SELECT COUNT(p.prec) AS precipitation_30d FROM precipitation p
     WHERE p.segment_id = :segment_id::int
       AND p.date_precipitation::date >= ':start_date'
       AND p.date_precipitation::date <= (':start_date'::date + '30 days'::interval)
       AND p.date_precipitation::date <= ':end_date'::date
       AND p.prec >= 0.5),
    (SELECT COUNT(p.prec) AS precipitation_45d FROM precipitation p
     WHERE p.segment_id = :segment_id::int
       AND p.date_precipitation::date >= ':start_date'
       AND p.date_precipitation::date <= (':start_date'::date + '45 days'::interval)
       AND p.date_precipitation::date <= ':end_date'::date
       AND p.prec >= 0.5),
    (SELECT COUNT(p.prec) AS precipitation_60d FROM precipitation p
     WHERE p.segment_id = :segment_id::int
       AND p.date_precipitation::date >= ':start_date'
       AND p.date_precipitation::date <= (':start_date'::date + '60 days'::interval)
       AND p.date_precipitation::date <= ':end_date'::date
       AND p.prec >= 0.5),
    (SELECT COUNT(p.prec) AS precipitation_75d FROM precipitation p
     WHERE p.segment_id = :segment_id::int
       AND p.date_precipitation::date >= ':start_date'
       AND p.date_precipitation::date <= (':start_date'::date + '75 days'::interval)
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
SELECT safra, planting_start_date, planting_end_date FROM safra_date WHERE state = 'PR' AND safra <= '2022/2023'
"""

QUERY_PRECIPITATION_ACC_30D = """
SELECT
    (SELECT SUM(p.prec) AS precipitation_15d_acc_b
     FROM precipitation p
     WHERE p.segment_id = :segment_id::int
       AND p.date_precipitation::date >= (':target_date'::date - '30 days'::interval)
       AND p.date_precipitation::date <= (':target_date'::date - '16 day'::interval)),
    (SELECT SUM(p.prec) AS precipitation_15d_acc_a
     FROM precipitation p
     WHERE p.segment_id = :segment_id::int
       AND p.date_precipitation::date >= (':target_date'::date - '15 days'::interval)
       AND p.date_precipitation::date <= (':target_date'::date - '1 day'::interval))
;
"""


QUERY_PRECIPITATION_COUNT_30D = """
SELECT
    (SELECT COUNT(p.prec) AS precipitation_15d_count_b
     FROM precipitation p
     WHERE p.segment_id = :segment_id::int
       AND p.date_precipitation::date >= (':target_date'::date - '30 days'::interval)
       AND p.date_precipitation::date <= (':target_date'::date - '16 day'::interval)
       AND p.prec > 0.5),
    (SELECT COUNT(p.prec) AS precipitation_15d_count_a
     FROM precipitation p
     WHERE p.segment_id = :segment_id::int
       AND p.date_precipitation::date >= (':target_date'::date - '15 days'::interval)
       AND p.date_precipitation::date <= (':target_date'::date - '1 day'::interval)
       AND p.prec > 0.5)
;
"""


QUERY_PRECIPITATION_ACC_SAFRA = """
SELECT SUM(p.prec) AS precipitation_safra_acc
FROM precipitation p
 WHERE p.segment_id = :segment_id::int
   AND p.date_precipitation::date >= ':start_date'::date
   AND p.date_precipitation::date <= (':target_date'::date - '1 day'::interval)
;
"""


QUERY_PRECIPITATION_COUNT_SAFRA = """
SELECT COUNT(p.prec) AS precipitation_safra_count
FROM precipitation p
 WHERE p.segment_id = :segment_id::int
   AND p.date_precipitation::date >= ':start_date'::date
   AND p.date_precipitation::date <= (':target_date'::date - '1 day'::interval)
   AND p.prec > 0.5
;
"""

QUERY_PRECIPITATION_SAFRA_RAW = """
SELECT p.segment_id, p.date_precipitation::date, p.prec
FROM precipitation p
 WHERE 1=1
   AND p.date_precipitation::date >= ':start_date'::date
   AND p.date_precipitation::date <= (':target_date'::date - '1 day'::interval)
;
"""

QUERY_PRECIPITATION_FOR_ALL_HARVESTS = """
SELECT p.segment_id, p.date_precipitation::date, p.prec
FROM precipitation p
WHERE 1=1
  AND date_precipitation >= (SELECT MIN(planting_start_date) FROM safra_date WHERE state = 'PR')
  AND date_precipitation <= (SELECT MAX(planting_end_date) FROM safra_date WHERE state = 'PR')
"""