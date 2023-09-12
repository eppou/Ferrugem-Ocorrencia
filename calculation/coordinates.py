from constants import (
    PARANA_MIN_LATITUDE,
    PARANA_MIN_LONGITUDE,
    PARANA_MAX_LATITUDE,
    PARANA_MAX_LONGITUDE,
    MAX_ATTEMPTS_RANDOM_COORDINATE
)
from procedures.constants import QUERY_PRECIPITATION_SEGMENTS
from shapely.geometry import Point
import random
from sqlalchemy import Connection, text


def find_nearest_segment_id(conn: Connection, lat, long) -> int:
    result = conn.execute(
        text(QUERY_PRECIPITATION_SEGMENTS.replace(":latitude", str(lat)).replace(":longitude", str(long)))
    ).fetchone()

    return int(result.t[0])


def determine_random_coordinates(collision_list: list[tuple[float, float]], min_distance: float) -> tuple[float, float]:
    for _ in range(MAX_ATTEMPTS_RANDOM_COORDINATE):
        # TODO: Recorte para o Paraná
        random_latitude = random.uniform(PARANA_MIN_LATITUDE, PARANA_MAX_LATITUDE)
        random_longitude = random.uniform(PARANA_MAX_LATITUDE, PARANA_MAX_LONGITUDE)

        candidate = Point(random_latitude, random_longitude)

        if not __has_collision(collision_list, min_distance, candidate):
            return random_latitude, random_longitude

    raise Exception(f"Tried up to {MAX_ATTEMPTS_RANDOM_COORDINATE} times to get a coordinate. Exhausted.")


def __has_collision(collision_list: list[tuple[float, float]], collision_min_distance: float, candidate: Point) -> bool:
    for coordinate_pair in collision_list:
        coordinate = Point(coordinate_pair[0], coordinate_pair[1])

        if coordinate.distance(candidate) < collision_min_distance:
            return True

    return False
