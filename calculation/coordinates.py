from constants import (
    PARANA_MIN_LATITUDE,
    PARANA_MIN_LONGITUDE,
    PARANA_MAX_LATITUDE,
    PARANA_MAX_LONGITUDE,
    MAX_ATTEMPTS_RANDOM_COORDINATE
)
from shapely.geometry import Point
import random


def determine_random_coordinates(collision_list: list[tuple[float, float]], distance: float) -> tuple[float, float]:
    for _ in range(MAX_ATTEMPTS_RANDOM_COORDINATE):
        random_latitude = random.uniform(PARANA_MIN_LATITUDE, PARANA_MAX_LATITUDE)
        random_longitude = random.uniform(PARANA_MAX_LATITUDE, PARANA_MAX_LONGITUDE)

        candidate = Point(random_latitude, random_longitude)

        if not __has_collision(collision_list, distance, candidate):
            return random_latitude, random_longitude

    raise Exception(f"Tried up to {MAX_ATTEMPTS_RANDOM_COORDINATE} times to get a coordinate. Exhausted.")


def __has_collision(collision_list: list[tuple[float, float]], collision_min_distance: float, candidate: Point) -> bool:
    for coordinate_pair in collision_list:
        coordinate = Point(coordinate_pair[0], coordinate_pair[1])

        if coordinate.distance(candidate) > collision_min_distance:
            return True

    return False
