import random

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sqlalchemy import Connection, text

from calculation.spatial import contains
from constants import (
    PARANA_MIN_LATITUDE,
    PARANA_MIN_LONGITUDE,
    PARANA_MAX_LATITUDE,
    PARANA_MAX_LONGITUDE,
    MAX_ATTEMPTS_RANDOM_COORDINATE,
    INPUT_PATH
)
from data_preparation.constants import QUERY_PRECIPITATION_SEGMENTS


def find_nearest_segment_id(conn: Connection, occurrence_id: int, lat, long) -> int:
    if lat is None or long is None or np.isnan(lat) or np.isnan(long):
        raise RuntimeError(f"Invalid values for lat/long {occurrence_id=} {lat=} {long=}")

    result = conn.execute(
        text(QUERY_PRECIPITATION_SEGMENTS.replace(":latitude", str(lat)).replace(":longitude", str(long)))
    ).fetchone()

    return int(result.t[0])


def determine_random_coordinate(collision_list: list[tuple[float, float]], min_distance: float) -> tuple[float, float]:
    for _ in range(MAX_ATTEMPTS_RANDOM_COORDINATE):
        random_latitude = random.uniform(PARANA_MIN_LATITUDE, PARANA_MAX_LATITUDE)
        random_longitude = random.uniform(PARANA_MIN_LONGITUDE, PARANA_MAX_LONGITUDE)

        candidate = Point(random_longitude, random_latitude)

        has_collision = __has_collision(collision_list, min_distance, candidate)
        is_in_parana = __is_in_parana(candidate)
        is_in_non_soybean_areas = __is_in_non_soybean_areas(candidate)

        if not has_collision and is_in_parana and not is_in_non_soybean_areas:
            return random_longitude, random_latitude

    raise Exception(f"Tried up to {MAX_ATTEMPTS_RANDOM_COORDINATE} times to get a coordinate. Exhausted.")


def __has_collision(collision_list: list[tuple[float, float]], collision_min_distance: float, candidate: Point) -> bool:
    for coordinate_pair in collision_list:
        coordinate = Point(coordinate_pair[0], coordinate_pair[1])

        if coordinate.distance(candidate) < collision_min_distance:
            return True

    return False


# Fazer desta forma é ineficiente mas por simplicidade da implementação atual por enquanto mantemos assim
def __is_in_parana(candidate: Point) -> bool:
    path = INPUT_PATH / "shapefiles_LimitesMunicipaisPR/LimitesMunicipaisPR.shp"

    gdf_parana_cities = gpd.read_file(path, crs="epsg:4674")
    gdf_parana_cities = gdf_parana_cities[["geometry"]]
    gdf_parana_cities.set_crs("epsg:4674", inplace=True, allow_override=True)

    r = contains(gdf_parana_cities, candidate)

    print(f"is_in_parana: {candidate} => {r}")

    return r


# Fazer desta forma é ineficiente mas por simplicidade da implementação atual por enquanto mantemos assim
def __is_in_non_soybean_areas(candidate: Point) -> bool:
    path = INPUT_PATH / "shapefiles_soybean_areas/shapefile_non_soybean_areas_parana_2021.gpkg"

    gdf_non_soybean_areas_parana = gpd.read_file(path, crs="epsg:4674")
    gdf_non_soybean_areas_parana = gdf_non_soybean_areas_parana[["geometry"]]
    gdf_non_soybean_areas_parana.set_crs("epsg:4674", inplace=True, allow_override=True)

    r = contains(gdf_non_soybean_areas_parana, candidate)

    print(f"is_in_non_soybean_areas: {candidate} => {r}")

    return r
