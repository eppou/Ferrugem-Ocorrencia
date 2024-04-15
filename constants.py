from pathlib import Path

INPUT_PATH = Path(__file__).parent / "input/"
OUTPUT_PATH = Path(__file__).parent / "output/"

DB_STRING = "postgresql://crke:135406@localhost:5432/mestrado_pesquisa"

PARANA_MIN_LATITUDE = -26.75
PARANA_MAX_LATITUDE = -22.44

PARANA_MIN_LONGITUDE = -55.00
PARANA_MAX_LONGITUDE = -48.00

MAX_ATTEMPTS_RANDOM_COORDINATE = 5000000

# This number is the max multiple of 7 >= 200 (which is about the biggest harvest_relative_day of occurrence)
MAX_HARVEST_RELATIVE_DAY = 301

EMERGENCE_DAYS_MIN = 5
EMERGENCE_DAYS_MAX = 8
