from datetime import date
import random


def determine_random_date(min_date: date, max_date: date) -> date:
    random_date = min_date + (max_date - min_date) * random.random()

    return random_date
