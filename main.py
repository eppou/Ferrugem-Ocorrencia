import sys
import time
from datetime import datetime
from typing import Callable

import data_preparation.prepare_severity_model as psm
import data_preparation.prepare_severity_per_occurrence as spo
import result.test_severity_model as smt
import result.train_test_model as ttm


def match_and_run():
    if len(sys.argv) < 2:
        return

    command = sys.argv[1]
    match command:
        case "train_test_model":
            run(ttm.run)
        case "test_severity_model":
            run(smt.run)
        case "prepare_severity_model":
            run(psm.run)
        case "prepare_severity_per_occurrence":
            run(spo.run)
        case _:
            print(f"Unknown command: {command}")


def run(object_to_run: Callable):
    print(datetime.now().strftime("Execution started at %Y-%m-%d %I:%M:%S %p"))
    print()
    start = time.time()

    object_to_run()

    end = time.time()

    print()
    print(datetime.now().strftime("Execution ended at %Y-%m-%d %I:%M:%S %p"))
    print(f"Execution took {round((end - start) * 10 ** 3)} ms")


if __name__ == "__main__":
    match_and_run()
