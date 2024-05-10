import sys
import time
from datetime import datetime
from typing import Callable
import inspect

import data_preparation.features as features
import data_preparation.features_with_zero as features_with_zero
import data_preparation.instances as instances
import data_preparation.precipitation as precipitation
import data_preparation.severity as severity
import result.concorrente as concorrente
import result.proposta as proposta
import script.output_load_jupyter as output_load_jupyter
import testlab.download_cptec as download_precipitation


def match_and_run():
    if len(sys.argv) < 2:
        return

    command = sys.argv[1]
    match command:
        case "proposta":
            run(proposta.run)

        case "concorrente":
            run(concorrente.run)

        case "severity":
            run(severity.run, [False])

        case "precipitation":
            run(precipitation.run)

        case "download_precipitation":
            run(download_precipitation.run)

        case "instances":
            run(instances.run)

        case "features":
            run(features.run)

        case "features_with_zero":
            run(features_with_zero.run)

        case "pipeline":
            run(instances.run)
            run(severity.run)
            run(features.run)
            run(features_with_zero.run)
            run(concorrente.run)
            run(proposta.run)

        case "pipeline_results":
            run(concorrente.run)
            run(proposta.run)

        case "output_load_jupyter":
            run(output_load_jupyter.run)

        case _:
            print(f"Unknown command: {command}")


def run(object_to_run: Callable, args: list = None):
    print(datetime.now().strftime(f"{inspect.getmodule(object_to_run).__name__}: Execution started at %Y-%m-%d %I:%M:%S %p"))
    print()
    start = time.time()

    if args is None or not args:
        object_to_run()
    else:
        object_to_run(*args)

    end = time.time()

    print()
    print(datetime.now().strftime(f"{inspect.getmodule(object_to_run).__name__}: Execution ended at %Y-%m-%d %I:%M:%S %p"))
    execution_ms = round((end - start) * 10 ** 3)
    execution_s = round(execution_ms / 1000.0, 2)
    print(f"Execution took {execution_ms}ms ({execution_s}s)")
    print()


if __name__ == "__main__":
    match_and_run()
