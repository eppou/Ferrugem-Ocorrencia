import sys
import time
from datetime import datetime
from typing import Callable

import data_preparation.prepare_ferrugem_ocorrencia_features as pof
import data_preparation.prepare_ferrugem_ocorrencia_instances as poi
import data_preparation.prepare_severity_per_occurrence as spo
import data_preparation.prepare_precipitation as pp
import result.test_baseline_model as baseline
import result.train_test_model as ttm
import testlab.download_cptec as dpp


def match_and_run():
    if len(sys.argv) < 2:
        return

    safras = [
        "2021/2022",
        "2020/2021",
        "2019/2020",
        "2018/2019",
        "2017/2018",
        "2016/2017",
        "2015/2016",
        "2014/2015",
        "2013/2014",
        "2012/2013",
        "2011/2012",
        "2010/2011",
    ]
    safras_param = [safras]

    command = sys.argv[1]
    match command:
        case "train_test_model":
            run(ttm.run)

        case "test_baseline_model":
            run(baseline.run)

        case "prepare_severity_per_occurrence":
            run(spo.run)

        case "prepare_precipitation":
            run(pp.run)

        case "download_precipitation":
            run(dpp.run)

        case "prepare_occurrence_instances":
            run(poi.run)

        case "prepare_occurrence_features":
            run(pof.run)

        case "pipeline":
            run(poi.run)
            run(pof.run)
            run(baseline.run)
            run(ttm.run)

        case "pipeline_results":
            run(baseline.run, safras_param)
            run(ttm.run, safras_param)

        case _:
            print(f"Unknown command: {command}")


def run(object_to_run: Callable, args: list = None):
    print(datetime.now().strftime("Execution started at %Y-%m-%d %I:%M:%S %p"))
    print()
    start = time.time()

    if args is None:
        object_to_run()
    else:
        object_to_run(*args)

    end = time.time()

    print()
    print(datetime.now().strftime("Execution ended at %Y-%m-%d %I:%M:%S %p"))
    execution_ms = round((end - start) * 10 ** 3)
    execution_s = round(execution_ms / 1000.0, 2)
    print(f"Execution took {execution_ms}ms ({execution_s}s)")


if __name__ == "__main__":
    match_and_run()
