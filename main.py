import inspect
import sys
import time
from datetime import datetime
from typing import Callable
import configparser

import data_preparation.features as features
import data_preparation.features_with_zero as features_with_zero
import data_preparation.features_with_augmentation as features_with_augmentation
import data_preparation.instances as instances
import data_preparation.precipitation as precipitation
import data_preparation.severity as severity
import result.concorrente as concorrente
import result.proposta_kemmer as proposta_kemmer
import result.proposta_classificacao as proposta_classifica
import result.proposta_classificacao_hibrida as proposta_classificacao_hibrida
import result.proposta_safra as proposta_safra
import result.proposta_talhao as proposta_talhao
import script.output_load_jupyter as output_load_jupyter
import testlab.download_cptec as download_precipitation
import result.evolution_maps as evolution_maps
import threshold as threshold
from config import Config, DatabaseConfig


def match_and_run(cfg: Config):
    if len(sys.argv) < 2:
        raise RuntimeError("Missing arguments")

    command = sys.argv[1]
    match command:
        case "proposta_kemmer":
            run([
                (proposta_kemmer.run, [cfg]),
            ])
            
        case "proposta_classificacao":
            run([
                (proposta_classifica.run, [cfg]),
            ])
            
        case "proposta_classificacao_hibrida":
            run([
                (proposta_classificacao_hibrida.run, [cfg]),
            ])
            
        case "proposta_regressao_talhao":
            run([
                (proposta_talhao.run, [cfg]),
            ])
            
        case "proposta_regressao_safra":
            run([
                (proposta_safra.run, [cfg]),
            ])

        case "berruski":
            run([
                (concorrente.run, [cfg]),
            ])

        case "severity":
            run([
                (severity.run, [cfg, False]),
            ])

        case "precipitation":
            run([
                (precipitation.run, [cfg]),
            ])

        case "download_precipitation":
            run([
                download_precipitation.run,
            ])

        case "instances":
            run([
                (instances.run, [cfg]),
            ])

        case "features":
            run([
                (features.run, [cfg])
            ])

        case "features_with_zero":
            run([
                features_with_zero.run,
            ])
        
        case "features_with_augmentation":
            run([
                (features_with_augmentation.run, [cfg]),
            ])

        case "evolution_maps":
            run([
                (evolution_maps.run, [cfg]),
            ])
            
        case "threshold":
            run([
                (threshold.run, [cfg]),
            ])   
            
        case "pipeline":
            run([
                (instances.run, [cfg]),
                # (severity.run, [cfg, False]),
                (features.run, [cfg]),
                features_with_zero.run,
                (concorrente.run, [cfg]),
                (proposta_kemmer.run, [cfg]),
            ])

        case "pipeline_results":
            run([
                (concorrente.run, [cfg]),
                (proposta_kemmer.run, [cfg]),
            ])

        case "output_load_jupyter":
            run([
                output_load_jupyter.run,
            ])

        case _:
            raise RuntimeError(f"Unknown command: {command}")


def run(objects_to_run: list[Callable|tuple[Callable, list]]):
    execution_started_at = datetime.now()
    start = execution_started_at.timestamp()

    print(execution_started_at.strftime(">>>>> Execution started at %Y-%m-%d %I:%M:%S %p <<<<<"))

    for object_to_run_t in objects_to_run:
        if type(object_to_run_t) is tuple:
            object_to_run = object_to_run_t[0]
            args = object_to_run_t[1]
        else:
            object_to_run = object_to_run_t
            args = None

        object_execution_started_at = datetime.now()
        print()
        print(object_execution_started_at.strftime(
            f"Start {inspect.getmodule(object_to_run).__name__}: Single execution started %Y-%m-%d %I:%M:%S %p %z"
        ))

        if args is None or not args:
            object_to_run(execution_started_at)
        else:
            object_to_run(execution_started_at, *args)

        print(datetime.now().strftime(
            f"End {inspect.getmodule(object_to_run).__name__}: Single execution ended at %Y-%m-%d %I:%M:%S %p %z")
        )

    end = time.time()
    print()
    print(datetime.now().strftime(">>>>> Execution ended at %Y-%m-%d %I:%M:%S %p <<<<<"))

    execution_ms = round((end - start) * 10 ** 3)
    execution_s = round(execution_ms / 1000.0, 2)
    print(f">>>>> Execution took {execution_ms}ms ({execution_s}s) <<<<<")
    print()

def parse_config() -> Config:
    config = configparser.ConfigParser()
    config.read(r"config.cfg")

    return Config(
        DatabaseConfig(
            dbstring=config.get("database", "dbstring")
        )
    )


if __name__ == "__main__":
    match_and_run(parse_config())
