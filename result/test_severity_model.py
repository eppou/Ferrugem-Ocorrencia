import pandas as pd
from constants import OUTPUT_PATH
import math

"""
Calcula o erro do resultado do modelo de Del Ponte (adaptado) em relação à data de ocorrência real
"""


def run():
    df = pd.read_csv(OUTPUT_PATH / "test_severity_model_results.csv", sep=",")

    # calculating average accuracy for 5d, 10d and 15d
    df['distance_5d'] = pow(df['day_in_harvest_5d'] - df['day_in_harvest'], 2)
    # df['day_in_harvest_error_5d'] = df['distance_5d'] / df['day_in_harvest']

    df['distance_10d'] = pow(df['day_in_harvest_10d'] - df['day_in_harvest'], 2)
    # df['day_in_harvest_error_10d'] = df['distance_10d'] / df['day_in_harvest']

    df['distance_15d'] = pow(df['day_in_harvest_15d'] - df['day_in_harvest'], 2)
    # df['day_in_harvest_error_15d'] = df['distance_15d'] / df['day_in_harvest']

    error_rmse = {
        'error_5d': math.sqrt(df['distance_5d'].sum() / df.shape[0]),
        'error_10d': math.sqrt(df['distance_10d'].sum() / df.shape[0]),
        'error_15d': math.sqrt(df['distance_15d'].sum() / df.shape[0]),
    }
    best_error_rmse = min(error_rmse, key=error_rmse.get)

    print(f"======> Average error for 5d: {error_rmse['error_5d']}")
    print(f"======> Average error for 10d: {error_rmse['error_10d']}")
    print(f"======> Average error for 15d: {error_rmse['error_15d']}")
    print()
    print(f"======> BEST average error: {best_error_rmse} ({round(error_rmse[best_error_rmse], 2)})")
