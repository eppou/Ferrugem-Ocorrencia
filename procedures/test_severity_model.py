import pandas as pd
from constants import OUTPUT_PATH


def run():
    df = pd.read_csv(OUTPUT_PATH / "test_severity_model_results.csv", sep=",")

    # calculating average accuracy for 5d, 10d and 15d
    df['distance_5d'] = abs(df['day_in_harvest_5d'] - df['day_in_harvest'])
    df['day_in_harvest_error_5d'] = df['distance_5d'] / df['day_in_harvest']

    df['distance_10d'] = abs(df['day_in_harvest_10d'] - df['day_in_harvest'])
    df['day_in_harvest_error_10d'] = df['distance_10d'] / df['day_in_harvest']

    df['distance_15d'] = abs(df['day_in_harvest_15d'] - df['day_in_harvest'])
    df['day_in_harvest_error_15d'] = df['distance_15d'] / df['day_in_harvest']

    avg_error = {
        'avg_error_5d': df['day_in_harvest_error_5d'].mean(),
        'avg_error_10d': df['day_in_harvest_error_10d'].mean(),
        'avg_error_15d': df['day_in_harvest_error_15d'].mean(),
    }
    best_avg_error = min(avg_error, key=avg_error.get)

    print(f"======> Average error for 5d: {avg_error['avg_error_5d']}")
    print(f"======> Average error for 10d: {avg_error['avg_error_10d']}")
    print(f"======> Average error for 15d: {avg_error['avg_error_15d']}")
    print()
    print(f"======> BEST average error: {best_avg_error} ({round(avg_error[best_avg_error] * 100, 2)}%)")
