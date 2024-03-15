import pandas as pd


def calculate_threshold_for_baseline_model(train_df: pd.DataFrame) -> tuple:
    """Finding the thresholds for 5, 10 and 15 days (average method)"""
    threshold_5d = train_df["severity_acc_5d_before_occurrence"].mean()
    threshold_10d = train_df["severity_acc_10d_before_occurrence"].mean()
    threshold_15d = train_df["severity_acc_15d_before_occurrence"].mean()

    print(f"=====> threshold_5d: {threshold_5d} | threshold_10d: {threshold_10d} | threshold_15d: {threshold_15d}")

    return threshold_5d, threshold_10d, threshold_15d
