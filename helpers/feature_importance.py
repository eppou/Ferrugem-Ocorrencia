import pandas as pd


def calculate_importance_avg(importance: list[dict]) -> dict:
    a_avg = {}
    size = len(importance)

    for i in importance:
        for k in i.keys():
            a_avg[k] = 0

    for i in importance:
        for k, v in i.items():
            a_avg[k] += v

    a_avg_f = {}
    for k, v in i.items():
        a_avg_f[k] = a_avg[k] / size

    a_avg_sorted = dict(sorted(a_avg_f.items(), key=lambda item: item[1], reverse=True))

    return a_avg_sorted


def calculate_k_best(k: int, df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by="score", ascending=False)

    return df.head(k)


def calculate_percentile(p: int, df: pd.DataFrame) -> pd.DataFrame:
    scores = df["score"]
    t = np.percentile(scores, p)

    return df[(df["score"] > t)]
