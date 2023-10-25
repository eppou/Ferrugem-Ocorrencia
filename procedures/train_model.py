from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from constants import OUTPUT_PATH
import pandas as pd


def run():
    df = pd.read_csv(OUTPUT_PATH / "ferrugem_ocorrencia_dataset.csv", sep=",")

    x = df[[
        "precipitation_15d", "precipitation_15d", "precipitation_30d", "precipitation_45d",
        "precipitation_60d", "precipitation_75d", "precipitation_90d",
        "precipitation_15d_count", "precipitation_30d_count", "precipitation_45d_count",
        "precipitation_60d_count", "precipitation_75d_count", "precipitation_90d_count"
    ]]
    y = df[["ocorrencia"]].astype(int)

    split_threshold = 650
    print(f"Total number of entries: {df.shape[0]}")
    print(f"Picking {split_threshold} entries for training ({split_threshold/df.shape[0] * 100}% of the dataset)")

    train_x = x[:split_threshold]
    train_y = y[:split_threshold]

    test_x = x[split_threshold:]
    test_y = y[split_threshold:]

    model = SVC(gamma="auto")
    model.fit(train_x, train_y.values.ravel())
    predicted_results = model.predict(test_x)

    accuracy = accuracy_score(test_y, predicted_results) * 100
    print("A acurácia foi %.2f%%" % accuracy)
