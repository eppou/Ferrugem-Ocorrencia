import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from constants import OUTPUT_PATH

SEED = 5
TEST_SIZE = 0.10


def run():
    # collecting and preparing the data
    data = pd.read_csv(OUTPUT_PATH / "instances_features_dataset.csv", sep=",")

    x = data[[
        "precipitation_15d", "precipitation_15d", "precipitation_30d", "precipitation_45d",
        "precipitation_60d", "precipitation_75d", "precipitation_90d",
        "precipitation_15d_count", "precipitation_30d_count", "precipitation_45d_count",
        "precipitation_60d_count", "precipitation_75d_count", "precipitation_90d_count"
    ]]
    y = data[["ocorrencia"]].astype(int)

    # segmenting the dataset for training
    np.random.seed(SEED)
    train_x_raw, test_x_raw, train_y, test_y = train_test_split(
        x, y,
        test_size=TEST_SIZE,
        stratify=y,
    )

    print(f"===> Total number of entries: {x.shape[0]}")
    print(f"===> Using {train_x_raw.shape[0]} for train, {test_x_raw.shape[0]} for test ({round(TEST_SIZE * 100)}%)")

    # scaler = StandardScaler()
    # scaler.fit(train_x_raw)
    # train_x = scaler.transform(train_x_raw)
    # test_x = scaler.transform(test_x_raw)

    # dummy classifier accuracy: To be used as a baseline
    dummy_mostfrequent = DummyClassifier(strategy="most_frequent")
    dummy_mostfrequent.fit(train_x_raw, train_y)
    dummy_accuracy = dummy_mostfrequent.score(test_x_raw, test_y) * 100
    print("===> DUMMY resulting accuracy: %.2f%%" % dummy_accuracy)

    # train the model and test, show accuracy results
    model = SVC(gamma="auto")
    model.fit(train_x_raw, train_y.values.ravel())
    predicted_results = model.predict(test_x_raw)

    accuracy = accuracy_score(test_y, predicted_results) * 100
    print("===> Resulting accuracy: %.2f%%" % accuracy)

    # plot the data for verification
    ax = sns.scatterplot(x="precipitation_30d", y="precipitation_30d_count", hue="ocorrencia",
                    data=pd.concat([x, y], axis=1), s=15)
    ax.text(120, 23, "Distribuição das ocorrências", fontstyle="oblique", color="red")
    plt.show()
