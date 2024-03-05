import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from constants import OUTPUT_PATH

SEED = 85682938
K_FOLDS = 5


def run():
    data_df = pd.read_csv(OUTPUT_PATH / "instances_features_dataset.csv")
    data_df = data_df[data_df["data_ocorrencia"].notnull()]

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    result_df_all_folds = pd.DataFrame()
    for i, (train_indices, test_indices) in enumerate(kf.split(data_df)):
        fold_num = i + 1

        train_x, train_y, test_x, test_y = prepare_train_test_for_fold(data_df, train_indices, test_indices)

        result_df = train_test_model(train_x, train_y, test_x, test_y, fold_num)

        result_df_all_folds = pd.concat([result_df_all_folds, result_df])

    write_result(result_df_all_folds)

    # plot the data for verification
    # ax = sns.scatterplot(x="precipitation_30d", y="precipitation_30d_count", hue="day_in_harvest",
    #                      data=pd.concat([x, y], axis=1), s=15)
    # ax.text(120, 23, "Distribuição do dia da safra das ocorrências", fontstyle="oblique", color="red")
    # plt.show()


def train_test_model(
        train_x: pd.DataFrame,
        train_y: pd.DataFrame,
        test_x: pd.DataFrame,
        test_y: pd.DataFrame,
        fold_number: int,
):
    print(f"=====> Fold: {fold_number}/{K_FOLDS}  Using {train_x.shape[0]} for train, {test_x.shape[0]} for test")

    # scaler = StandardScaler()
    # scaler.fit(train_x)
    # train_x = scaler.transform(train_x)
    # test_x = scaler.transform(test_x)

    # dummy classifier accuracy: To be used as a baseline
    # dummy_mostfrequent = DummyClassifier(strategy="most_frequent")
    # dummy_mostfrequent.fit(train_x, train_y)
    # dummy_accuracy = dummy_mostfrequent.score(test_x, test_y) * 100
    # print("===> DUMMY resulting accuracy: %.2f%%" % dummy_accuracy)

    # train the model and test, show accuracy results
    model = SVC(gamma="auto")
    model.fit(train_x, train_y.values.ravel())
    predicted_harvest_relative_day_array = model.predict(test_x)

    result_df = test_y.copy()

    result_df = result_df.rename(columns={"day_in_harvest": "harvest_relative_day"})

    result_df['predicted_harvest_relative_day'] = predicted_harvest_relative_day_array
    result_df['distance'] = abs(result_df['harvest_relative_day'] - result_df['predicted_harvest_relative_day'])


    result_df["fold"] = fold_number
    result_df.reset_index(inplace=True)

    # result_df['error'] = result_df['distance'] / result_df['harvest_relative_day']
    # accuracy = result_df['error'].mean() * 100
    # print(f"=====> Fold: {fold_number}/{K_FOLDS} Resulting average error: %.2f%%" % accuracy)

    return result_df


def prepare_train_test_for_fold(df: pd.DataFrame, train_indices, test_indices) -> tuple:
    """Prepare train and test datasets, x and y. Return: train_x, train_y, test_x, test_y"""

    train_df = df.filter(items=train_indices, axis=0)
    test_df = df.filter(items=test_indices, axis=0)

    train_x = train_df[[
        "precipitation_15d", "precipitation_30d", "precipitation_45d",
        "precipitation_60d", "precipitation_75d", "precipitation_90d",
        "precipitation_15d_count", "precipitation_30d_count", "precipitation_45d_count",
        "precipitation_60d_count", "precipitation_75d_count", "precipitation_90d_count"
    ]]
    train_y = train_df[["day_in_harvest"]].astype(int)

    test_x = test_df[[
        "precipitation_15d", "precipitation_30d", "precipitation_45d",
        "precipitation_60d", "precipitation_75d", "precipitation_90d",
        "precipitation_15d_count", "precipitation_30d_count", "precipitation_45d_count",
        "precipitation_60d_count", "precipitation_75d_count", "precipitation_90d_count"
    ]]
    test_y = test_df[["day_in_harvest"]].astype(int)

    return train_x, train_y, test_x, test_y


def write_result(result_df: pd.DataFrame):
    filename = "train_test_model_results_all.csv"
    result_df.to_csv(OUTPUT_PATH / filename)
