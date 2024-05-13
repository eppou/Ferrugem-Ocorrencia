from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sqlalchemy import create_engine

from constants import DB_STRING
from helpers.input_output import get_latest_file
from helpers.result import write_result
from source.occurrence import get_safras

SEED = 85682938
K_FOLDS = 5
RESULT_FOLDER = "proposta"


def run(execution_started_at: datetime, safras: list = None):
    features_df = pd.read_csv(get_latest_file("features", "features_all.csv"))
    features_with_zero_df = pd.read_csv(get_latest_file("features", "features_with_zero_all.csv"))

    get_results(features_df, execution_started_at, "", safras)
    get_results(features_with_zero_df, execution_started_at, "with_zero", safras)


def get_results(features_df: pd.DataFrame, execution_started_at: datetime, result_description: str, safras: list = None):
    db_con_engine = create_engine(DB_STRING)
    conn = db_con_engine.connect()

    data_df = features_df
    data_df = data_df[data_df["data_ocorrencia"].notnull()]
    data_df = data_df.drop(columns=["Unnamed: 0"])

    data_all_safras_df = pd.DataFrame()

    if safras is None:
        safras = [s['safra'] for s in get_safras(conn)]
        safras = safras[1:]  # Removendo a última safra, pois está muito incompleta

    print("=====> Resultados para CADA safra")
    result_df_all_safras = pd.DataFrame()

    for safra in safras:
        data_df_safra = data_df.copy()

        print(f"- Filtrando por Safra: {safra}")
        data_df_safra = data_df_safra[data_df_safra["safra"] == safra]
        data_all_safras_df = pd.concat([data_all_safras_df, data_df_safra])

        data_df_safra = data_df_safra.reset_index()

        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

        print("- Calculando resultado para folds")
        result_df_all_folds = pd.DataFrame()
        for i, (train_indices, test_indices) in enumerate(kf.split(data_df_safra)):
            fold_num = i + 1

            train_x, train_y, test_x, test_y = prepare_train_test_for_fold(data_df_safra, train_indices, test_indices)

            result_df = train_test_model(train_x, train_y, test_x, test_y, safra, fold_num)

            result_df_all_folds = pd.concat([result_df_all_folds, result_df])

        write_result(RESULT_FOLDER, result_description, execution_started_at, result_df_all_folds, safra)

        result_df_all_safras = pd.concat([result_df_all_safras, result_df_all_folds])
        print("\n\n")

    write_result(RESULT_FOLDER, result_description, execution_started_at, result_df_all_safras, "all")

    print("=====> Resultados considerando TODAS as safras juntas")
    data_df_all = data_all_safras_df.copy()
    data_df_all.reset_index(inplace=True)

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    result_df_all_folds = pd.DataFrame()

    for i, (train_indices, test_indices) in enumerate(kf.split(data_df_all)):
        fold_num = i + 1

        train_x, train_y, test_x, test_y = prepare_train_test_for_fold(data_df_all, train_indices, test_indices)

        result_df = train_test_model(train_x, train_y, test_x, test_y, None, fold_num)

        result_df_all_folds = pd.concat([result_df_all_folds, result_df])

    write_result(RESULT_FOLDER, result_description, execution_started_at, result_df_all_folds, None)

    # plot the data for verification
    # ax = sns.scatterplot(x="precipitation_30d", y="precipitation_30d_count", hue="planting_relative_day",
    #                      data=pd.concat([x, y], axis=1), s=15)
    # ax.text(120, 23, "Distribuição do dia da safra das ocorrências", fontstyle="oblique", color="red")
    # plt.show()


def train_test_model(
        train_x: pd.DataFrame,
        train_y: pd.DataFrame,
        test_x: pd.DataFrame,
        test_y: pd.DataFrame,
        safra: str | None,
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
    # model = SVC(gamma="auto")

    model = RandomForestRegressor()
    model.fit(train_x, train_y.values.ravel())
    predicted_planting_relative_day_array = model.predict(test_x)

    result_df = test_y.copy()

    result_df["safra"] = safra

    result_df['predicted_planting_relative_day'] = predicted_planting_relative_day_array
    result_df['distance'] = abs(result_df['planting_relative_day'] - result_df['predicted_planting_relative_day'])

    result_df["fold"] = fold_number
    result_df.reset_index(inplace=True)

    # result_df['error'] = result_df['distance'] / result_df['planting_relative_day']
    # accuracy = result_df['error'].mean() * 100
    # print(f"=====> Fold: {fold_number}/{K_FOLDS} Resulting average error: %.2f%%" % accuracy)

    return result_df


def prepare_train_test_for_fold(df: pd.DataFrame, train_indices, test_indices) -> tuple:
    """Prepare train and test datasets, x and y. Return: train_x, train_y, test_x, test_y"""

    train_df = df.filter(items=train_indices, axis=0)
    test_df = df.filter(items=test_indices, axis=0)

    train_x = train_df.filter(axis=1, regex="precipitation_")
    train_y = train_df[["planting_relative_day"]].astype(int)

    test_x = test_df.filter(axis=1, regex="precipitation_")
    test_y = test_df[["planting_relative_day"]].astype(int)

    return train_x, train_y, test_x, test_y
