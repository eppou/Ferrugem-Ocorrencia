from datetime import datetime

import pandas as pd
from natsort import natsort_keygen
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, SelectPercentile, r_regression
from sklearn.model_selection import KFold
from sqlalchemy import create_engine

from constants import DB_STRING
from helpers.feature_importance import calculate_importance_avg
from helpers.input_output import get_latest_file
from helpers.result import write_result
from source.occurrence import get_safras

SEED = 85682938
K_FOLDS = 5
RESULT_FOLDER = "proposta"

FEATURE_SELECTION_K_BEST = 5
FEATURE_SELECTION_PERCENTILE = 10


def run(execution_started_at: datetime, safras: list = None):
    features_df = pd.read_csv(get_latest_file("features", "features_all.csv"))

    get_results(features_df, execution_started_at, "", safras)


def get_results(
        features_df: pd.DataFrame,
        execution_started_at: datetime,
        result_description: str,
        safras: list = None
):
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
    result_df_all_harvests = pd.DataFrame()
    feature_importance_all_harvests = []

    for safra in safras:
        data_df_safra = data_df.copy()

        print(f"- Filtrando por Safra: {safra}")
        data_df_safra = data_df_safra[data_df_safra["safra"] == safra]
        data_all_safras_df = pd.concat([data_all_safras_df, data_df_safra])

        data_df_safra = data_df_safra.reset_index()

        X, y = prepare_X_y(data_df_safra)

        # Selecionando K melhores features
        X_KBest = SelectKBest(r_regression, k=FEATURE_SELECTION_K_BEST).fit_transform(X, y.values.ravel())
        print(f"- Safra {safra} - SelectKBest: before: {X.shape} / after: {X_KBest.shape}")

        # Selecionando features dentre os 10% melhores
        X_Percentile = SelectPercentile(r_regression, percentile=10).fit_transform(X, y.values.ravel())
        print(f"- Safra {safra} - SelectPercentile: before: {X.shape} / after: {X_Percentile.shape}")

        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

        print("- Calculando resultado para folds")
        result_df_all_folds = pd.DataFrame()
        feature_importance_harvest = []
        for i, (train_indices, test_indices) in enumerate(kf.split(data_df_safra)):
            fold_num = i + 1

            train_x, train_y, test_x, test_y = filter_train_test_for_fold(X, y, train_indices, test_indices)

            result_df, importance = train_test_model(train_x, train_y, test_x, test_y, safra, fold_num)

            feature_importance_harvest.append(importance)
            feature_importance_all_harvests.append(importance)
            result_df_all_folds = pd.concat([result_df_all_folds, result_df])

        write_result(RESULT_FOLDER, result_description, execution_started_at, result_df_all_folds, safra)

        print(f"- Feature importance para safra {safra}")
        # Calculando média dos feature importances para todos os folds, safra
        feature_importance_harvest_avg = calculate_importance_avg(feature_importance_harvest)
        feature_importance_harvest_avg_arr = [(k, v) for k, v in feature_importance_harvest_avg.items()]
        feature_importance_harvest_df = pd.DataFrame(feature_importance_harvest_avg_arr, columns=["feature", "score"])
        feature_importance_harvest_df = feature_importance_harvest_df.sort_values(by=["score"], key=natsort_keygen(), ascending=False)

        for i in feature_importance_harvest_avg_arr:
            print(f"{safra}: Feature {i[0]}: Score {i[1]:.5f}")

        write_result(RESULT_FOLDER, result_description, execution_started_at, feature_importance_harvest_df, safra, "importances")

        result_df_all_harvests = pd.concat([result_df_all_harvests, result_df_all_folds])
        print("\n\n")

    feature_importance_all_harvests_avg = calculate_importance_avg(feature_importance_all_harvests)
    feature_importance_all_harvests_avg_arr = [(k, v) for k, v in feature_importance_all_harvests_avg.items()]
    feature_importance_all_harvests_df = pd.DataFrame(feature_importance_all_harvests_avg_arr, columns=["feature", "score"])
    feature_importance_all_harvests_df = feature_importance_all_harvests_df.sort_values(by=["score"], key=natsort_keygen(), ascending=False)

    write_result(RESULT_FOLDER, result_description, execution_started_at, feature_importance_all_harvests_df, "all", "importances")
    write_result(RESULT_FOLDER, result_description, execution_started_at, result_df_all_harvests, "all")

    print("=====> Resultados considerando TODAS as safras juntas")
    data_df_all = data_all_safras_df.copy()
    data_df_all.reset_index(inplace=True)

    X, y = prepare_X_y(data_df_all)

    # Selecionando K melhores features
    X_KBest = SelectKBest(r_regression, k=FEATURE_SELECTION_K_BEST).fit_transform(X, y.values.ravel())
    print(f"- TODAS as safras - SelectKBest: before: {X.shape} / after: {X_KBest.shape}")

    # Selecionando features dentre os 10% melhores
    X_Percentile = SelectPercentile(r_regression, percentile=10).fit_transform(X, y.values.ravel())
    print(f"- TODAS as safras - SelectPercentile: before: {X.shape} / after: {X_Percentile.shape}")

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    result_df_all_folds = pd.DataFrame()

    print("- Calculando resultado para folds")
    importance_all = []
    for i, (train_indices, test_indices) in enumerate(kf.split(data_df_all)):
        fold_num = i + 1

        train_x, train_y, test_x, test_y = filter_train_test_for_fold(X, y, train_indices, test_indices)

        result_df, importance = train_test_model(train_x, train_y, test_x, test_y, None, fold_num)

        result_df_all_folds = pd.concat([result_df_all_folds, result_df])
        importance_all.append(importance)

    write_result(RESULT_FOLDER, result_description, execution_started_at, result_df_all_folds, None)

    print("- Feature importance para TODAS as safras juntas")
    # Feature importances para todas as safras juntas
    feature_importance_all_avg = calculate_importance_avg(importance_all)
    feature_importance_all_avg_arr = [(k, v) for k, v in feature_importance_all_avg.items()]
    feature_importance_all_df = pd.DataFrame(feature_importance_all_avg_arr, columns=["feature", "score"])
    feature_importance_all_df = feature_importance_all_df.sort_values(by=["score"], key=natsort_keygen(), ascending=False)

    for k, v in feature_importance_all_avg.items():
        print(f"Feature {k}: Score {v:.5f}")

    write_result(RESULT_FOLDER, f"{result_description}_importances", execution_started_at, feature_importance_all_df, None, "importances")

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
) -> (pd.DataFrame, dict):
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

    # Get importance
    importance = model.feature_importances_

    # Map column names to numbers
    c = list(train_x.columns.values.tolist())
    column_name_map = {}
    for n, name in enumerate(c):
        column_name_map[n] = name

    # Summarize feature importance
    importance_d = {}
    for i, v in enumerate(importance):
        column_name = column_name_map[i]
        importance_d[column_name] = v

    return result_df, importance_d


def prepare_X_y(df: pd.DataFrame) -> tuple:
    X = df.filter(axis=1, regex="precipitation_")
    y = df[["planting_relative_day"]].astype(int)

    # Preserve index
    X.index = df.index
    y.index = df.index

    return X, y


def filter_train_test_for_fold(X: pd.DataFrame, y: pd.DataFrame, train_indices, test_indices) -> tuple:
    train_X_df = X.filter(items=train_indices, axis=0)
    train_y_df = y.filter(items=train_indices, axis=0)

    test_X_df = X.filter(items=test_indices, axis=0)
    test_y_df = y.filter(items=test_indices, axis=0)

    return train_X_df, train_y_df, test_X_df, test_y_df
