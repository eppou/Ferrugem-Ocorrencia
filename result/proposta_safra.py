from sqlalchemy import create_engine
from config import Config
from helpers.feature_importance import calculate_importance_avg, calculate_k_best, calculate_percentile
from helpers.input_output import get_latest_file, output_file
from helpers.result import write_result, read_result

import pandas as pd
import numpy as np
from datetime import datetime
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    median_absolute_error, root_mean_squared_error
)
from xgboost import XGBRegressor


# ==========================================================
# ------------------- 1. PREPARAÇÃO DE DADOS ----------------
# ==========================================================

def load_and_prepare_data() -> pd.DataFrame:
    """Carrega o CSV mais recente de features e prepara as colunas necessárias."""
    df = pd.read_csv(get_latest_file("features", "features_SI.csv"))

    df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')
    df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'], format='%Y-%m-%d')

    # Agrupar por ocorrência para achar o target
    df_target = df.sort_values('data').groupby('ocorrencia_id').tail(1).reset_index(drop=True)
    mapping = df_target.set_index('ocorrencia_id')['dias_desde_plantio'].to_dict()

    # Determinar safra (ano que a colheita termina)
    safra_year = np.where(df['data_ocorrencia'].dt.month >= 9,
                          df['data_ocorrencia'].dt.year,
                          df['data_ocorrencia'].dt.year - 1)
    safra_start = pd.to_datetime(safra_year.astype(str) + '-09-01')

    # Calcular target e features adicionais
    df['target'] = (df['data_ocorrencia'] - safra_start).dt.days
    df['dia_plantio'] = ((df['data'] - pd.to_timedelta(df['dias_desde_plantio'], unit='D')) - safra_start).dt.days

    # Determinar safra completa (ex: 2018 significa safra 2017/2018)
    df['safra'] = np.where(df['data_ocorrencia'].dt.month >= 9,
                           df['data_ocorrencia'].dt.year + 1,
                           df['data_ocorrencia'].dt.year)

    return df


# ==========================================================
# ------------------- 2. TREINO E TESTE ---------------------
# ==========================================================

def train_and_evaluate(df: pd.DataFrame, safra_teste: int,
                       model_type: str = "rf",
                       include_temporais: bool = True) -> dict:
    """Treina e avalia o modelo usando uma safra específica como teste."""

    df_train = df[df['safra'] != safra_teste].reset_index(drop=True).drop(columns=['safra'])
    df_test = df[df['safra'] == safra_teste].reset_index(drop=True).drop(columns=['safra'])

    # --- Escolha de features ---
    drop_cols = ['ocorrencia_id', 'data', 'data_ocorrencia', 'target']
    if not include_temporais:
        drop_cols += ['dias_desde_plantio', 'dia_plantio']  # remove colunas temporais

    X_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
    y_train = df_train['target']

    X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])
    y_test = df_test['target']

    # --- Escolha do modelo ---
    if model_type.lower() == "xgb":
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        )
        nome_modelo = "XGB_regressor_safra"
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        nome_modelo = "RF_regressor_safra"

    print(f"\n🌾 Treinando modelo {nome_modelo} para safra {safra_teste}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return compute_metrics(y_test, y_pred, safra_teste, nome_modelo)


# ==========================================================
# ------------------- 3. MÉTRICAS ---------------------------
# ==========================================================

def compute_metrics(y_true, y_pred, safra_teste: int, nome_modelo: str) -> dict:
    """Calcula todas as métricas de avaliação e retorna como dicionário."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)

    tolerance = 3
    diffs = y_pred - y_true
    TP = np.sum(np.abs(diffs) <= tolerance)
    FP = np.sum(diffs < -tolerance)
    FN = np.sum(diffs > tolerance)
    VN = len(y_true) - (TP + FP + FN)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Safra {safra_teste} → "
          f"R²: {r2:.3f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, "
          f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

    return {
        "ano": safra_teste,
        "nome_do_modelo": nome_modelo,
        "R2_medio": r2,
        "MAE_medio": mae,
        "RMSE_medio": rmse,
        "MedAE": medae,
        "VP": TP, "FP": FP, "FN": FN, "VN": VN,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


# ==========================================================
# ------------------- 4. EXECUÇÃO PRINCIPAL -----------------
# ==========================================================
def run(execution_started_at: datetime,
        cfg: Config,
        safras: list = None,
        model_type: str = "rf",
        include_temporais: bool = False,
        nome_do_modelo: str = None,
        output_path: str = "resultados.csv"):
    """Executa o pipeline completo de treino, teste e avaliação por safra."""
    df = load_and_prepare_data()
    safras_unicas = sorted(df['safra'].unique()) if safras is None else safras

    nome_do_modelo = nome_do_modelo or (model_type.upper() + ("_temp" if include_temporais else ""))

    resultados = []
    for safra in safras_unicas:
        res = train_and_evaluate(df, safra, model_type, include_temporais)
        res["timestamp_execucao"] = execution_started_at.strftime("%Y-%m-%d %H:%M:%S")
        resultados.append(res)

    resultados_df = pd.DataFrame(resultados)

    # === Renomear e ordenar colunas conforme solicitado ===
    resultados_df.rename(columns={
        "ano": "ano",
        "nome_do_modelo": "nome_do_modelo",
        "MAE_medio": "MAE_medio",
        "RMSE_medio": "RMSE_medio",
        "R2_medio": "R2_medio",
        "MedAE": "MedAE",
        "Precision": "Precision",
        "Recall": "Recall",
        "F1": "F1"
    }, inplace=True)

    # Criar coluna "Erro_dias" = MAE_medio (ou pode ser outra métrica)
    resultados_df["Erro_dias"] = resultados_df["MAE_medio"]

    # Reordenar colunas exatamente como desejado
    colunas_ordenadas = [
        "ano", "nome_do_modelo", "Erro_dias", "R2_medio", "MAE_medio",
        "RMSE_medio", "VP", "FP", "FN", "VN", "Recall", "Precision", "F1", "MedAE"
    ]
    resultados_df = resultados_df[colunas_ordenadas]

    # --- salvar (append) no CSV ---
    if os.path.exists(output_path):
        resultados_df.to_csv(output_path, mode='a', header=False, index=False)
        print(f"\n✅ Resultados adicionados ao arquivo existente: {output_path}")
    else:
        resultados_df.to_csv(output_path, index=False)
        print(f"\n✅ Novo arquivo de resultados criado: {output_path}")

    print_summary(resultados_df)
    return resultados_df



# ==========================================================
# ------------------- 5. RESUMO FINAL -----------------------
# ==========================================================

def print_summary(resultados_df: pd.DataFrame):
    """Imprime métricas médias gerais."""
    print("\n=== RESULTADO MÉDIO ===")
    print("R² médio: {:.3f}, MAE médio: {:.2f}, RMSE médio: {:.2f}, MedAE médio: {:.2f}, "
          "Precision médio: {:.2f}, Recall médio: {:.2f}, F1 médio: {:.2f}".format(
        resultados_df['R2_medio'].mean(),
        resultados_df['MAE_medio'].mean(),
        resultados_df['RMSE_medio'].mean(),
        resultados_df['MedAE'].mean(),
        resultados_df['Precision'].mean(),
        resultados_df['Recall'].mean(),
        resultados_df['F1'].mean()
    ))
