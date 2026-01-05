from sqlalchemy import create_engine
from config import Config
from helpers.input_output import get_latest_file
from datetime import datetime
from joblib import Parallel, delayed

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score, mean_absolute_error, median_absolute_error, root_mean_squared_error
)
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


# ================================================================
# 1. PREPARAÇÃO DE DADOS
# ================================================================
def load_and_prepare_data() -> pd.DataFrame:
    df = pd.read_csv(get_latest_file("features", "features_SI.csv"))
    df['data'] = pd.to_datetime(df['data'])
    df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'])

    # Target: dias_desde_plantio da ocorrência
    df_target = (
        df.sort_values('data')
          .groupby('ocorrencia_id')
          .tail(1)
          .reset_index(drop=True)
    )
    mapping = df_target.set_index('ocorrencia_id')['dias_desde_plantio'].to_dict()
    df['target'] = df['ocorrencia_id'].map(mapping)

    # Determinar safra
    df['safra'] = np.where(
        df['data_ocorrencia'].dt.month >= 9,
        df['data_ocorrencia'].dt.year + 1,
        df['data_ocorrencia'].dt.year
    )

    return df


# ================================================================
# 2. FUNÇÕES DE MÉTRICA
# ================================================================
def regression_metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "medae": median_absolute_error(y_true, y_pred)
    }


def tolerance_metrics(y_true, y_pred, tolerance=3):
    diffs = y_pred - y_true
    TP = np.sum(np.abs(diffs) <= tolerance)
    FP = np.sum(diffs < -tolerance)
    FN = np.sum(diffs > tolerance)
    VN = len(y_true) - (TP + FP + FN)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1, "TP": TP, "FP": FP, "FN": FN, "VN": VN}


# ================================================================
# 3. TREINAMENTO E AVALIAÇÃO DE UMA SAFRA
# ================================================================
def train_and_evaluate_safra(df: pd.DataFrame,
                             safra_teste: int,
                             model_type: str = "xgb",
                             include_temporais: bool = True,
                             show_importance: bool = False,
                             tolerance=3) -> dict:

    df_train = df[df['safra'] != safra_teste].reset_index(drop=True)
    df_test = df[df['safra'] == safra_teste].reset_index(drop=True)

    if df_test.empty or df_train.empty:
        return None

    # --- Controle das colunas ---
    drop_cols = ['ocorrencia_id', 'data', 'data_ocorrencia', 'target', 'safra']
    if not include_temporais:
        for c in ['dias_desde_plantio', 'estadio_fenologico']:
            if c in df_train.columns:
                drop_cols.append(c)

    X = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
    y = df_train['target']
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
        nome_modelo = "XGBoost_regressao_talhao"
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            n_jobs=-1,
            random_state=42
        )
        nome_modelo = "RF_regressao_talhao"
        
    if include_temporais:
        nome_modelo += "_temp"

    # ========== Validação Cruzada ==========
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2s, maes, rmses, precisions, recalls, f1s = [], [], [], [], [], []

    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        reg = regression_metrics(y_valid, y_pred)
        tol = tolerance_metrics(y_valid, y_pred, tolerance)

        r2s.append(reg["r2"])
        maes.append(reg["mae"])
        rmses.append(reg["rmse"])
        precisions.append(tol["precision"])
        recalls.append(tol["recall"])
        f1s.append(tol["f1"])

    # ========== Treino final ==========
    model.fit(X, y)
    y_pred_test = model.predict(X_test)

    reg_final = regression_metrics(y_test, y_pred_test)
    tol_final = tolerance_metrics(y_test, y_pred_test, tolerance)

    # ========== Print (formato CSV) ==========
    erro_dias = mean_absolute_error(y_test, y_pred_test)
    print(
        f"{safra_teste},{nome_modelo},{erro_dias:.2f},{reg_final['r2']:.3f},{reg_final['mae']:.2f},{reg_final['rmse']:.2f},"
        f"{tol_final['TP']},{tol_final['FP']},{tol_final['FN']},{tol_final['VN']},"
        f"{tol_final['recall']:.2f},{tol_final['precision']:.2f},{reg_final['medae']:.2f},{tol_final['f1']:.2f}"
    )

    #mostra a feature importance
    if show_importance:
        importances = model.feature_importances_
        feature_names = X.columns
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        print("\nFeature Importances:")
        for index, row in feature_importance_df.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")
            
    return {
        "ano": safra_teste,
        "nome_do_modelo": nome_modelo,
        "Erro_dias": erro_dias,
        "R2_medio": reg_final['r2'],
        "MAE_medio": reg_final['mae'],
        "RMSE_medio": reg_final['rmse'],
        "VP": tol_final['TP'],
        "FP": tol_final['FP'],
        "FN": tol_final['FN'],
        "VN": tol_final['VN'],
        "Recall": tol_final['recall'],
        "Precision": tol_final['precision'],
        "MedAE": reg_final['medae'],
        "F1": tol_final['f1']
    }


# ================================================================
# 4. EXECUÇÃO PRINCIPAL (PARALELA)
# ================================================================
def run(execution_started_at: datetime,
        cfg: Config,
        safras: list = None,
        n_jobs=-1,
        model_type="RF",
        include_temporais=False,
        show_importance=False,
        output_path="resultados.csv"):

    df = load_and_prepare_data()
    todas_safras = sorted(df['safra'].unique()) if safras is None else safras

    print(f"\n🚀 Iniciando treinamento para {len(todas_safras)} safras... (modelo={model_type}, temporais={include_temporais})")

    resultados = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(train_and_evaluate_safra)(df, safra_teste, model_type, include_temporais,show_importance)
        for safra_teste in todas_safras
    )

    resultados = [r for r in resultados if r is not None]

    if not resultados:
        print("⚠️ Nenhum resultado válido gerado.")
        return

    df_resultados = pd.DataFrame(resultados)

    # === Garantir colunas e ordem conforme formato solicitado ===
    colunas_ordenadas = [
        "ano", "nome_do_modelo", "Erro_dias", "R2_medio", "MAE_medio",
        "RMSE_medio", "VP", "FP", "FN", "VN", "Recall", "Precision", "F1", "MedAE"
    ]

    # Criar colunas ausentes caso faltem (para robustez)
    for col in colunas_ordenadas:
        if col not in df_resultados.columns:
            df_resultados[col] = np.nan

    df_resultados = df_resultados[colunas_ordenadas]

    # --- Salvar CSV (append) com cabeçalho fixo ---
    if os.path.exists(output_path):
        df_resultados.to_csv(output_path, mode='a', header=False, index=False)
        print(f"\n✅ Resultados adicionados ao arquivo existente: {output_path}")
    else:
        df_resultados.to_csv(output_path, index=False)
        print(f"\n✅ Novo arquivo criado: {output_path}")

    # --- Resumo geral ---
    print("\n=== MÉDIA GERAL ===")
    print(
        f"R² médio: {df_resultados['R2_medio'].mean():.3f}, "
        f"MAE médio: {df_resultados['MAE_medio'].mean():.2f}, "
        f"RMSE médio: {df_resultados['RMSE_medio'].mean():.2f}, "
        f"Precision: {df_resultados['Precision'].mean():.2f}, "
        f"Recall: {df_resultados['Recall'].mean():.2f}, "
        f"F1: {df_resultados['F1'].mean():.2f}"
    )

    return df_resultados
