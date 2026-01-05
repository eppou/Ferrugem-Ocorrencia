import joblib
from sqlalchemy import create_engine
from config import Config
from helpers.feature_importance import calculate_importance_avg, calculate_k_best, calculate_percentile
from helpers.input_output import get_latest_file, output_file
from helpers.result import write_result, read_result

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor


# ============================================================
#  CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ============================================================

def carregar_dados():
    """Carrega e prepara o dataset de features."""
    df = pd.read_csv(get_latest_file("features", "features_SI.csv"))
    df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')
    df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'], format='%Y-%m-%d')

    # Determina safra
    df['safra'] = np.where(
        df['data_ocorrencia'].dt.month >= 9,
        df['data_ocorrencia'].dt.year,
        df['data_ocorrencia'].dt.year - 1
    )

    return df


# ===========================================================
#  DIVISÃO POR SAFRA E BALANCEAMENTO
# ============================================================

def dividir_por_safra(df, safra_teste):
    """Divide o dataset em treino e teste com base na safra."""
    df_grouped = df.groupby('ocorrencia_id')
    df_grouped_test = [g for g in df_grouped if g[1]['safra'].iloc[0] == safra_teste]
    df_grouped_train = [g for g in df_grouped if g[1]['safra'].iloc[0] != safra_teste]

    if len(df_grouped_test) == 0:
        print(f" Safra {safra_teste} sem dados para teste.")
        return None, None

    df_train = pd.concat([group for _, group in df_grouped_train])
    df_test = pd.concat([group for _, group in df_grouped_test])
    return df_train, df_test


def balancear_amostras(df_train):
    """Balanceia o dataset de treino entre classes 0 e 1."""
    target_1 = df_train[df_train['target'] == 1]
    target_0 = df_train[df_train['target'] == 0].sample(
        n=target_1.shape[0], random_state=52
    )
    return pd.concat([target_0, target_1])


# ============================================================
# TREINAMENTO COM K-FOLD (GENÉRICO PARA RF / XGB / LGBM)
# ============================================================

def treinar_kfold(X, y, model_type="rf", n_splits=5):
    """Executa o K-Fold com o modelo escolhido e retorna métricas e modelo final."""

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2s, maes, rmses = [], [], []

    # Função auxiliar para criar modelo
    def criar_modelo():
        if model_type == "lgbm":
            return lgb.LGBMRegressor(
                objective="regression_l1",
                metric="mae",
                n_estimators=2000,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.3,
                reg_lambda=0.5,
                random_state=42,
                verbosity=-1
            )
        elif model_type == "xgb":
            ratio = 1.5
            return xgb.XGBRegressor(
                objective="binary:logistic",
                scale_pos_weight=ratio,
                eval_metric="auc",
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.5,
                colsample_bylevel=0.6,
                reg_alpha=0.3,
                reg_lambda=0.5,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        elif model_type == "rf":
            return RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=42
            )
        else:
            raise ValueError(f"Modelo desconhecido: {model_type}")

    # K-Fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = criar_modelo()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        r2s.append(r2_score(y_val, y_pred))
        maes.append(mean_absolute_error(y_val, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_val, y_pred)))
    feature_importance = calcular_importancia(model, X_train)
        

    # Treina modelo final com todos os dados
    final_model = criar_modelo()
    final_model.fit(X, y)

    return {
        "r2_mean": np.mean(r2s),
        "mae_mean": np.mean(maes),
        "rmse_mean": np.mean(rmses),
        "model": final_model
    }


# ============================================================
#  IMPORTÂNCIA DAS FEATURES
# ============================================================

def calcular_importancia(model, X):
    """Calcula e imprime a importância das features."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        print("\n Importância das Features (Top 15)")
        print(feature_importance_df.head(15))
        return feature_importance_df
    else:
        print(" Modelo não fornece feature_importances_ diretamente.")
        return None


# ============================================================
#  AVALIAÇÃO NA SAFRA DE TESTE
# ============================================================

def avaliar_safra(model, df_test, include_temporais):
    # 1. Preparar os dados de uma só vez
    cols_to_drop = ['ocorrencia_id', 'data', 'data_ocorrencia', 'target', 'safra']
    if not include_temporais:
        cols_to_drop += ['dias_desde_plantio', 'estadio_fenologico']
    
    X_test_all = df_test.drop(columns=cols_to_drop)
    y_test_all = df_test['target']
    
    # 2. Predição em Lote (MUITO mais rápido)
    # Ao invés de chamar predict 4000 vezes, chamamos 1 vez.
    y_pred_all = model.predict(X_test_all)
    
    # 3. Adicionar predições ao dataframe para processamento agrupado
    df_results = df_test[['ocorrencia_id', 'target']].copy()
    df_results['pred'] = y_pred_all
    
    erros,vp, fp, fn, vn = [],0, 0, 0, 0
    
    # Agrupamos agora apenas para calcular métricas específicas ou lógica de negócio
    # Como o Pandas é otimizado, isso é rápido.
    for ocorrencia_id, group in df_results.groupby('ocorrencia_id'):
        y_t = group['target']
        y_p = group['pred']
        
        # Lógica do erro (Dias)
        # np.where retorna índices, pegamos o relativo ao grupo
        indices_acima_threshold = np.where(y_p.values >= 0.60)[0]
        if len(indices_acima_threshold) > 0:
            last_index = indices_acima_threshold[-1]
            erros.append(last_index)
        
        # Matriz de confusão (Vetorizada dentro do grupo)
        pred_label = (y_p >= 0.66).astype(int)
        true_label = y_t.astype(int)
        
        # Isso substitui o loop linha a linha
        vp += ((pred_label == 1) & (true_label == 1)).sum()
        fp += ((pred_label == 1) & (true_label == 0)).sum()
        fn += ((pred_label == 0) & (true_label == 1)).sum()
        vn += ((pred_label == 0) & (true_label == 0)).sum()

    # Cálculo final das métricas globais (mais robusto que média das médias)
    r2 = r2_score(y_test_all, y_pred_all)
    mae = mean_absolute_error(y_test_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_test_all, y_pred_all))
    
    recall = vp / (vp + fn) if (vp + fn) > 0 else 0
    precision = vp / (vp + fp) if (vp + fp) > 0 else 0

    return {
        "erro_dias": np.mean(erros) if erros else np.nan,
        "r2_test": r2,
        "mae_test": mae,
        "rmse_test": rmse,
        "vp": vp, "fp": fp, "fn": fn, "vn": vn,
        "recall": recall,
        "precision": precision
    }

# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

import os

# Certifique-se de ter importado joblib e os no início
import joblib
import os

def run(execution_started_at: datetime, cfg: Config,
        safras: list = None,
        model_type: str = "xgb",
        include_temporais: bool = True,
        nome_do_modelo: str = "XGB_classificador_temp",
        output_path: str = "resultados.csv"):

    df = carregar_dados()
    
    # Cria uma pasta para organizar esses modelos, se não existir
    pasta_modelos = "modelos_por_safra"
    if not os.path.exists(pasta_modelos):
        os.makedirs(pasta_modelos)

    if safras is None:
        safras = sorted(df['safra'].unique())

    resultados = []

    for safra_teste in safras:
        print(f"\n================ SAFRA {safra_teste} COMO TESTE ================")
        
        # Separa treino (todas as outras) e teste (safra atual)
        df_train, df_test = dividir_por_safra(df, safra_teste)
        
        if df_train is None:
            continue

        df_train = balancear_amostras(df_train)

        # Escolha de features
        drop_cols = ['ocorrencia_id','data','data_ocorrencia','target','safra']
        if not include_temporais:
            drop_cols += ['dias_desde_plantio','estadio_fenologico']

        X = df_train.drop(columns=drop_cols)
        y = df_train['target']

        # O modelo retornado aqui ('model') já foi treinado com TODO o df_train (fit final)
        metrics_kfold = treinar_kfold(X, y, model_type=model_type)

        # ============================================================
        #  SALVANDO O MODELO DESTA SAFRA
        # ============================================================
        # Ex: Se safra_teste é 2022, este modelo foi treinado com 2004-2021
        # Salvamos como "modelo_2022.pkl" para indicar "modelo usado para prever 2022"
        
        nome_arquivo = f"{nome_do_modelo}_{safra_teste}.pkl"
        caminho_completo = os.path.join(pasta_modelos, nome_arquivo)
        
        joblib.dump(metrics_kfold['model'], caminho_completo)
        print(f" Modelo salvo: {caminho_completo}")
        # ============================================================

        print("\n **Resultados Médios do K-Fold (treino interno)**")
        print(f"R2 Médio: {metrics_kfold['r2_mean']:.4f}")
        # ... (restante dos prints)

        metrics_test = avaliar_safra(metrics_kfold["model"], df_test, include_temporais)

        print(f"\n Avaliação final na Safra {safra_teste}")
        for k, v in metrics_test.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

        resultados.append({
            "ano": safra_teste,
            "nome_do_modelo": nome_do_modelo,
            "caminho_modelo": caminho_completo, # Opcional: Salvar no CSV onde está o modelo
            "Erro_dias": metrics_test["erro_dias"],
            "R2_medio": metrics_kfold["r2_mean"],
            "MAE_medio": metrics_kfold["mae_mean"],
            "RMSE_medio": metrics_kfold["rmse_mean"],
            "VP": metrics_test["vp"],
            "FP": metrics_test["fp"],
            "FN": metrics_test["fn"],
            "VN": metrics_test["vn"],
            "Recall": metrics_test["recall"],
            "Precision": metrics_test["precision"]
        })

    # Converte resultados em DataFrame e salva CSV
    df_resultados = pd.DataFrame(resultados)
    
    # ... (Seu código de cálculo de F1 e salvamento do CSV continua igual) ...
    # Apenas certifique-se de copiar o bloco final de salvar o CSV do seu código original
    
    if os.path.exists(output_path):
        df_resultados.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df_resultados.to_csv(output_path, index=False)

    return df_resultados