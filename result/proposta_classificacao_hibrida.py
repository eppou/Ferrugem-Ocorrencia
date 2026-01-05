from sqlalchemy import create_engine
from config import Config
from helpers.feature_importance import calculate_importance_avg, calculate_k_best, calculate_percentile
from helpers.input_output import get_latest_file, output_file
from helpers.result import write_result, read_result

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from datetime import datetime
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# 0️⃣ DEFINIÇÃO DA CLASSE DO ENSEMBLE
# ============================================================

class EnsemblePredictor:
    """
    Classe que encapsula os dois modelos (Tempo e Clima) e o Meta-Modelo.
    Isso permite usar .predict() como se fosse um modelo único.
    """
    def __init__(self, model_clima, model_tempo, meta_model, cols_clima, cols_tempo):
        self.model_clima = model_clima
        self.model_tempo = model_tempo
        self.meta_model = meta_model
        self.cols_clima = cols_clima
        self.cols_tempo = cols_tempo

    def fit(self, X, y):
        # Nota: O fit real acontece na função treinar_ensemble_cv para garantir OOF
        # Este método existe para compatibilidade se necessário
        pass

    def predict(self, X):
        # 1. Gera predições do especialista em Clima
        pred_clima = self.model_clima.predict(X[self.cols_clima])
        
        # 2. Gera predições do especialista em Tempo
        pred_tempo = self.model_tempo.predict(X[self.cols_tempo])
        
        # 3. Empilha as predições
        X_stack = np.column_stack((pred_clima, pred_tempo))
        
        # 4. O Meta-Modelo decide o resultado final
        return self.meta_model.predict(X_stack)
    
    @property
    def feature_importances_(self):
        # Retorna uma média ponderada ou apenas do modelo climático para análise
        return self.model_clima.feature_importances_

# ============================================================
# 1️⃣ CARREGAMENTO E PREPARAÇÃO DOS DADOS
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

# ============================================================
# 2️⃣ DIVISÃO POR SAFRA E BALANCEAMENTO
# ============================================================

def dividir_por_safra(df, safra_teste):
    """Divide o dataset em treino e teste com base na safra."""
    df_grouped = df.groupby('ocorrencia_id')
    df_grouped_test = [g for g in df_grouped if g[1]['safra'].iloc[0] == safra_teste]
    df_grouped_train = [g for g in df_grouped if g[1]['safra'].iloc[0] != safra_teste]

    if len(df_grouped_test) == 0:
        print(f"⚠️ Safra {safra_teste} sem dados para teste.")
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
# 3️⃣ DEFINIÇÃO DOS MODELOS BASE
# ============================================================

def get_base_model(model_type):
    """Retorna uma instância nova do modelo base."""
    if model_type == "xgb":
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=800, # Um pouco menos pois teremos 2 modelos
            learning_rate=0.05,
            max_depth=5, # Profundidade controlada
            subsample=0.8,
            colsample_bytree=0.6, 
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    elif model_type == "lgbm":
        return lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
    else:
        # Default fallback
        return RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)

# ============================================================
# 4️⃣ TREINAMENTO DO ENSEMBLE (STACKING)
# ============================================================

def treinar_ensemble_cv(X, y, model_type="xgb", n_splits=5):
    """
    Treina o Clima Model e o Tempo Model usando Cross-Validation para gerar 
    predições Out-of-Fold (OOF) para treinar o Meta-Modelo.
    """
    
    # 1. Definição das Features para cada modelo
    # Globais que vão em ambos
    cols_globais = ['ocorrencia_latitude', 'ocorrencia_longitude', 'enso']
    
    # Exclusivas de tempo
    cols_tempo_exclusivas = ['dias_desde_plantio', 'estadio_fenologico']
    
    # Monta listas finais
    cols_tempo = cols_tempo_exclusivas + [c for c in cols_globais if c in X.columns]
    
    # Clima pega TUDO que não for exclusivas de tempo
    cols_clima = [c for c in X.columns if c not in cols_tempo_exclusivas]

    # Arrays para guardar as predições OOF (Out of Fold)
    oof_preds_clima = np.zeros(len(X))
    oof_preds_tempo = np.zeros(len(X))
    
    # Métricas CV
    r2s, maes, rmses = [], [], []
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"🔄 Iniciando Stacking CV ({n_splits} folds)...")
    
    # --- LOOP DE CROSS-VALIDATION ---
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # A) Treina Especialista em Clima
        model_clima = get_base_model(model_type)
        model_clima.fit(X_train[cols_clima], y_train)
        pred_clima_val = model_clima.predict(X_val[cols_clima])
        oof_preds_clima[val_idx] = pred_clima_val
        
        # B) Treina Especialista em Tempo
        model_tempo = get_base_model(model_type)
        model_tempo.fit(X_train[cols_tempo], y_train)
        pred_tempo_val = model_tempo.predict(X_val[cols_tempo])
        oof_preds_tempo[val_idx] = pred_tempo_val

    # --- TREINAMENTO DO META-MODELO ---
    # O input do meta modelo são as predições dos dois modelos
    X_stack_train = np.column_stack((oof_preds_clima, oof_preds_tempo))
    
    # Usamos Ridge (Regressão Linear com regularização) para encontrar os pesos ótimos
    # Ele vai aprender algo como: Final = 0.3 * Clima + 0.7 * Tempo
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_stack_train, y)
    
    print(f"⚖️  Pesos do Ensemble -> Clima: {meta_model.coef_[0]:.4f} | Tempo: {meta_model.coef_[1]:.4f}")

    # --- RETREINAMENTO FINAL (Full Dataset) ---
    # Agora treinamos os modelos base com TODOS os dados para usar no teste
    final_model_clima = get_base_model(model_type)
    final_model_clima.fit(X[cols_clima], y)
    
    final_model_tempo = get_base_model(model_type)
    final_model_tempo.fit(X[cols_tempo], y)
    
    # Cria objeto wrapper para retornar
    ensemble = EnsemblePredictor(
        final_model_clima, 
        final_model_tempo, 
        meta_model, 
        cols_clima, 
        cols_tempo
    )
    
    # Calcula métricas internas do CV usando as previsões OOF empilhadas
    final_oof_preds = meta_model.predict(X_stack_train)
    
    return {
        "r2_mean": r2_score(y, final_oof_preds),
        "mae_mean": mean_absolute_error(y, final_oof_preds),
        "rmse_mean": np.sqrt(mean_squared_error(y, final_oof_preds)),
        "model": ensemble
    }

# ============================================================
# 5️⃣ AVALIAÇÃO (COMPATÍVEL)
# ============================================================

def avaliar_safra(model, df_test):
    # Nota: removemos include_temporais porque o ensemble gerencia isso internamente
    
    cols_to_drop = ['ocorrencia_id', 'data', 'data_ocorrencia', 'target', 'safra']
    
    X_test_all = df_test.drop(columns=cols_to_drop)
    y_test_all = df_test['target']
    
    # O método .predict do nosso EnsemblePredictor cuida de separar as colunas
    y_pred_all = model.predict(X_test_all)
    
    # --- PÓS-PROCESSAMENTO (Média Móvel) ---
    # Adicionei isso baseado na nossa conversa anterior para melhorar o erro_dias
    # y_pred_all = pd.Series(y_pred_all).rolling(window=3, min_periods=1).mean().values
    
    df_results = df_test[['ocorrencia_id', 'target']].copy()
    df_results['pred'] = y_pred_all
    
    erros, vp, fp, fn, vn = [], 0, 0, 0, 0
    
    for ocorrencia_id, group in df_results.groupby('ocorrencia_id'):
        y_t = group['target']
        y_p = group['pred']
        
        # Threshold
        indices_acima_threshold = np.where(y_p.values >= 0.60)[0]
        if len(indices_acima_threshold) > 0:
            last_index = indices_acima_threshold[-1] 
            erros.append(last_index)
        
        # Matriz confusão
        pred_label = (y_p >= 0.60).astype(int)
        true_label = y_t.astype(int)
        
        vp += ((pred_label == 1) & (true_label == 1)).sum()
        fp += ((pred_label == 1) & (true_label == 0)).sum()
        fn += ((pred_label == 0) & (true_label == 1)).sum()
        vn += ((pred_label == 0) & (true_label == 0)).sum()

    # Métricas
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
# 6️⃣ PIPELINE PRINCIPAL
# ============================================================

def run(execution_started_at: datetime, cfg: Config,
        safras: list = None,
        model_type: str = "xgb",
        nome_do_modelo: str = "Ensemble_Tempo_Clima",
        output_path: str = "resultados_ensemble.csv"):

    df = carregar_dados()
    if safras is None:
        safras = sorted(df['safra'].unique())

    resultados = []

    for safra_teste in safras:
        print(f"\n================ SAFRA {safra_teste} COMO TESTE ================")
        df_train, df_test = dividir_por_safra(df, safra_teste)
        if df_train is None:
            continue

        df_train = balancear_amostras(df_train)

        # Prepara X e y
        drop_cols = ['ocorrencia_id','data','data_ocorrencia','target','safra']
        X = df_train.drop(columns=drop_cols)
        y = df_train['target']

        # TREINA O ENSEMBLE
        metrics_cv = treinar_ensemble_cv(X, y, model_type=model_type)

        print("\n📊 **Resultados do Ensemble (CV Interno)**")
        print(f"R2 Combinado: {metrics_cv['r2_mean']:.4f}")
        print(f"MAE Combinado: {metrics_cv['mae_mean']:.4f}")

        # AVALIA NA SAFRA DE TESTE
        metrics_test = avaliar_safra(metrics_cv["model"], df_test)

        print(f"\n📌 Avaliação final na Safra {safra_teste}")
        for k, v in metrics_test.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

        resultados.append({
            "ano": safra_teste,
            "nome_do_modelo": nome_do_modelo,
            "Erro_dias": metrics_test["erro_dias"],
            "R2_medio": metrics_cv["r2_mean"],
            "MAE_medio": metrics_cv["mae_mean"],
            "RMSE_medio": metrics_cv["rmse_mean"],
            "VP": metrics_test["vp"],
            "FP": metrics_test["fp"],
            "FN": metrics_test["fn"],
            "VN": metrics_test["vn"],
            "Recall": metrics_test["recall"],
            "Precision": metrics_test["precision"]
        })

    # Salva resultados
    df_resultados = pd.DataFrame(resultados)
    df_resultados["F1"] = 2 * ((df_resultados["Recall"] * df_resultados["Precision"]) / (df_resultados["Recall"] + df_resultados["Precision"])).fillna(0)
    
    print(("Erro medio final (dias): "
           f"{df_resultados['Erro_dias'].mean():.4f} | "
           f"R2 médio final: {df_resultados['R2_medio'].mean():.4f} | "
           f"RMSE médio final: {df_resultados['RMSE_medio'].mean():.4f}"))
    if os.path.exists(output_path):
        df_resultados.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df_resultados.to_csv(output_path, index=False)
    
    print(f"\n✅ Resultados salvos em: {output_path}")

    return df_resultados