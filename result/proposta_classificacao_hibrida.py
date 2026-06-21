import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from config import Config
from helpers.input_output import get_latest_file, output_path

def classificar_periodo(data):
    mes = data.month
    if mes in [9, 10, 11]:
        return '1_Começo (Set-Nov)'
    elif mes in [12, 1]:
        return '2_Meio (Dez-Jan)'
    return '3_Fim (Fev-Abr)'

def calcular_metricas(y_true, y_pred, erros_dias=None, prefixo=""):
    labels = [0, 1] 
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    
    resultados = {
        f'{prefixo}Precisao': precision_score(y_true, y_pred, zero_division=0),
        f'{prefixo}Recall': recall_score(y_true, y_pred, zero_division=0),
        f'{prefixo}F1_Score': f1_score(y_true, y_pred, zero_division=0),
        f'{prefixo}TP': tp,
        f'{prefixo}FP': fp,
        f'{prefixo}FN': fn,
        f'{prefixo}TN': tn
    }
    
    if erros_dias is not None:
        resultados[f'{prefixo}Erro_Medio_Dias'] = erros_dias.abs().mean() if not erros_dias.dropna().empty else 0.0
        
    return resultados

def run(execution_started_at: datetime, cfg: Config, target_safras: list = None):
    if target_safras is None:
        target_safras = []

    DATASET_PATH = get_latest_file("features", "features_SI.csv")
    PASTA_MODELOS = "modelos_por_safra"
    PASTA_REGRESSAO = "modelos_regressao"
    
    NOME_MODELO_CLASS = "XGB_classificador_temp"
    NOME_MODELO_REG = "XGB_regressor_safra"
    
    LIMIAR_CLASSIFICADOR = 0.66
    LIMIAR_REGRESSOR_DIAS = 13
    
    OUTPUT_FOLDER = output_path(execution_started_at, 'benchmarks')
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

    df = pd.read_csv(DATASET_PATH)
    df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')
    df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'], format='%Y-%m-%d', errors='coerce')
    df['target'] = df['target'].astype(int) 

    if not target_safras:
        target_safras = sorted(df['data'].dt.year.unique())

    resultados_gerais = []
    resultados_periodos = []

    for safra_alvo in target_safras:
        ano_modelo = safra_alvo - 1
        safra_start = pd.to_datetime(f"{ano_modelo}-09-01")
        data_fim_safra = pd.to_datetime(f"{safra_alvo}-04-01")
        
        path_class = os.path.join(PASTA_MODELOS, f"{NOME_MODELO_CLASS}_{(ano_modelo)}.pkl")
        path_reg = os.path.join(PASTA_REGRESSAO, f"{NOME_MODELO_REG}_{(safra_alvo)}.pkl")
        
        if not os.path.exists(path_class) or not os.path.exists(path_reg):
            continue

        model_class = joblib.load(path_class)
        model_reg = joblib.load(path_reg)

        mask_safra = (df['data'] >= safra_start) & (df['data'] <= data_fim_safra)
        df_safra = df[mask_safra].copy()
        
        if df_safra.empty:
            continue

        df_safra['dia_plantio'] = ((df_safra['data'] - pd.to_timedelta(df_safra['dias_desde_plantio'], unit='D')) - safra_start).dt.days
        df_safra['periodo_safra'] = df_safra['data'].apply(classificar_periodo)

        cols_drop = ['ocorrencia_id', 'data', 'data_ocorrencia', 'target', 'safra']
        X = df_safra.drop(columns=[c for c in cols_drop if c in df_safra.columns])
        y_true = df_safra['target'].values

        if hasattr(model_class, "feature_names_in_"): X_class = X[model_class.feature_names_in_]
        if hasattr(model_reg, "feature_names_in_"): X_reg = X[model_reg.feature_names_in_]

        prob_class = model_class.predict(X_class)
        y_pred_class_only = (prob_class >= LIMIAR_CLASSIFICADOR).astype(int)

        preds_dias_corridos = model_reg.predict(X_reg)
        data_prev_reg = safra_start + pd.to_timedelta(preds_dias_corridos, unit='D')
        delta_dias_regressor = (data_prev_reg - df_safra['data']).dt.days

        veto_ativo = delta_dias_regressor > LIMIAR_REGRESSOR_DIAS
        y_pred_hibrido = y_pred_class_only.copy()
        y_pred_hibrido[veto_ativo] = 0 
        
        # Erro do Híbrido
        df_safra['data_predicao_final'] = np.where(
            y_pred_hibrido == 1, 
            df_safra['data'], 
            data_prev_reg     
        )
        df_safra['erro_dias'] = (df_safra['data_predicao_final'] - df_safra['data_ocorrencia']).dt.days

        df_safra['y_true'] = y_true
        df_safra['y_pred_class'] = y_pred_class_only
        df_safra['y_pred_hibrido'] = y_pred_hibrido

        # Erro do Classificador Only
        dias_com_alerta_class = df_safra[df_safra['y_pred_class'] == 1]
        primeiro_sim_por_ocorrencia = dias_com_alerta_class.groupby('ocorrencia_id')['data'].min()
        df_safra['data_pred_class_only'] = df_safra['ocorrencia_id'].map(primeiro_sim_por_ocorrencia)
        df_safra['erro_dias_class'] = (df_safra['data_pred_class_only'] - df_safra['data_ocorrencia']).dt.days

        #importance do classificador
        if hasattr(model_class, "feature_importances_"):
            importancias = model_class.feature_importances_
            features = X_class.columns
            df_importancia = pd.DataFrame({'Feature': features, 'Importance': importancias})
            df_importancia.sort_values(by='Importance', ascending=False, inplace=True)
            df_importancia.to_csv(os.path.join(OUTPUT_FOLDER, f"feature_importance_{safra_alvo}.csv"), index=False)
            
        erros_infectados_hibrido = df_safra.loc[df_safra['target'] == 1, 'erro_dias']
        erros_infectados_class = df_safra.loc[df_safra['target'] == 1, 'erro_dias_class']
        
        metr_class = calcular_metricas(y_true, y_pred_class_only, erros_dias=erros_infectados_class, prefixo="ClassOnly_")
        metr_hibrido = calcular_metricas(y_true, y_pred_hibrido, erros_dias=erros_infectados_hibrido, prefixo="Hibrido_")
        
        linha_geral = {'Safra': safra_alvo, 'Total_Amostras': len(df_safra)}
        linha_geral.update(metr_class)
        linha_geral.update(metr_hibrido)
        resultados_gerais.append(linha_geral)

        for periodo, df_periodo in df_safra.groupby('periodo_safra'):
            y_t = df_periodo['y_true']
            y_p_c = df_periodo['y_pred_class']
            y_p_h = df_periodo['y_pred_hibrido']
            
            erros_inf_per_hibrido = df_periodo.loc[df_periodo['target'] == 1, 'erro_dias']
            erros_inf_per_class = df_periodo.loc[df_periodo['target'] == 1, 'erro_dias_class']
            
            metr_c_per = calcular_metricas(y_t, y_p_c, erros_dias=erros_inf_per_class, prefixo="ClassOnly_")
            metr_h_per = calcular_metricas(y_t, y_p_h, erros_dias=erros_inf_per_hibrido, prefixo="Hibrido_")
            
            linha_periodo = {
                'Safra': safra_alvo, 
                'Periodo': periodo, 
                'Total_Amostras': len(df_periodo)
            }
            linha_periodo.update(metr_c_per)
            linha_periodo.update(metr_h_per)
            resultados_periodos.append(linha_periodo)

    if resultados_gerais:
        df_geral = pd.DataFrame(resultados_gerais)
        df_periodos = pd.DataFrame(resultados_periodos)
        
        path_geral = os.path.join(OUTPUT_FOLDER, "benchmark_safra_geral.csv")
        path_periodos = os.path.join(OUTPUT_FOLDER, "benchmark_por_periodo.csv")
        
        df_geral.to_csv(path_geral, index=False, sep=';', decimal=',')
        df_periodos.to_csv(path_periodos, index=False, sep=';', decimal=',')