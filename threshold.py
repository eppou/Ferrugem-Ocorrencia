import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import confusion_matrix
from config import Config
from helpers.input_output import get_latest_file

def encontrar_melhor_threshold_safra(y_true, y_proba, beta=1.5):
    """
    Testa thresholds e retorna o melhor baseado no F-Beta Score.
    :param beta: 1.0 = F1 (Equilíbrio total)
                 1.5 = Prioriza um pouco mais o Recall (Sua escolha)
                 2.0 = Prioriza muito o Recall (Critério agrícola pesado)
    """
    thresholds = np.arange(0.01, 0.95, 0.01)
    melhor_score = -1
    dados_vencedor = {}

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, vp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        total_doenca = vp + fn
        
        # Evita divisão por zero
        if total_doenca == 0: continue 

        # 1. Calcula Métricas Básicas
        recall = vp / total_doenca
        precision = vp / (vp + fp) if (vp + fp) > 0 else 0
        
        # 2. Calcula F-Beta (A Mágica acontece aqui)
        # Fórmula: (1 + beta²) * (P * R) / ((beta² * P) + R)
        if (precision + recall) == 0:
            f_beta = 0
        else:
            numerador = (1 + beta**2) * (precision * recall)
            denominador = ((beta**2 * precision) + recall)
            f_beta = numerador / denominador
            
        # 3. Escolhe o maior F-Beta
        if f_beta > melhor_score:
            melhor_score = f_beta
            dados_vencedor = {
                'threshold': t,
                'score_fbeta': f_beta,
                'recall': recall,
                'precision': precision,
                'vp': vp, 'fn': fn, 'fp': fp
            }
            
    return dados_vencedor

def run_analise_global():
    # --- CONFIG ---
    DATASET_PATH = get_latest_file("features", "features_SI.csv")
    PASTA_MODELOS = "modelos_por_safra"
    NOME_BASE_MODELO = "XGB_classificador_temp"
    
    print(" Carregando dataset completo...")
    df = pd.read_csv(DATASET_PATH)
    
    df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'])

    # CRIAR COLUNA SAFRA
    df['safra'] = np.where(
        df['data_ocorrencia'].dt.month >= 9,
        df['data_ocorrencia'].dt.year,
        df['data_ocorrencia'].dt.year - 1
    )
    
    safras = sorted(df['safra'].unique())
    resultados_gerais = []

    print(f"\n Iniciando Análise (Critério: F1.5 - Leve prioridade ao Recall)...")

    for safra in safras:
        path_modelo = os.path.join(PASTA_MODELOS, f"{NOME_BASE_MODELO}_{safra}.pkl")
        
        if not os.path.exists(path_modelo):
            print(f" Pulei Safra {safra}: Modelo não encontrado.")
            continue

        df_safra = df[df['safra'] == safra].copy()
        if df_safra.empty: continue

        # Prepara X (Sem remover lat/long)
        cols_ignore = ['ocorrencia_id', 'data', 'data_ocorrencia', 'target', 'safra']
        cols_existentes = [c for c in cols_ignore if c in df_safra.columns]
        X = df_safra.drop(columns=cols_existentes)
        y = df_safra['target']

        try:
            model = joblib.load(path_modelo)
            
            if hasattr(model, "feature_names_in_"):
                X = X[model.feature_names_in_]
            
            raw_pred = model.predict(X)
            y_proba = np.clip(raw_pred, 0, 1) 
            
        except Exception as e:
            print(f" Erro na safra {safra}: {e}")
            continue

        # BUSCA O MELHOR THRESHOLD COM BETA 1.5
        res = encontrar_melhor_threshold_safra(y, y_proba, beta=1.5)
        
        if not res:
            print(f" Safra {safra}: Dados insuficientes.")
            continue

        print(f" Safra {safra}: Threshold Ideal = {res['threshold']:.2f} "
              f"(F1.5: {res['score_fbeta']:.2f} | Recall: {res['recall']:.1%} | Precision: {res['precision']:.1%})")

        resultados_gerais.append({
            'Safra': safra,
            'Ideal_Threshold': res['threshold'],
            'F1.5_Score': res['score_fbeta'],
            'Recall': res['recall'],
            'Precision': res['precision']
        })

    # --- RELATÓRIO FINAL ---
    df_res = pd.DataFrame(resultados_gerais)
    
    print("\n" + "="*50)
    print(" RESULTADO CONSOLIDADO (Critério F1.5)")
    print("="*50)
    
    if not df_res.empty:
        media_t = df_res['Ideal_Threshold'].mean()
        mediana_t = df_res['Ideal_Threshold'].median()
        min_t = df_res['Ideal_Threshold'].min()
        std_t = df_res['Ideal_Threshold'].std()

        print(df_res[['Safra', 'Ideal_Threshold', 'Recall', 'Precision', 'F1.5_Score']].to_string(index=False))
        print("-" * 50)
        print(f" MÉDIA DO THRESHOLD:   {media_t:.4f}")
        print(f" MEDIANA:              {mediana_t:.4f}")
        print(f" Desvio Padrão:        {std_t:.4f}")
        
        print("\n DICA:")
        print("Use a MÉDIA se o desvio for baixo (< 0.1).")
        print("Use a MEDIANA se houver outliers (um ano muito diferente dos outros).")
    else:
        print("Nenhum resultado gerado.")

if __name__ == "__main__":
    run_analise_global()