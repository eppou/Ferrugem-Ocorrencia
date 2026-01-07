import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime, timedelta
from config import Config
from helpers.input_output import get_latest_file, output_path

# ============================================================
#  FUNÇÕES AUXILIARES
# ============================================================

def gerar_feature_geojson(row):
    data_prev_str = None
    if pd.notnull(row.get('data_prevista_exibicao')):
        val = row['data_prevista_exibicao']
        data_prev_str = val.strftime('%Y-%m-%d') if isinstance(val, pd.Timestamp) else str(val)

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [float(row['ocorrencia_longitude']), float(row['ocorrencia_latitude'])]
        },
        "properties": {
            "id": int(row['ocorrencia_id']) if 'ocorrencia_id' in row else None,
            
            # --- NOVOS CAMPOS SOLICITADOS ---
            "classe_real": int(row['visual_classe_real']),      # 0 ou 1 (Realidade)
            "classe_predita": int(row['visual_classe_predita']), # 0 ou 1 (O que o modelo disse)
            # --------------------------------
            
            "data_simulacao": row['data_simulacao_str'],
            "data_real_ocorrencia": row['data_ocorrencia'].strftime('%Y-%m-%d'),
            "probabilidade_para_hoje": float(row['predito_prob']), 
            "status_risco": row['status_exibicao'],
            "previsao_chegada_ferrugem": data_prev_str,
            "dias_ate_chegada": int(row['dias_ate_chegada']) if pd.notnull(row.get('dias_ate_chegada')) else None,
            "erro_final_dias": int(row['erro_final']) if pd.notnull(row.get('erro_final')) else None
        }
    }
    
# ============================================================
#  SCRIPT PRINCIPAL
# ============================================================

def run(execution_started_at: datetime, cfg: Config, target_safras: list = [2007]):
    
    # 1. Configurações de Pastas e Nomes
    DATASET_PATH = get_latest_file("features", "features_SI.csv")
    PASTA_MODELOS = "modelos_por_safra"
    PASTA_REGRESSAO = "modelos_regressao" # Pasta onde salvou o regressor
    
    NOME_MODELO_CLASS = "XGB_classificador_temp"
    NOME_MODELO_REG = "XGB_regressor_safra" # Nome que você definiu ao salvar
    
    LIMIAR_CLASSIFICADOR = 0.5
    LIMIAR_REGRESSOR_DIAS = 13 # Dias para a trava de segurança do regressor se o classificador apitar positivo mas o regressor disser que está longe. desconsidera resultado do classificador nesses casos.
    
    OUTPUT_FOLDER = output_path(execution_started_at, 'geojson_maps')
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    
    STEP_DAYS = 15
    if not target_safras: target_safras = [datetime.now().year]

    print("📂 A carregar dataset...")
    df = pd.read_csv(DATASET_PATH)
    df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'], format='%Y-%m-%d')
    df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')

    for safra_alvo in target_safras:
        ano_modelo = safra_alvo - 1
        
        # Datas de referência da safra (Crucial para o Regressor)
        # O regressor foi treinado com target = dias a partir de 01/09
        safra_start = pd.to_datetime(f"{ano_modelo}-09-01")
        data_fim_safra = pd.to_datetime(f"{safra_alvo}-04-01")
        
        print(f"\n🚜 SAFRA {safra_alvo} (Inicio referência: {safra_start.date()})")
        
        # --- 2. CARGA DOS DOIS MODELOS ---
        path_class = os.path.join(PASTA_MODELOS, f"{NOME_MODELO_CLASS}_{ano_modelo}.pkl")
        path_reg = os.path.join(PASTA_REGRESSAO, f"{NOME_MODELO_REG}_{(safra_alvo)}.pkl") # Regressor testa na safra atual
        
        if not os.path.exists(path_class):
            print(f"   ❌ Classificador não encontrado: {path_class}")
            continue
        if not os.path.exists(path_reg):
            print(f"   ❌ Regressor não encontrado: {path_reg}. Verifique o ano/nome.")
            # Opcional: continue ou use try/except
            continue

        model_class = joblib.load(path_class)
        model_reg = joblib.load(path_reg)

        # --- 3. PREPARAÇÃO DOS DADOS ---
        df_safra = df[
            (df['data_ocorrencia'] >= safra_start) & 
            (df['data_ocorrencia'] <= data_fim_safra)
        ].copy()
        if df_safra.empty: continue

        # Engenharia de Features para o Regressor (Igual ao seu treino)
        # Cria 'dia_plantio' se não existir ou recalcula para garantir consistência
        df_safra['dia_plantio'] = ((df_safra['data'] - pd.to_timedelta(df_safra['dias_desde_plantio'], unit='D')) - safra_start).dt.days
        
        # --- 4. PREDIÇÃO EM MASSA (Com Trava de Segurança) ---
        
        cols_drop = ['ocorrencia_id', 'data', 'data_ocorrencia', 'target', 'safra']
        
        # 1. REGRESSOR (Calculamos primeiro para ter a estimativa de dias)
        X_reg = df_safra.drop(columns=[c for c in cols_drop if c in df_safra.columns])
        if hasattr(model_reg, "feature_names_in_"):
            X_reg = X_reg[model_reg.feature_names_in_]
            
        preds_dias_corridos = model_reg.predict(X_reg)
        
        # Data prevista pelo Regressor
        df_safra['data_prevista_regressao'] = safra_start + pd.to_timedelta(preds_dias_corridos, unit='D')
        
        # CÁLCULO DO DELTA (Visão do Regressor em relação ao dia da coleta)
        # "Quantos dias faltam para a doença chegar segundo o regressor, baseado na data de hoje?"
        df_safra['delta_dias_regressor'] = (df_safra['data_prevista_regressao'] - df_safra['data']).dt.days

        # 2. CLASSIFICADOR
        X_class = df_safra.drop(columns=[c for c in cols_drop if c in df_safra.columns])
        if hasattr(model_class, "feature_names_in_"):
            X_class = X_class[model_class.feature_names_in_]
        
        df_safra['predito_prob'] = np.clip(model_class.predict(X_class), 0, 1)
        
        # 3. APLICAÇÃO DA TRAVA (Veto do Regressor)
        # Regra: Se o classificador apitar, mas o regressor disser que falta mais de 10 dias...
        # ...nós zeramos a classificação (consideramos falso positivo do classificador).
        
        # Cria a máscara de quem seria "Risco" inicialmente
        mask_risco_inicial = df_safra['predito_prob'] >= LIMIAR_CLASSIFICADOR
        
        # Cria a máscara do "Veto" (Regressor diz que está longe)
        mask_longe = df_safra['delta_dias_regressor'] > LIMIAR_REGRESSOR_DIAS
        
        # Aplica o veto: Zera a probabilidade e remove o risco onde as duas condições são verdadeiras
        df_safra.loc[mask_risco_inicial & mask_longe, 'predito_prob'] = 0.0
        
        # Recalcula o booleano final após a trava
        df_safra['risco_detectado'] = df_safra['predito_prob'] >= LIMIAR_CLASSIFICADOR

        # 4. CONSOLIDAÇÃO DA DATA (O "Melhor dos Mundos" refinado)
        # Se SOBROU algum risco detectado (ou seja, Classificador apitou E Regressor concordou que está perto),
        # aí sim assumimos que a infecção é HOJE.
        
        mask_risco_validado = df_safra['risco_detectado'] == True
        df_safra.loc[mask_risco_validado, 'data_prevista_regressao'] = df_safra.loc[mask_risco_validado, 'data']
        
        # ============================================================
        # 5. LOOP TEMPORAL COM LÓGICA HÍBRIDA E MÉTICAS
        # ============================================================
        registros_infectados = {} 
        data_atual = safra_start
        
        while data_atual <= data_fim_safra:
            
            # Snapshot do dia
            df_dia = df_safra[df_safra['data'] <= data_atual].copy()
            if df_dia.empty:
                data_atual += timedelta(days=STEP_DAYS)
                continue
            
            df_dia = df_dia.sort_values('data').drop_duplicates(subset='ocorrencia_id', keep='last')

            l_prev, l_dias, l_erro, l_status,l_real, l_pred = [], [], [], [], [], []
            
            # >>> NOVAS METRICAS: Contadores do Dia <<<
            metricas_dia = {'VP': 0, 'FN': 0, 'Total_Infectados': 0}
            
            for _, row in df_dia.iterrows():
                oid = row['ocorrencia_id']
                dt_real = row['data_ocorrencia']
                
                # ===================================================
                # CENÁRIO 1: JÁ INFECTADO (PASSADO)
                # ===================================================
                if dt_real <= data_atual:
                    metricas_dia['Total_Infectados'] += 1
                    
                    # Define CLASSE REAL como 1 (Doença confirmada)
                    l_real.append(1)

                    if oid not in registros_infectados:
                        # Busca histórico se o modelo alertou antes
                        subset_alerta = df_safra[
                            (df_safra['ocorrencia_id'] == oid) & 
                            (df_safra['risco_detectado'] == True)
                        ]
                        
                        if not subset_alerta.empty:
                            primeiro_alerta = subset_alerta.sort_values('data').iloc[0]
                            data_prevista_na_epoca = primeiro_alerta['data_prevista_regressao']
                            erro = (data_prevista_na_epoca - dt_real).days
                            status = "Infectado (Antecipado)" if erro < 0 else "Infectado (Atrasado)"
                            
                            registros_infectados[oid] = {
                                'prev': data_prevista_na_epoca,
                                'erro': erro,
                                'status': f"{status} ({abs(erro)}d)",
                                'classificacao': 'VP'
                            }
                        else:
                            registros_infectados[oid] = {
                                'prev': None, 
                                'erro': None, 
                                'status': "Infectado (Sem Alerta)",
                                'classificacao': 'FN'
                            }
                    
                    mem = registros_infectados[oid]
                    l_prev.append(mem['prev'])
                    l_erro.append(mem['erro'])
                    l_status.append(mem['status'])
                    l_dias.append(None)
                    
                    if mem['classificacao'] == 'VP':
                        metricas_dia['VP'] += 1
                        l_pred.append(1) # VP: Era 1 e o modelo disse 1 (em algum momento)
                    else:
                        metricas_dia['FN'] += 1
                        l_pred.append(0) # FN: Era 1 e o modelo disse 0 (silêncio)

                # ===================================================
                # CENÁRIO 2: AINDA SAUDÁVEL (FUTURO)
                # ===================================================
                else:
                    # Define CLASSE REAL como 0 (Ainda não aconteceu)
                    l_real.append(0)
                    
                    # Define CLASSE PREDITA baseado no alerta de HOJE
                    # (Lembrando que o 'risco_detectado' já passou pela sua trava de segurança dos 10 dias)
                    pred_hoje = 1 if row['risco_detectado'] else 0
                    l_pred.append(pred_hoje)

                    dt_prevista = row['data_prevista_regressao']
                    dias_restantes = (dt_prevista - data_atual).days
                    l_prev.append(dt_prevista)
                    l_dias.append(dias_restantes)
                    l_erro.append(None)
                    
                    if dias_restantes > 0:
                        l_status.append(f"Previsto em {dias_restantes} dias")
                    elif dias_restantes == 0:
                        l_status.append("Previsto para HOJE")
                    else:
                        l_status.append(f"Atrasado ({abs(dias_restantes)} dias)")

            # Acopla ao DF antes de gerar o JSON
            df_dia['data_prevista_exibicao'] = l_prev
            df_dia['dias_ate_chegada'] = l_dias
            df_dia['erro_final'] = l_erro
            df_dia['status_exibicao'] = l_status
            
            # NOVAS COLUNAS PARA O GERADOR
            df_dia['visual_classe_real'] = l_real
            df_dia['visual_classe_predita'] = l_pred
            
            df_dia['data_simulacao_str'] = data_atual.strftime('%Y-%m-%d')
            
            features = df_dia.apply(gerar_feature_geojson, axis=1).tolist()
            
            # >>> CÁLCULO DE ESTATÍSTICAS FINAIS DO DIA <<<
            erros_validos = [e for e in l_erro if e is not None]
            erro_medio = np.mean(erros_validos) if erros_validos else 0
            
            # Recall (Sensibilidade) = VP / (VP + FN)
            total_inf = metricas_dia['Total_Infectados']
            recall = (metricas_dia['VP'] / total_inf * 100) if total_inf > 0 else 0.0

            geojson_data = {
                "type": "FeatureCollection",
                "metadata": {
                    "data_referencia": data_atual.strftime('%Y-%m-%d'),
                    "safra": str(safra_alvo),
                    "total_pontos": len(features),
                    "erro_medio_dias_regressor": round(erro_medio, 1),
                    # Novos Metadados para o Frontend
                    "metricas_classificacao": {
                        "total_infectados": total_inf,
                        "VP": metricas_dia['VP'],
                        "FN": metricas_dia['FN'],
                        "recall_percentual": round(recall, 1)
                    }
                },
                "features": features
            }

            filename = f"mapa_{safra_alvo}_{data_atual.strftime('%Y-%m-%d')}.geojson"
            with open(os.path.join(OUTPUT_FOLDER, filename), 'w', encoding='utf-8') as f:
                json.dump(geojson_data, f, ensure_ascii=False, indent=2)
            
            # >>> PRINT MAIS COMPLETO <<<
            print(f"   📅 {data_atual.strftime('%d/%m')} | "
                  f"Infecções Reais: {total_inf} | "
                  f"VP: {metricas_dia['VP']} | FN: {metricas_dia['FN']} | "
                  f"Recall: {recall:.1f}% | "
                  f"Erro Regressão: {erro_medio:.1f}d")

            data_atual += timedelta(days=STEP_DAYS)

    print("\n🏁 Finalizado!")