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
            "data_simulacao": row['data_simulacao_str'],
            "data_real_confirmacao": row['data_ocorrencia'].strftime('%Y-%m-%d'),
            "status_monitoramento": row['status_exibicao'],
            "probabilidade_atual": float(row['predito']) if pd.notnull(row['predito']) else None,
            "previsao_chegada_data": data_prev_str,
            "dias_ate_chegada": int(row['dias_ate_chegada']) if pd.notnull(row.get('dias_ate_chegada')) else None,
            "erro_final_dias": int(row['erro_final']) if pd.notnull(row.get('erro_final')) else None
        }
    }

# ============================================================
#  SCRIPT PRINCIPAL
# ============================================================

def run(execution_started_at: datetime, cfg: Config, target_safras: list = [2018]):
    
    DATASET_PATH = get_latest_file("features", "features_SI.csv")
    PASTA_MODELOS = "modelos_por_safra"
    NOME_BASE_MODELO = "XGB_classificador_temp"
    LIMIAR = 0.66
    
    OUTPUT_FOLDER = output_path(execution_started_at, 'geojson_evolution_metrics_fixed')
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    
    STEP_DAYS = 15
    if not target_safras: target_safras = [datetime.now().year]

    print("📂 A carregar dataset...")
    df = pd.read_csv(DATASET_PATH)
    df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'], format='%Y-%m-%d')
    df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')

    for safra_alvo in target_safras:
        ano_modelo = safra_alvo - 1
        data_inicio_safra = pd.to_datetime(f"{ano_modelo}-09-01")
        data_fim_safra = pd.to_datetime(f"{safra_alvo}-04-01")
        
        print(f"\n🚜 SAFRA {safra_alvo}")
        
        path_modelo = os.path.join(PASTA_MODELOS, f"{NOME_BASE_MODELO}_{ano_modelo}.pkl")
        if not os.path.exists(path_modelo):
            print(f"   ❌ Modelo não encontrado: {path_modelo}")
            continue
        model = joblib.load(path_modelo)

        df_safra = df[
            (df['data_ocorrencia'] >= data_inicio_safra) & 
            (df['data_ocorrencia'] <= data_fim_safra)
        ].copy()
        if df_safra.empty: continue

        cols_drop = ['ocorrencia_id', 'data', 'data_ocorrencia', 'target', 'safra']
        cols_existentes = [c for c in cols_drop if c in df_safra.columns]
        X_full = df_safra.drop(columns=cols_existentes)
        if hasattr(model, "feature_names_in_"): X_full = X_full[model.feature_names_in_]

        df_safra['predito'] = np.clip(model.predict(X_full), 0, 1)
        df_safra['risco_detectado'] = df_safra['predito'] >= LIMIAR

        registros_infectados = {} 
        data_atual = data_inicio_safra
        
        while data_atual <= data_fim_safra:
            
            df_dia = df_safra[df_safra['data'] <= data_atual].copy()
            if df_dia.empty:
                data_atual += timedelta(days=STEP_DAYS)
                continue
            
            df_dia = df_dia.sort_values('data').drop_duplicates(subset='ocorrencia_id', keep='last')

            l_prev, l_dias, l_erro, l_status = [], [], [], []

            for _, row in df_dia.iterrows():
                oid = row['ocorrencia_id']
                dt_real = row['data_ocorrencia']
                
                # --- MEMÓRIA ---
                if dt_real <= data_atual:
                    if oid not in registros_infectados:
                        # Pega o primeiro aviso da história
                        alerta_passado = df_safra[
                            (df_safra['ocorrencia_id'] == oid) & 
                            (df_safra['risco_detectado'] == True)
                        ]['data'].min()
                        
                        if pd.notnull(alerta_passado):
                            erro = (alerta_passado - dt_real).days
                            status = "Infectado (Antecipado)" if erro < 0 else "Infectado (Atrasado)"
                            registros_infectados[oid] = {'prev': alerta_passado, 'erro': erro, 'status': status}
                        else:
                            registros_infectados[oid] = {'prev': None, 'erro': None, 'status': "Infectado (Sem Aviso)"}
                    
                    mem = registros_infectados[oid]
                    l_prev.append(mem['prev'])
                    l_erro.append(mem['erro'])
                    l_status.append(mem['status'])
                    l_dias.append(None)
                else:
                    futuro = df_safra[
                        (df_safra['ocorrencia_id'] == oid) &
                        (df_safra['data'] > data_atual) & 
                        (df_safra['risco_detectado'] == True)
                    ]['data'].min()
                    
                    if pd.notnull(futuro):
                        dias = (futuro - data_atual).days
                        l_prev.append(futuro)
                        l_dias.append(dias)
                        l_status.append(f"Risco em {dias} dias")
                    else:
                        l_prev.append(None)
                        l_dias.append(None)
                        l_status.append("Monitorando")
                    l_erro.append(None)

            df_dia['data_prevista_exibicao'] = l_prev
            df_dia['dias_ate_chegada'] = l_dias
            df_dia['erro_final'] = l_erro
            df_dia['status_exibicao'] = l_status
            df_dia['data_simulacao_str'] = data_atual.strftime('%Y-%m-%d')

            # ============================================================
            # 📊 MÉTRICAS CORRIGIDAS (Aqui está a correção do Print)
            # ============================================================
            y_true = df_dia['target'].astype(int)
            y_pred_bin = (df_dia['predito'] >= LIMIAR).astype(int)

            VP = ((y_true == 1) & (y_pred_bin == 1)).sum()
            VN = ((y_true == 0) & (y_pred_bin == 0)).sum()
            FP = ((y_true == 0) & (y_pred_bin == 1)).sum()
            FN = ((y_true == 1) & (y_pred_bin == 0)).sum()
            
            # Totais Matemáticos
            total_pontos = len(df_dia)
            total_reais_doentes = VP + FN  # A verdade absoluta do snapshot
            
            acuracia = (VP + VN) / total_pontos * 100 if total_pontos > 0 else 0
            
            # Erro Médio (apenas dos que já têm erro fechado)
            erros_validos = [e for e in l_erro if e is not None]
            erro_medio = np.mean(erros_validos) if erros_validos else 0

            # Salva
            features = df_dia.apply(gerar_feature_geojson, axis=1).tolist()
            
            geojson_data = {
                "type": "FeatureCollection",
                "metadata": {
                    "data_referencia": data_atual.strftime('%Y-%m-%d'),
                    "safra": str(safra_alvo),
                    "total_pontos": total_pontos,
                    "infectados_snapshot": int(total_reais_doentes),
                    "acuracia": round(acuracia, 1),
                    "erro_medio_final": round(erro_medio, 1)
                },
                "features": features
            }

            filename = f"mapa_{safra_alvo}_{data_atual.strftime('%Y-%m-%d')}.geojson"
            with open(os.path.join(OUTPUT_FOLDER, filename), 'w', encoding='utf-8') as f:
                json.dump(geojson_data, f, ensure_ascii=False, indent=2)
            
            # ============================================================
            # 🖨️ PRINT COM MATEMÁTICA CORRETA
            # ============================================================
            # Agora "Infectados (Target=1)" é a soma explicita de VP + FN. Não tem como dar errado.
            print(f"   📅 {data_atual.strftime('%d/%m')} | Acc: {acuracia:.1f}% | Erro: {erro_medio:.1f}d")
            print(f"      VP: {VP} (Pegou) | FN: {FN} (Perdeu) | Infectados (Target=1): {total_reais_doentes}")
            print("-" * 40)

            data_atual += timedelta(days=STEP_DAYS)

    print("\n🏁 Finalizado!")