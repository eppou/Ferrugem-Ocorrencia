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
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [float(row['ocorrencia_longitude']), float(row['ocorrencia_latitude'])]
        },
        "properties": {
            "id": int(row['ocorrencia_id']) if 'ocorrencia_id' in row else None,
            "data": row['data_ocorrencia'].strftime('%Y-%m-%d'),
            "real": int(row['target']),
            "predito_prob": float(row['predito']), # Agora garantido entre 0 e 1
            "predito_classe": 1 if row['predito'] >= 0.77 else 0,
            "acertou": bool((row['predito'] >= 0.77) == (row['target'] == 1))
        }
    }

# ============================================================
#  SCRIPT PRINCIPAL
# ============================================================

def run(execution_started_at: datetime, cfg: Config, target_safras: list = [2017]):
    """
    Gera GeoJSONs focados na janela da safra (Setembro a Abril).
    param target_safras: Lista de anos de colheita. Ex: [2021] processa Set/20 a Abr/21.
    """
    
    # --- 1. CONFIGURAÇÕES DE INPUT---
    DATASET_PATH = get_latest_file("features", "features_SI.csv")
    PASTA_MODELOS = "modelos_por_safra"
    NOME_BASE_MODELO = "XGB_classificador_temp"
    
    OUTPUT_FOLDER = output_path(execution_started_at, 'geojson_evolution')
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    STEP_DAYS = 15 

    # Se não informar safras, define uma lista padrão ou avisa
    if not target_safras:
        print(" Nenhuma safra especificada. Usando safra atual como exemplo.")
        target_safras = [datetime.now().year]

    # --- 2. CARGA DE DADOS ---
    print(" Carregando dataset completo...")
    df = pd.read_csv(DATASET_PATH)
    df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'], format='%Y-%m-%d')
    df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')
    # Otimização: remove colunas desnecessárias para processamento, mas mantém Lat/Long
    # df = df.drop(columns=['...']) # Opcional se o DF for gigante

    # ============================================================
    #  LOOP POR SAFRA (A unidade principal agora é a SAFRA)
    # ============================================================
    for safra_alvo in target_safras:
        
        # 1. Definir o Modelo e a Janela de Tempo
        # Se safra_alvo é 2021:
        # Modelo usado: 2020 (treinado com dados até o fim da safra 2020)
        ano_modelo = safra_alvo - 1
        
        # Janela: 01/09/2020 a 01/04/2021
        data_inicio_safra = pd.to_datetime(f"{ano_modelo}-09-01")
        data_fim_safra = pd.to_datetime(f"{safra_alvo}-04-01")
        
        print(f"\n PROCESSANDO SAFRA {safra_alvo}")
        print(f" Janela: {data_inicio_safra.date()} até {data_fim_safra.date()}")
        print(f" Modelo Solicitado: {NOME_BASE_MODELO}_{ano_modelo}.pkl")

        # 2. Carregar Modelo
        path_modelo = os.path.join(PASTA_MODELOS, f"{NOME_BASE_MODELO}_{(ano_modelo)}.pkl")
        if not os.path.exists(path_modelo):
            print(f" ERRO: Modelo {path_modelo} não encontrado. Pulando safra.")
            continue
        
        model = joblib.load(path_modelo)

        # 3. Filtrar Dataset para a Safra inteira (Otimização)
        # Pegamos tudo dessa janela de uma vez para não filtrar o DF gigante mil vezes
        df_safra = df[
            (df['data_ocorrencia'] >= data_inicio_safra) & 
            (df['data_ocorrencia'] <= data_fim_safra)
        ].copy()

        if df_safra.empty:
            print("   Sem dados para este período no CSV.")
            continue

        # 4. Loop Temporal (Passo a passo dentro da safra)
        data_atual = data_inicio_safra
        acuracia_media = []
        
        while data_atual <= data_fim_safra:
            
            # Recorte cumulativo: do início da safra (01/09) até o dia atual
            df_dia = df_safra[df_safra['data'] <= data_atual].copy()
            
            # Remove duplicatas (último status de cada ocorrência até o momento)
            if not df_dia.empty:
                df_dia = df_dia.sort_values('data').drop_duplicates(subset='ocorrencia_id', keep='last')
            
            # Se não tem nada acumulado (início da safra), pula
            if df_dia.empty:
                data_atual += timedelta(days=STEP_DAYS)
                continue

            # --- PREPARAÇÃO PARA PREDIÇÃO ---
            cols_ignore = ['ocorrencia_id', 'data', 'data_ocorrencia', 'target', 'safra']
            
            cols_existentes_drop = [c for c in cols_ignore if c in df_dia.columns]
            X = df_dia.drop(columns=cols_existentes_drop)

            # Reordena colunas conforme o modelo exige (Segurança XGBoost)
            if hasattr(model, "feature_names_in_"):
                # Filtra apenas colunas que o modelo conhece e na ordem certa
                # Se faltar coluna no DF que o modelo precisa, vai dar erro aqui
                X = X[model.feature_names_in_]

            try:
                # Predição
                raw_pred = model.predict(X)
                
                # CORREÇÃO: Clip para garantir intervalo [0, 1]
                # Remove problemas de -0.05 ou 1.08
                df_dia['predito'] = np.clip(raw_pred, 0, 1)
                
                #conta o numero de acertou igual a 1 para saber quantos acertos teve
                df_dia['target'] = df_dia['target'].astype(int)
                # ... (logo após o np.clip) ...
                df_dia['predito'] = np.clip(raw_pred, 0, 1)

                # ============================================================
                #  CÁLCULO ROBUSTO (Matriz de Confusão Explícita)
                # ============================================================
                # 1. Garante binário INTEIRO (0 ou 1) para ambos
                y_true = df_dia['target'].astype(int)
                y_pred_bin = (df_dia['predito'] >= 0.77).astype(int)

                # 2. Calcula os 4 quadrantes matematicamente
                # VP (Verdadeiro Positivo): Era 1 e previu 1
                VP = ((y_true == 1) & (y_pred_bin == 1)).sum()
                
                # VN (Verdadeiro Negativo): Era 0 e previu 0
                VN = ((y_true == 0) & (y_pred_bin == 0)).sum()
                
                # FP (Falso Positivo): Era 0 e previu 1 (Alarme Falso)
                FP = ((y_true == 0) & (y_pred_bin == 1)).sum()
                
                # FN (Falso Negativo): Era 1 e previu 0 (Perdeu a doença)
                FN = ((y_true == 1) & (y_pred_bin == 0)).sum()

                # 3. Métricas Derivadas
                total = len(df_dia)
                total_acertos = VP + VN
                acuracia = (total_acertos / total * 100) if total > 0 else 0
                acuracia_media.append(acuracia)
                
                # Recall (Sensibilidade): Capacidade de detectar a doença
                total_doenca = VP + FN
                recall = (VP / total_doenca * 100) if total_doenca > 0 else 0
                
                # Especificidade: Capacidade de acerta o "saudável"
                total_saudavel = VN + FP
                especificidade = (VN / total_saudavel * 100) if total_saudavel > 0 else 0
                # ============================================================

                # Gera GeoJSON (Código igual ao anterior)
                features = df_dia.apply(gerar_feature_geojson, axis=1).tolist()

                # ... (Montagem do geojson_data igual ao anterior) ...
                # ... (Salvamento do arquivo igual ao anterior) ...
                
                #  PRINT "RAIO-X" CORRIGIDO
                print(f"    Raio-X do dia {data_atual.strftime('%d/%m')}:")
                print(f"      - Total Pontos: {total}")
                print(f"      - Acurácia:     {total_acertos} ({acuracia:.1f}%)")
                print(f"      - Cenário Real: {total_doenca} Doentes | {total_saudavel} Saudáveis")
                print(f"      - Matriz Conf.: VP={VP} | FP={FP} | VN={VN} | FN={FN}")
                print(f"      - O Modelo viu: {VP} doentes (Recall: {recall:.1f}%)")
                if total_saudavel > 0:
                    print(f"      - Acerto Saud.: {VN} saudáveis (Especif.: {especificidade:.1f}%)")
                print("-" * 40)
                acertos = total_acertos
                
                # Gera Lista de Features GeoJSON
                features = df_dia.apply(gerar_feature_geojson, axis=1).tolist()

                # Monta Objeto Final
                geojson_data = {
                    "type": "FeatureCollection",
                    "metadata": {
                        "data_referencia": data_atual.strftime('%Y-%m-%d'),
                        "safra_referencia": str(safra_alvo),
                        "modelo_usado": f"{NOME_BASE_MODELO}_{ano_modelo}.pkl",
                        "total_pontos": len(features)
                    },
                    "features": features
                }

                # Salva
                filename = f"mapa_{safra_alvo}_{data_atual.strftime('%Y-%m-%d')}.geojson"
                caminho_salvar = os.path.join(OUTPUT_FOLDER, filename)
                
                with open(caminho_salvar, 'w', encoding='utf-8') as f:
                    json.dump(geojson_data, f, ensure_ascii=False)
                
                
                print(f"    Gerado: {filename} ({len(features)} pts) acertos: {acertos}")

            except Exception as e:
                print(f"    Erro em {data_atual.date()}: {e}")

            # Avança o tempo
            data_atual += timedelta(days=STEP_DAYS)

    print("\n Processamento de todas as safras finalizado.")
    acuracia_media_final = np.mean(acuracia_media) if acuracia_media else 0
    print(f"acuracia_media_final: {acuracia_media_final:.2f}%")