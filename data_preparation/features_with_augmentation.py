import pandas as pd
from datetime import datetime, timedelta
from helpers.input_output import get_latest_file, output_file
from data_preparation.instances import calculate_planting_start_date
from config import Config



#TODO: GET PRECIPTATION DATA FROM THE DB

'''
 Para cada intervalo de dias (7, 14, 21, 28 e 35 dias), calculamos duas features:
 1. Precipitação acumulada no intervalo (soma da coluna 'prec' no intervalo especificado).
 2. Número de dias com precipitação no intervalo (quantidade de dias com valores de 'prec' > 0).
 As features geradas são adicionadas à instância atual com os seguintes nomes:
 - 'prec_acumulada_{intervalo}_dias': representa a soma da precipitação no intervalo de {intervalo} dias.
 - 'dias_chuva_{intervalo}_dias': representa a quantidade de dias com chuva (> 0 de precipitação) no intervalo de {intervalo} dias.
 O processo é repetido para cada data de análise (data_atual) e para cada intervalo definido.
'''

def run(execution_started_at: datetime, cfg: Config, harvest: list = None):
    # 1. Carregar os dados das ocorrências e de precipitação
    ocorrencias = pd.read_csv(get_latest_file("instances", "instances_all.csv"))  # Dataset de ocorrências de ferrugem
    precipitacao = pd.read_csv(get_latest_file("precipitation", "precipitation.csv"))  # Dataset de precipitação

    # Exemplo de estrutura esperada:
    # ocorrencias: DataFrame com colunas ['data_ocorrencia', 'segment_id', ...]
    # precipitacao: DataFrame com colunas ['data', 'segment_id', 'precipitacao']

    # Converter datas para formato datetime
    ocorrencias['data_ocorrencia'] = pd.to_datetime(ocorrencias['data_ocorrencia'])
    precipitacao['date_precipitation'] = pd.to_datetime(precipitacao['date_precipitation'])

    # 2. Iterar pelas ocorrências para gerar instâncias diárias
    instancias = []


    for _, row in ocorrencias.iterrows():
        #print para saber em qual iteração está
        print(f"Processing row {row['ocorrencia_id']} date {row['data_ocorrencia']}")
        
        segment_id = row['segment_id_precipitation']
        data_ocorrencia = row['data_ocorrencia']

        data_plantio = calculate_planting_start_date(row)
        data_plantio = data_plantio + timedelta(days=4)

        # Retroceder dias até o início da safra
        data_atual = data_ocorrencia + pd.Timedelta(days=3)
        while data_atual >= data_plantio:
            # Criar uma nova instância para a data atual
            instancia = row.to_dict()  # Copiar todas as colunas da ocorrência original
            instancia['data'] = data_atual  # Atualizar a data para a instância
 
            instancia['target'] = 1 if (data_ocorrencia - pd.Timedelta(days=3) <= data_atual <= data_ocorrencia + pd.Timedelta(days=3)) else 0

            for intervalo in range(7, 36, 7):  # Intervalos de 7, 14, 21, 28 e 35 dias
                precipitacao_intervalo = precipitacao[
                    (precipitacao['segment_id'] == segment_id) &
                    (precipitacao['date_precipitation'] >= data_atual - pd.Timedelta(days=intervalo)) &
                    (precipitacao['date_precipitation'] < data_atual)
                ]

                # Feature: precipitação acumulada no intervalo
                instancia[f'prec_acumulada_{intervalo}_dias'] = precipitacao_intervalo['prec'].sum()

                # Feature: número de dias com precipitação no intervalo
                instancia[f'dias_chuva_{intervalo}_dias'] = (precipitacao_intervalo['prec'] > 0).sum()

            # Adicionar a instância à lista final
            instancias.append(instancia)

            # Retroceder dois dias
            data_atual -= pd.Timedelta(days=2)

    # 4. Criar um DataFrame com as instâncias geradas
    df_instancias = pd.DataFrame(instancias)

    # 5. Salvar as instâncias em um CSV
    df_instancias.to_csv(
        output_file(execution_started_at,"features", "features_all_SI.csv")
        , index=False
    )
    
    #filtra só as features de precipitação e occorencia_id,data_ocorrencia,safra
    df_instancias = df_instancias.filter(regex='prec_acumulada|dias_chuva|ocorrencia_id|data_ocorrencia|safra|target')
    df_instancias.to_csv(
        output_file(execution_started_at,"features", "features_SI.csv")
        , index=False
    )

    print(f"Instâncias diárias geradas e salvas em 'features_SI.csv'")
