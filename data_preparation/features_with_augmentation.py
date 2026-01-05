import pandas as pd
from datetime import datetime, timedelta
from helpers.input_output import get_latest_file, output_file
from data_preparation.instances import calculate_planting_start_date, calculate_phenological_stage
from config import Config
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
import numpy as np
from scipy.stats import skew


INTERVALOS = list(range(7, 56, 3))  # Pode alterar para range(7, 36, 7) se quiser apenas múltiplos de 7 e trabalhar com o intervalos iguais aos do Kemmer

enso_fases = {
    2004: "El Niño",
    2005: "Neutro",
    2006: "El Niño",
    2007: "La Niña",
    2008: "La Niña",
    2009: "El Niño",
    2010: "La Niña",
    2011: "La Niña",
    2012: "Neutro",
    2013: "Neutro",
    2014: "El Niño",
    2015: "El Niño",
    2016: "El Niño",
    2017: "Neutro",
    2018: "La Niña",
    2019: "Neutro",
    2020: "La Niña",
    2021: "La Niña",
    2022: "La Niña"
}


def carregar_dados():
    ocorrencias = pd.read_csv(get_latest_file("instances", "instances_all.csv"))
    precipitacao = pd.read_csv(get_latest_file("precipitation", "precipitation.csv"))

    ocorrencias['data_ocorrencia'] = pd.to_datetime(ocorrencias['data_ocorrencia'])
    precipitacao['date_precipitation'] = pd.to_datetime(precipitacao['date_precipitation'])
    precipitacao = precipitacao.rename(columns={'precipitacao': 'prec'})

    return ocorrencias, precipitacao

def agrupar_e_calcular_skew(lista, tamanho_grupo):
    agrupado = [
        sum(lista[i:i+tamanho_grupo])
        for i in range(0, len(lista), tamanho_grupo)
    ]
    
    if len(set(agrupado)) <= 1 or np.std(agrupado) < 1e-8:
        return 0.0  # ou np.nan
    return skew(agrupado)

def analise_distribuicao(instancia, prec, data_plantio, data_base):
    """
    Realiza a análise de distribuição da precipitação em relação à data de ocorrência.
    
    Parâmetros:
    - instancia: dicionário com a chave 'data' (datetime)
    - prec: DataFrame com colunas 'date_precipitation' (datetime) e 'prec' (float)
    - tamanho_intervalo: número de dias de cada intervalo
    - limite_dias: total de dias a partir da data para considerar na análise
    """
    chuvas_por_intervalo = []
    intervalo_dias = data_base - data_plantio
    i = data_plantio

    #pega a chuva de todos os dias do plantio até a data base no df prec
    chuvas_por_intervalo = prec[
        (prec['date_precipitation'] >= data_plantio) &
        (prec['date_precipitation'] <= data_base)
    ]['prec'].tolist()
    

    chuvas_total = sum(chuvas_por_intervalo)
    if chuvas_total > 0:
        chuvas_normalizadas = [v / chuvas_total for v in chuvas_por_intervalo]
    else:
        chuvas_normalizadas = [0] * len(chuvas_por_intervalo)  # Sem chuva

    
    # Desvio padrão da distribuição (dispersão)
    distribuicao_dispersao = np.std(chuvas_por_intervalo)

    # Posição do pico (intervalo com mais chuva)
    pico = int(np.argmax(chuvas_por_intervalo))

    # Assimetria (skewness)
    distribuicao_assimetria = agrupar_e_calcular_skew(chuvas_normalizadas, 4) #lembrar de detalhar(o agrupar) isso para escrita posteriormente
    
    #Mediana
    mediana = np.median(chuvas_por_intervalo) 
    
    #Moda
    moda = max(set(chuvas_por_intervalo), key=chuvas_por_intervalo.count)
    
    #coeficiente de variação
    if np.mean(chuvas_por_intervalo) > 0:
        coeficiente_variacao = np.std(chuvas_por_intervalo) / np.mean(chuvas_por_intervalo)
    else:
        coeficiente_variacao = 0
        
        
    chuva_array = np.array(chuvas_por_intervalo)
    # Atualiza a instância com os resultados
    instancia['chuva_dispersao'] = distribuicao_dispersao
    instancia['chuva_pico_intervalo'] = pico
    instancia['chuva_pico_valor'] = chuvas_por_intervalo[pico]
    instancia['chuva_assimetria'] = distribuicao_assimetria
    instancia['chuva_mediana'] = int(np.argmin(np.abs(chuva_array - mediana)))
    instancia['chuva_mediana_valor'] = mediana
    instancia['chuva_moda'] = np.where(chuva_array == moda)[0][0]
    instancia['chuva_moda_valor'] = moda
    instancia['prec_acumulada'] = sum(chuvas_por_intervalo)
    instancia['dias_chuva'] = sum(1 for v in chuvas_por_intervalo if v > 0.5)
    instancia['dias_desde_plantio'] = intervalo_dias.days
    instancia['chuva_coeficiente_variacao'] = coeficiente_variacao
    instancia['chuva_media'] = np.mean(chuvas_por_intervalo)
        
def gerar_instancias_por_ocorrencia(row, precipitacao):
    instancias = []

    segment_id = row.segment_id_precipitation
    data_ocorrencia = row.data_ocorrencia
    data_plantio = calculate_planting_start_date(row._asdict())
    data_atual = data_ocorrencia

    prec_segment = precipitacao[precipitacao['segment_id'] == segment_id]

    while data_atual >= (data_plantio + timedelta(days=20)):
        instancia = row._asdict()
        instancia['data'] = data_atual
        instancia['target'] = int(data_ocorrencia - timedelta(days=7) <= data_atual <= data_ocorrencia + timedelta(days=7))

        #features precipitação
        analise_distribuicao(instancia, prec_segment,data_plantio, data_atual)
            
        #features de data
        instancia['plantio_mes'] = (data_plantio.month + 3) % 12    #features discretizada onde 0 é setembro e 11 é agosto
        instancia['plantio_semana'] = (data_plantio - datetime(data_plantio.year, 9, 1)).days // 7 #features discretizada onde 0 é a primeira semana de setembro e 51 é a última semana de agosto
        instancia['plantio_quinzena'] = (data_plantio - datetime(data_plantio.year, 9, 1)).days // 15 #feature discretizada quinzenal onde 0 é a primeira quinzena de setembro e 25 é a última quinzena de agosto
        
        
        #estágio fenológico
        instancia['estadio_fenologico'] = calculate_phenological_stage(data_plantio, data_atual)
        
        #data para puxar o enso
        ano = data_atual.year
        fenomeno = enso_fases.get(ano, "Neutro")
        if fenomeno == "El Niño":
            instancia['enso'] = 1
        elif fenomeno == "La Niña":
            instancia['enso'] = -1
        else:
            instancia['enso'] = 0       

        instancias.append(instancia)
        data_atual -= timedelta(days=2)

    return instancias


def processar_todas_instancias(ocorrencias, precipitacao, n_jobs=-1):
    print("Iniciando processamento paralelo das instâncias...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(gerar_instancias_por_ocorrencia)(row, precipitacao)
        for row in ocorrencias.itertuples(index=False)
    )
    # Achata a lista de listas
    instancias = [inst for sublist in results for inst in sublist]
    return pd.DataFrame(instancias)


def salvar_resultados(df_instancias, execution_started_at):
    df_instancias.to_csv(
        output_file(execution_started_at, "features", "features_all_SI.csv"),
        index=False
    )

    df_instancias.filter(
        regex='dias_desde_plantio|prec_acumulada|dias_chuva|ocorrencia_id|data|data_ocorrencia|estadio_fenologico|target|chuva_centro_massa|chuva_dispersao|chuva_pico|chuva_assimetria|chuva_mediana|chuva_moda|chuva_media|enso|feature_|chuva_coeficiente_variacao|latitude|longitude'
    ).to_csv(
        output_file(execution_started_at, "features", "features_SI.csv"),
        index=False
    )

    print("Instâncias diárias geradas e salvas.")


def run(execution_started_at: datetime, cfg: Config, harvest: list = None):
    ocorrencias, precipitacao = carregar_dados()
    df_instancias = processar_todas_instancias(ocorrencias, precipitacao)
    salvar_resultados(df_instancias, execution_started_at)
