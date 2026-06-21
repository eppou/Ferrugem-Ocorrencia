# Okay se vc caiu nesse codigo peço desculpa ele realmente é uma abordagem diferente eu juro, assim como o visualization 1 ele abre um site para visualizar resultados
# Apos os diferentes modelos percebi que uma abordagem casando regressao e classificação fosse interessante
# para um serviço integrado creio que essa seja a melhor abordagem e para isso tem esse benchmark, para gera-lo proposta_classificaçao_hibrida
# se estiver tentanto replicar peço desculpa pela falta de constancia mas no fim tudo faz sentido e pense que o codigo fica com o historico do passado sempre ali pra revisitar :)
# enfim creio que seja a analise final e deixo esse leve desabafo para mim mesmo ou para futuro leitor, boa sorte XD

import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- Configuração da Página ---
st.set_page_config(page_title="Dashboard de Benchmark Agronômico", layout="wide", page_icon="🚜")

# --- Funções de Carregamento de Dados ---
@st.cache_data
def load_data(file):
    # Lê os arquivos usando o separador e decimal que configuramos no script anterior
    return pd.read_csv(file, sep=';', decimal=',')

# --- Interface Principal ---
st.title("🚜 Dashboard de Benchmark: Modelo Híbrido vs Classificador")
st.markdown("Analise as métricas de performance do seu modelo preditivo de doenças ao longo das safras e períodos.")

# --- Barra Lateral (Sidebar) ---
st.sidebar.header("📂 Configuração de Dados")

# Permitir upload manual ou usar caminho padrão
usar_upload = st.sidebar.checkbox("Fazer upload manual dos CSVs?", value=False)

df_geral, df_periodos = None, None

if usar_upload:
    file_geral = st.sidebar.file_uploader("Upload benchmark_safra_geral.csv", type=['csv'])
    file_periodos = st.sidebar.file_uploader("Upload benchmark_por_periodo.csv", type=['csv'])
    if file_geral: df_geral = load_data(file_geral)
    if file_periodos: df_periodos = load_data(file_periodos)
else:
    # Tenta ler da pasta local 'benchmarks' se existir
    path_geral = os.path.join("benchmarks", "benchmark_safra_geral.csv")
    path_periodos = os.path.join("benchmarks", "benchmark_por_periodo.csv")
    
    try:
        df_geral = load_data(path_geral)
        df_periodos = load_data(path_periodos)
        st.sidebar.success("Arquivos locais carregados com sucesso!")
    except FileNotFoundError:
        st.sidebar.error("Arquivos não encontrados na pasta 'benchmarks/'. Marque a caixa acima para fazer upload.")

# Se os dados foram carregados, montamos os gráficos
if df_geral is not None:
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Filtros de Visualização")
    
    visao = st.sidebar.radio("Selecione a Visão:", ["Safra Geral", "Por Período"])
    
    # Extrair as métricas base (tirando o prefixo ClassOnly_ ou Hibrido_)
    colunas_metricas = [c.replace('ClassOnly_', '') for c in df_geral.columns if c.startswith('ClassOnly_')]
    metrica_alvo = st.sidebar.selectbox("Selecione a Métrica Principal:", colunas_metricas, index=colunas_metricas.index('F1_Score') if 'F1_Score' in colunas_metricas else 0)
    
    # Preparar dados para o gráfico (Transformar colunas em linhas para o Plotly - Melt)
    def preparar_dados_plot(df, metrica, agrupar_por):
        cols_interesse = [agrupar_por, f'ClassOnly_{metrica}', f'Hibrido_{metrica}']
        df_plot = df[cols_interesse].copy()
        df_melted = df_plot.melt(id_vars=[agrupar_por], 
                                 value_vars=[f'ClassOnly_{metrica}', f'Hibrido_{metrica}'],
                                 var_name='Modelo', value_name='Valor')
        # Limpar o nome do modelo para ficar mais bonito no gráfico
        df_melted['Modelo'] = df_melted['Modelo'].str.replace(f'_{metrica}', '')
        return df_melted

    # --- FUNÇÃO PARA PLOTAR A MÉDIA GERAL ---
    def plotar_media_geral(df_plot, metrica):
        st.markdown("---")
        st.subheader(f"🎯 Média Geral Combinada: {metrica}")
        
        # Calcula a média agrupando apenas pelo modelo
        df_media = df_plot.groupby('Modelo')['Valor'].mean().reset_index()
        
        # Cria um gráfico de barras com os valores escritos na própria barra
        fig_media = px.bar(df_media, x='Modelo', y='Valor', color='Modelo', text_auto='.3f',
                           title=f"Média de {metrica} (Com base nos filtros atuais)",
                           color_discrete_sequence=['#EF553B', '#00CC96'])
        
        # Ajusta o texto para ficar fora da barra, facilitando a leitura
        fig_media.update_traces(textposition='outside')
        # Reduz a altura do gráfico para não ocupar tanto espaço vertical
        fig_media.update_layout(height=400) 
        
        st.plotly_chart(fig_media, use_container_width=True)

    if visao == "Safra Geral":
        st.header(f"📈 Visão Geral por Safra: {metrica_alvo}")
        
        safras_disponiveis = df_geral['Safra'].unique().tolist()
        safras_selecionadas = st.sidebar.multiselect("Filtrar Safras:", safras_disponiveis, default=safras_disponiveis)
        
        df_filtrado = df_geral[df_geral['Safra'].isin(safras_selecionadas)]
        df_plot = preparar_dados_plot(df_filtrado, metrica_alvo, 'Safra')
        
        # Garantir que Safra seja string para o eixo X não ficar com "2021.5"
        df_plot['Safra'] = df_plot['Safra'].astype(str)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de Barras Agrupadas
            fig_bar = px.bar(df_plot, x='Safra', y='Valor', color='Modelo', barmode='group',
                             title=f"Comparativo de {metrica_alvo} (Barras)",
                             color_discrete_sequence=['#EF553B', '#00CC96'])
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col2:
            # Gráfico de Linhas
            fig_line = px.line(df_plot, x='Safra', y='Valor', color='Modelo', markers=True,
                               title=f"Evolução de {metrica_alvo} (Linha)",
                               color_discrete_sequence=['#EF553B', '#00CC96'])
            st.plotly_chart(fig_line, use_container_width=True)
            
        # Adicionando o gráfico de média geral
        plotar_media_geral(df_plot, metrica_alvo)
            
        st.markdown("---")
        st.subheader("Tabela de Dados (Safra Geral)")
        st.dataframe(df_filtrado, use_container_width=True)

    elif visao == "Por Período" and df_periodos is not None:
        st.header(f"📊 Visão por Período da Safra: {metrica_alvo}")
        
        safras_disponiveis = df_periodos['Safra'].unique().tolist()
        safra_alvo = st.sidebar.selectbox("Selecione a Safra para Análise de Período:", safras_disponiveis)
        
        df_filtrado = df_periodos[df_periodos['Safra'] == safra_alvo]
        df_plot = preparar_dados_plot(df_filtrado, metrica_alvo, 'Periodo')
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_bar = px.bar(df_plot, x='Periodo', y='Valor', color='Modelo', barmode='group',
                             title=f"Comparativo de {metrica_alvo} na Safra {safra_alvo}",
                             color_discrete_sequence=['#EF553B', '#00CC96'])
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col2:
            fig_line = px.line(df_plot, x='Periodo', y='Valor', color='Modelo', markers=True,
                               title=f"Curva de {metrica_alvo} na Safra {safra_alvo}",
                               color_discrete_sequence=['#EF553B', '#00CC96'])
            # Garante que a linha siga a ordem lógica dos períodos
            fig_line.update_xaxes(categoryorder='category ascending')
            st.plotly_chart(fig_line, use_container_width=True)

        # Adicionando o gráfico de média geral
        plotar_media_geral(df_plot, metrica_alvo)

        st.markdown("---")
        st.subheader(f"Tabela de Dados (Safra {safra_alvo})")
        st.dataframe(df_filtrado, use_container_width=True)

else:
    st.info("👈 Por favor, carregue os arquivos CSV na barra lateral ou certifique-se de que eles estão na pasta 'benchmarks/' para visualizar o dashboard.")