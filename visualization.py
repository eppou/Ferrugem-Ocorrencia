import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard de Modelos", layout="wide")

st.title("📊 Dashboard Interativo de Modelos — Métricas, Filtros e Comparações")

uploaded = st.file_uploader("Envie seu CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    
    # Tratamento de nomes para Inglês (Data Cleaning)
    if 'name_model' in df.columns:
        df['name_model'] = df['name_model'].replace({
            'RF_regressor_safra': 'RF Regressor Season (Precipitation)',
            'RF_regressor_safra_temp': 'RF Regressor Season ',
            'RF_regressao_talhao' : 'RF Regressor Field (Precipitation)',
            'RF_regressao_talhao_temp' : 'RF Regressor Field ',
            'RF_classificador_temp': 'RF Classificator ',
            'RF_classificador': 'RF Classificator (Precipitation)',
            
            'XGB_regressor_safra': 'XGB Regressor Season (Precipitation)',
            'XGB_regressor_safra_temp': 'XGB Regressor Season ',
            'XGBoost_regressao_talhao' : 'XGB Regressor Field (Precipitation)',
            'XGBoost_regressao_talhao_temp' : 'XGB Regressor Field ',
            'XGB_classificador_temp': 'XGB Classificator ',
            'XGB_classificador': 'XGB Classificator (Precipitation)',
        })

    st.success("Arquivo carregado com sucesso!")

    with st.expander("Pré-visualizar dados"):
        st.dataframe(df.head())

    # Identificação de métricas
    colunas_erro = [c for c in df.columns if "Erro" in c or "MAE" in c or "RMSE" in c]
    colunas_extra = ["R2", "F1", "Recall", "Precision", "VP", "FP"]
    colunas_barra = colunas_erro + colunas_extra

    # ============================================================================
    # CONFIGURAÇÃO DE TRADUÇÃO PARA OS GRÁFICOS (PT -> EN)
    # ============================================================================
    # Dicionário global para renomear eixos automaticamente
    labels_en = {
        "ano": "Year",
        "name_model": "Model",
        "R2": "Mean R²",
        "value": "Value",
        "variable": "Metric"
    }

    # ============================================================================
    # SB — FILTROS
    # ============================================================================
    st.sidebar.header("Filtros")

    tipo_modelo = st.sidebar.selectbox(
        "Filtrar modelos contendo:",
        ["Todos", "Season", "Field", "Classificator", "Regressor"]
    )

    modelos = sorted(df["name_model"].unique())

    if tipo_modelo != "Todos":
        modelos = [m for m in modelos if tipo_modelo.lower() in m.lower()]

    modelos_sel = st.sidebar.multiselect(
        "Modelos",
        modelos,
        default=modelos
    )

    df_f = df[df["name_model"].isin(modelos_sel)]

    # ----------- Seleção de anos
    anos_disponiveis = sorted(df_f["ano"].unique())
    anos_sel = st.sidebar.multiselect(
        "Selecione os anos",
        anos_disponiveis,
        default=anos_disponiveis
    )

    df_f = df_f[df_f["ano"].isin(anos_sel)]

    # ----------- Seleção métrica de erro
    erro_select = st.sidebar.selectbox(
        "Métrica de erro (para gráficos de linha)",
        colunas_erro
    )

    st.markdown(f"## 📈 Análise Temporal: R² e `{erro_select}`")
    
    # ----------------------------------------------------------------------
    # FUNÇÃO PARA CONFIGURAR O DOWNLOAD (Cliente-Side)
    # ----------------------------------------------------------------------
    def get_config(nome_arquivo):
        return {
            'toImageButtonOptions': {
                'format': 'jpeg',        
                'filename': nome_arquivo,
                'height': 600,          
                'width': 1200,          
                'scale': 3 # Aumentei para 3 para ficar super nítido em papers (300dpi aprox)
            },
            'displayModeBar': True      
        }
    
    st.info("ℹ️ Para baixar: Passe o mouse sobre o gráfico e clique no ícone da câmera (📸).")

    # ============================================================================
    # GRÁFICO 1: MÉDIA POR ANO
    # ============================================================================
    df_mean = df_f.groupby("ano").agg({
        "R2": "mean",
        erro_select: "mean"
    }).reset_index()

    fig_mean = px.line(
        df_mean,
        x="ano",
        y=["R2", erro_select],
        markers=True,
        title="Annual Averages - Filtered Models", # Título em Inglês
        labels=labels_en, # Aplica tradução dos eixos
        template="simple_white" # Fundo branco para artigos
    )
    # Ajuste fino para limpar o nome da legenda (remove 'variable=...')
    fig_mean.update_layout(legend_title_text='Metric')
    
    st.plotly_chart(fig_mean, use_container_width=True, config=get_config("annual_averages"))

    # ============================================================================
    # GRÁFICO 2: LINHAS POR MODELO (R2)
    # ============================================================================
    st.markdown("### 📊 R² médio por modelo")

    fig_r2 = px.line(
        df_f,
        x="ano",
        y="R2",
        color="name_model",
        markers=True,
        title="Mean R² by Model",
        labels=labels_en,
        template="simple_white"
    )
    st.plotly_chart(fig_r2, use_container_width=True, config=get_config("r2_by_model"))

    # ============================================================================
    # GRÁFICO 3: LINHAS POR MODELO (ERRO)
    # ============================================================================
    st.markdown(f"### 📊 {erro_select} por modelo")

    fig_err = px.line(
        df_f,
        x="ano",
        y=erro_select,
        color="name_model",
        markers=True,
        title=f"{erro_select} by Model",
        labels=labels_en,
        template="simple_white"
    )
    st.plotly_chart(fig_err, use_container_width=True, config=get_config(f"error_{erro_select}"))

    # ============================================================================
    # GRÁFICO 4: BARRAS
    # ============================================================================
    st.markdown("---")
    st.markdown("## 📊 Comparação entre Modelos — Médias por Modelo")

    barra_select = st.selectbox(
        "Selecione a métrica para o gráfico de barras:",
        colunas_barra
    )

    df_barra = df_f.groupby("name_model").agg({barra_select: "mean"}).reset_index()
    df_barra = df_barra.sort_values(by=barra_select, ascending=False)

    fig_bar = px.bar(
        df_barra,
        x="name_model",
        y=barra_select,
        title=f"Mean by Model — {barra_select}",
        text_auto=True,
        labels=labels_en,
        template="simple_white"
    )
    st.plotly_chart(fig_bar, use_container_width=True, config=get_config(f"bar_chart_{barra_select}"))

else:
    st.info("Envie o arquivo CSV para começar.")