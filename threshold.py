import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
from datetime import datetime

def analisar_e_salvar_graficos(caminho_csv):
    df = pd.read_csv(caminho_csv)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"analise_resultados_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Tratamento do valor 999
    limiares_reais = sorted([x for x in df['Limiar_Reg_Dias'].unique() if x != 999])
    valor_visual_999 = max(limiares_reais) + 10 if limiares_reais else 999
    df['Limiar_Plot'] = df['Limiar_Reg_Dias'].replace(999, valor_visual_999)
    df['Rotulo_Trava'] = df['Limiar_Reg_Dias'].replace(999, 'Sem Trava').astype(str)
    
    sns.set_theme(style="whitegrid", context="talk")

    # --- GRAFICO 1: HEATMAPS (RECALL E ERRO) ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(20, 8))
    for i, metrica in enumerate(['Recall', 'Erro_Medio']):
        pivot = df.pivot_table(index='Limiar_Class', columns='Limiar_Reg_Dias', values=metrica)
        cmap = "YlGnBu" if metrica == 'Recall' else "RdYlGn_r"
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap=cmap, ax=axes1[i], center=0 if i==1 else None)
        axes1[i].set_title(f'Média Global: {metrica}')
    
    plt.tight_layout()
    fig1.savefig(f"{output_dir}/1_heatmaps_globais.png")
    plt.close()

    # --- GRAFICO 2: PARETO POR SAFRA ---
    # Este gráfico mostra se o comportamento é consistente entre diferentes anos
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='Erro_Medio', y='Recall', hue='Safra', style='Safra', 
                    size='Limiar_Class', palette='viridis', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.title('Dispersão Recall vs Erro por Safra')
    plt.savefig(f"{output_dir}/2_pareto_por_safra.png")
    plt.close()

    # --- GRAFICO 3: EVOLUÇÃO TEMPORAL POR SAFRA (FACET GRID) ---
    # Mostra o Recall conforme relaxamos a trava, separado por Safra
    g = sns.FacetGrid(df, col="Safra", hue="Limiar_Class", height=5, aspect=1.2, col_wrap=2)
    g.map(sns.lineplot, "Limiar_Plot", "Recall", marker="o")
    
    # Ajustar labels do eixo X para cada faceta
    for ax in g.axes.flat:
        ax.set_xticks(df['Limiar_Plot'].unique())
        ax.set_xticklabels(df['Rotulo_Trava'].unique(), rotation=45)
    
    g.add_legend(title="Limiar Classificador")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Evolução do Recall por Safra conforme Limiar de Regressão')
    g.savefig(f"{output_dir}/3_evolucao_por_safra.png")
    plt.close()

    # --- GRAFICO 4: BOXPLOT DE ESTABILIDADE DO ERRO ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Limiar_Reg_Dias', y='Erro_Medio', palette='coolwarm')
    plt.title('Variabilidade do Erro Médio conforme Limiar de Trava')
    plt.savefig(f"{output_dir}/4_estabilidade_erro.png")
    plt.close()

    print(f"\n✅ Análise concluída! 4 gráficos salvos na pasta: {output_dir}")
    
    # Recomendação simplificada por Safra
    print("\n💡 RESUMO POR SAFRA (Melhores Recalls):")
    resumo = df.sort_values(['Safra', 'Recall'], ascending=[True, False]).groupby('Safra').head(1)
    print(resumo[['Safra', 'Limiar_Class', 'Limiar_Reg_Dias', 'Recall', 'Erro_Medio']].to_string(index=False))

if __name__ == "__main__":
    lista_arquivos = glob.glob('grid_search_results_*.csv')
    if not lista_arquivos:
        print("❌ Nenhum CSV encontrado.")
    else:
        recente = max(lista_arquivos, key=os.path.getctime)
        analisar_e_salvar_graficos(recente)