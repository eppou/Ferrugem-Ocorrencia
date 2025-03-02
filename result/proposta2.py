from sqlalchemy import create_engine

from config import Config
from helpers.feature_importance import calculate_importance_avg, calculate_k_best, calculate_percentile
from helpers.input_output import get_latest_file
from helpers.result import write_result, read_result

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score,classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
#import seaborn as sns

import re

def maior_sequencia_uns(vetor):
    vetor_str = ''.join(map(str, vetor))  # Converte para string
    sequencias = re.finditer(r'1+', vetor_str)  # Encontra todas as sequências de 1's
    
    maior_seq = max(sequencias, key=lambda x: len(x.group()), default=None)  # Pega a maior
    
    if maior_seq:
        return maior_seq.start(), maior_seq.end() - 1, maior_seq.group()  # Índices e sequência
    else:
        return None  # Caso não haja 1's no vetor

def run(execution_started_at: datetime, cfg: Config, safras: list = None):    
    
    '''
    # 1. Carregar os dados
    df = pd.read_csv(get_latest_file("features", "features_SI.csv"))

    # Balanceamento: Igualar a quantidade de dados entre target=1 e target=0
    target_1 = df[df['target'] == 1]
    target_0 = df[df['target'] == 0].sample(n=target_1.shape[0], random_state=42)
    
    df = pd.concat([target_0, target_1])

    # 2. Separar os dados em features (X) e target (y)
    X = df.drop(columns=['target', 'data_ocorrencia', 'ocorrencia_id', 'safra'])
    y = df['target']

    # Configuração do K-Fold Cross-Validation
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Para armazenar as métricas de cada fold
    accuracies, precisions, recalls, f1_scores,r2s = [], [], [], [], []
    conf_matrix_total = np.zeros((2, 2))  # Matriz de confusão acumulada

    # Loop pelos folds
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n🚀 Fold {fold}/{k_folds}")

        # Separar dados de treino e teste para o fold atual
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Criar e treinar o modelo Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Fazer previsões
        y_pred = rf_model.predict(X_test)

        # Avaliação do fold
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Armazena os resultados do fold
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        r2s.append(r2)
        conf_matrix_total += conf_matrix

        # Exibir métricas do fold atual
        print(f"Acurácia: {accuracy:.4f}, Precisão: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, R2: {r2:.4f}")

    # Média das métricas após K-Fold
    print("\n📊 **Resultados Médios do K-Fold**")
    print(f"Acurácia Média: {np.mean(accuracies):.4f}")
    print(f"Precisão Média: {np.mean(precisions):.4f}")
    print(f"Recall Médio: {np.mean(recalls):.4f}")
    print(f"F1-score Médio: {np.mean(f1_scores):.4f}")

    # Exibir matriz de confusão acumulada
    print("\nMatriz de Confusão Acumulada (Soma dos Folds):")
    print(conf_matrix_total)
    # print vp,vn,fp,fn
    vp = conf_matrix_total[1][1]
    vn = conf_matrix_total[0][0]
    fp = conf_matrix_total[0][1]
    fn = conf_matrix_total[1][0]
    print(f"VP: {vp}, VN: {vn}, FP: {fp}, FN: {fn}")

    # Importância das Features (baseada no último modelo treinado)
    importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    print("\n📌 Importância das Features:")
    print(importances.sort_values(ascending=False))
    '''
    
    #--------------------------------------------------------------------------------------------------------#
    
    
    
    
    
    
    
    
    
    
    # pega o conjunto de teste e separa para cada um dos ocorrencia_id
    # 1. Carregar os dados
    df = pd.read_csv(get_latest_file("features", "features_SI.csv"))
    #agrupa por ocorrencia_id
    df_grouped = df.groupby('ocorrencia_id')
    
    #separa 80% dos grupos para treino e 20% para teste
    df_grouped_train, df_grouped_test = train_test_split(list(df_grouped), test_size=0.2, random_state=42)
    
    # cria um dataframe com os grupos de treino
    df_grouped_train = pd.concat([group for name, group in df_grouped_train])
    
    # Balanceamento: Igualar a quantidade de dados entre target=1 e target=0
    target_1 = df_grouped_train[df_grouped_train['target'] == 1]
    target_0 = df_grouped_train[df_grouped_train['target'] == 0].sample(n=target_1.shape[0], random_state=42)
    
    df_grouped_train = pd.concat([target_0, target_1])
    
    # 2. Separar os dados em features (X) e target (y)
    X = df_grouped_train.drop(columns=['target', 'data_ocorrencia', 'ocorrencia_id', 'safra'])
    y = df_grouped_train['target']
    
    # Configuração do K-Fold Cross-Validation
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Para armazenar as métricas de cada fold
    accuracies, precisions, recalls, f1_scores,r2s = [], [], [], [], []
    conf_matrix_total = np.zeros((2, 2))  # Matriz de confusão acumulada
    
    # Loop pelos folds
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n🚀 Fold {fold}/{k_folds}")
    
        # Separar dados de treino e teste para o fold atual
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
        # Criar e treinar o modelo Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
    
        # Fazer previsões
        y_pred = rf_model.predict(X_test)
    
        # Avaliação do fold
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
        # Armazena os resultados do fold
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        r2s.append(r2)
        conf_matrix_total += conf_matrix
    
        # Exibir métricas do fold atual
        print(f"Acurácia: {accuracy:.4f}, Precisão: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, R2: {r2:.4f}")
        
    # Média das métricas após K-Fold
    print("\n📊 **Resultados Médios do K-Fold**")
    print(f"Acurácia Média: {np.mean(accuracies):.4f}")
    print(f"Precisão Média: {np.mean(precisions):.4f}")
    print(f"Recall Médio: {np.mean(recalls):.4f}")
    print(f"F1-score Médio: {np.mean(f1_scores):.4f}")
    print(f"R2 Médio: {np.mean(r2s):.4f}")


    erros = []
    
    #itera sobre os grupos
    for name, group in df_grouped_test:
        #separa os dados em features e target
        X = group.drop(columns=['target', 'data_ocorrencia', 'ocorrencia_id', 'safra'])
        y = group['target']
        #faz a previsão
        y_pred = rf_model.predict(X)
        #verifica se tem algum 1 na previsão
        if 1 in y_pred:
            inicio = maior_sequencia_uns(y_pred)
            if inicio != None:
                erros.append(inicio[0])               
    
    print(f"Erro médio: {np.mean(erros)}")
    print(f"Incapaz de inferir: {len(df_grouped_test) - len(erros)}")
    
