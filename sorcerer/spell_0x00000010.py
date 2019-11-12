# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:05:50 2019

@author: Danilo Santos
"""
import pandas as pd

import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from grimoire.EnginneringForest import EnginneringForest


print(">>>> Iniciando a execução do script")
print(">> Cria o ambiente de logs")

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')


def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def reset_logger(name):
    from os import remove
    from os.path import isfile
    if isfile(name):
        remove(name)


reset_logger('logger_spell_0x00000010_eg.log')
reset_logger('logger_spell_0x00000010_es.log')
reset_logger('logger_spell_0x00000010_rf.log')
reset_logger('logger_spell_0x00000010_gd.log')
logger_eg = setup_logger('logger_eg', 'logger_spell_0x00000010_eg.log')
logger_es = setup_logger('logger_eg', 'logger_spell_0x00000010_es.log')
logger_rf = setup_logger('logger_rf', 'logger_spell_0x00000010_rf.log')
logger_gb = setup_logger('logger_gd', 'logger_spell_0x00000010_gd.log')

print("Ambiente de logs criado")
print(">> Carregando o dataset")

df_acute = pd.read_csv('datasets/acute/diagnosis.csv',
                       engine='c',
                       memory_map=True,
                       low_memory=True)

print(">> Dataset carregado com sucesso")

print(">> Imprimindo cabecalho do dataset")
print(df_acute.head())

print(">> Tamanho do dataset")
print(df_acute.shape)

print(">> Informações do dataset")
print(df_acute.info())


print(">> Inicia a separacao dos dados da classe alvo")
X = df_acute[['temperatura', 'nausea', 'dorlombar',
              'urinepushing', 'miccao', 'queimacao', 'inflamacao']]
# Labels
y = df_acute['target']


print(">> Prepara as configurações do loop")

numero_arvores = [7, 21, 35, 35]
numero_atributos_selecao = [1, 2, 3, 4]
numero_random_state = [0, 20, 40, 60, 80, 100, 120]
numero_interacoes = 5
dados_execucao_eg = []
dados_execucao_es = []
dados_execucao_rf = []
dados_execucao_gb = []

print(">> Inicia os loops para testar o código")

for na, ns in zip(numero_arvores, numero_atributos_selecao):
    tmp_execucao_eg = []
    tmp_execucao_es = []
    tmp_execucao_rf = []
    tmp_execucao_gb = []
    for rs in numero_random_state:
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.30,
                                                            random_state=rs,
                                                            shuffle=True,
                                                            stratify=y)
        tmp_execucao_eg.append(na)
        tmp_execucao_eg.append(rs)
        tmp_execucao_es.append(na)
        tmp_execucao_es.append(rs)
        tmp_execucao_rf.append(na)
        tmp_execucao_rf.append(rs)
        tmp_execucao_gb.append(na)
        tmp_execucao_gb.append(rs)
        for ni in range(numero_interacoes):
            # Execução do EG com dados categóricos e estratégia block
            model_eg = EnginneringForest(select_features=ns)
            model_eg.encoder_enable = True
            model_eg.encoder_target = True
            model_eg.encoder_data = True
            model_eg.estrategy_trainning = 1
            model_eg.is_data_categorical = True

            model_eg.fit(X_train, y_train)
            y_pred = model_eg.predict(X_test)
            y_test_coded = model_eg.encoder_y.transform(y_test)
            mac = accuracy_score(y_test_coded, y_pred)
            tmp_execucao_eg.append(mac)

            # Execução do EG com dados categóricos e estratégia single
            model_es = EnginneringForest(select_features=ns)
            model_es.encoder_enable = True
            model_es.encoder_target = True
            model_es.encoder_data = True
            model_es.estrategy_trainning = 0
            model_es.is_data_categorical = True

            model_es.fit(X_train, y_train)
            y_pred = model_es.predict(X_test)
            y_test_coded = model_es.encoder_y.transform(y_test)
            mac = accuracy_score(y_test_coded, y_pred)
            tmp_execucao_es.append(mac)

            # Execução do Random Forest
            model_rf = RandomForestClassifier(n_estimators=na,
                                              criterion='entropy')
            model_rf.fit(X_train, y_train)
            y_pred = model_rf.predict(X_test)
            mac = accuracy_score(y_test, y_pred)
            tmp_execucao_rf.append(mac)

            # Execução do Gradient Boosting
            model_gb = GradientBoostingClassifier(n_estimators=na)
            model_gb.fit(X_train, y_train)
            y_pred = model_gb.predict(X_test)
            mac = accuracy_score(y_test, y_pred)
            tmp_execucao_gb.append(mac)

        # salva os dados de teste
        dados_execucao_eg.append(tmp_execucao_eg)
        tmp_execucao_eg.clear()
        dados_execucao_es.append(tmp_execucao_es)
        tmp_execucao_es.clear()
        dados_execucao_rf.append(tmp_execucao_rf)
        tmp_execucao_rf.clear()
        dados_execucao_gb.append(tmp_execucao_gb)
        tmp_execucao_gb.clear()

logger_eg.info(dados_execucao_eg)
logger_es.info(dados_execucao_es)
logger_rf.info(dados_execucao_rf)
logger_gb.info(dados_execucao_gb)
