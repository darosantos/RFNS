#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:59:30 2019

@author: midas
"""

print(">>>> Iniciando a execução do script")
print(">> Este script executa o codigo do EnginneringForest")
print(">> Iniciando os imports standards")

import pandas as pd
import math
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from grimoire.EnginneringForest import EnginneringForest

print(">> Fim dos imports standards")
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
        
reset_logger('logger_spell_0x00000002_rs_80.log')
logger_spell = setup_logger('logger_spell', 'logger_spell_0x000000012_rs_80.log')
        
print("Ambiente de logs criado")
print(">> Carregando o dataset")

df_heart = pd.read_csv('../datasets/heart.csv',
                       engine='c', 
                       memory_map=True, 
                       low_memory=True)

print(">> Dataset carregado com sucesso")

print(">> Imprimindo cabecalho do dataset")
print( df_heart.head() )

print(">> Tamanho do dataset")
print(df_heart.shape)

print(">> Informações do dataset")
print(df_heart.info())


print(">> Inicia a separacao dos dados da classe alvo")
X=df_heart[['age', 'sex', 'cp', 'trestbps',  'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
# Labels
y=df_heart['target']


n_point_interations = math.ceil(df_heart.shape[1] / 2)
n_media_interations = 5
ef_accuracy = [[]]
ef_matrix_confusion = []
rf_accuracy = [[]]
rf_matrix_confusion = []

for i in range(n_media_interations):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, 
                                                        random_state=80, 
                                                        shuffle=True, 
                                                        stratify=y)
    
    print('>> Execucao do treino {0}'.format(i+1))
    logger_spell.info('N Media Interations = {0}'.format(i+1))
    ef_tmp_accuracy = []
    ef_tmp_matrix_confusion = []
    rf_tmp_accuracy = []
    rf_tmp_matrix_confusion = []
    ef_tmp_n_tree = []
    
    print('>> Inicia os testes do EF')
    for j in range(n_point_interations):
        print('>> Execucao EF {0}'.format(j+1))
        logger_spell.info('Execucao EF {0}'.format(j+1))
        model_eg = EnginneringForest(select_features=j+1)
        model_eg.fit(X_train, y_train)
        # model_eg.chunck = 32
        y_pred = model_eg.predict(X_test)
    
        mac = accuracy_score(y_test, y_pred)
        mcm = confusion_matrix(y_test,y_pred)
        
        ef_tmp_accuracy.append(mac)
        ef_tmp_matrix_confusion.append(mcm)
        ef_tmp_n_tree.append(len(model_eg.estimators_))
        
        print('Acuracia = {0}'.format(mac))
        print("Matriz de confusao \n{0}".format(str(mcm)))
    
        del model_eg
    ef_accuracy.append(ef_tmp_accuracy)
    ef_matrix_confusion.append(ef_tmp_matrix_confusion)
    logger_spell.info('Accuracy EF \n{0}'.format(ef_tmp_accuracy))
    logger_spell.info('Matrix Confusion EF \n{0}'.format(ef_matrix_confusion))

    print('>> Inicia os testes do RF')
    for j in ef_tmp_n_tree:
        print(">> N arvores para o RF {0}".format(j))
        logger_spell.info("N arvores para o RF {0}".format(j))
        model_rf = RandomForestClassifier(n_estimators=j, 
                                          criterion='entropy')
        model_rf.fit(X_train, y_train)
        # model_eg.chunck = 32
        y_pred = model_rf.predict(X_test)
    
        mac = accuracy_score(y_test, y_pred)
        mcm = confusion_matrix(y_test,y_pred)
        
        rf_tmp_accuracy.append(mac)
        rf_tmp_matrix_confusion.append(mcm)
        
        print('Acuracia = {0}'.format(mac))
        print("Matriz de confusao \n{0}".format(str(mcm)))
    
        del model_rf
    rf_accuracy.append(rf_tmp_accuracy)
    rf_matrix_confusion.append(rf_tmp_matrix_confusion)
    logger_spell.info('Accuracy RF \n{0}'.format(rf_tmp_accuracy))
    logger_spell.info('Matrix Confusion RF \n{0}'.format(rf_matrix_confusion))
    

print('Terminado o treinamento')