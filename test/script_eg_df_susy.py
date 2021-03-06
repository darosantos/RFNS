print(">>>> Iniciando a execução do script")
print(">> Este script executa o codigo do EnginneringForest")
print(">> Iniciando os imports standards")

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
#from sklearn.preprocessing import RobustScaler

import logging

from EnginneringForest import EnginneringForest

print(">> Fim dos imports standards")
print(">> Carregando o dataset")

columns_name = ['target', 'lepton_1_pT', 'lepton_1_eta', 'lepton_1_phi', 'lepton_2_pT', 'lepton_2_eta', 
        'lepton_2_phi', 'missing_energy_magnitude', 'missing_energy_phi', 'MET_rel', 'axial_MET',
        'M_R', 'M_TR_2', 'R', 'MT2', 'S_R', 'M_Delta_R', 'dPhi_r_b', 'cos_theta_r1']

columns_dtype = {name: np.float64 for name in columns_name}

df_susy = pd.read_csv('SUSY.csv', 
                      names=[name.lower() for name in columns_name], 
                      engine='c', 
                      memory_map=True, 
                      low_memory=True)

print(">> Dataset carregado com sucesso")
print(">> Imprimindo cabecalho do dataset")
print( df_susy.head() )
print(">> Tamanho do dataset")
print(df_susy.shape)
print(">> Informações do dataset")
print(df_susy.info())

#print(">> Colcoa os dados em escala")
#transformer = RobustScaler(copy=True, quantile_range=(30.0, 70.0), 
#                           with_centering=True, with_scaling=True)
#df_susy = transformer.fit_transform(df_susy)
#print(">> Dados em escala")


print(">> Inicia a separacao dos dados de treino e teste")

X=df_susy[[name.lower() for name in columns_name[1:]]]
y=df_susy['target']

# Split dataset into training set and test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, shuffle=True, stratify=y)
del df_susy

print(">> Fim da separação dos dados")
print(">> Dimensoes de treino")
print(X_train.shape)
print(">> Dimensões de teste")
print(X_test.shape)
print(">> Cria o ambiente de logging")

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
    
print(">> Ambiente de log criado com sucesso")
print(">> Declara a classe BaseEnginnering")
print(">> Cria o ambiente de log para salvar os dados de acuracia e matriz de confusao")

reset_logger('logger_accuracy_dataset_susy2.log')
reset_logger('logger_matrix_confusion_dataset_susy2.log')
logger_accuracy_eg = setup_logger('accuracy_eg', 'logger_accuracy_dataset_susy2.log')
logger_matrix_confusion_eg = setup_logger('matrix_confusion_eg', 'logger_matrix_confusion_dataset_susy2.log')

print(">> Ambiente dos logs criados com sucesso")
print(">> Inicia o treinamento de cada conjunto de arvores")

for n_tree in range(18):    
    print('>>> Iniciando execucao - ', n_tree)
    print('>> Cria o modelo')
    model_eg = EnginneringForest(select_features=n_tree+1, name_log='test_susy_nf_{0}'.format(n_tree+1))
    model_eg.chunck = 128
    print('>> Treina o modelo')
    model_eg.fit(X_train, y_train)
    # model_eg.chunck = 32
    print('>> Testa o modelo')
    y_pred = model_eg.predict(X_test)

    mac = accuracy_score(y_test, y_pred)
    logger_accuracy_eg.info(str(mac))

    mcm = confusion_matrix(y_test,y_pred)
    logger_matrix_confusion_eg.info(str(mcm))
    
    print('Acuracia = {0}'.format(mac))
    print("Matriz de confusao \n{0}".format(str(mcm)))

    del model_eg

print("Fim do treinamento de cada conjunto de arvores")
print(">>>> Fim do script")



