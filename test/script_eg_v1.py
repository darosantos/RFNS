print(">>>> Iniciando a execução do script")
print(">> Este script executa o codigo do EnginneringForest")
print(">> Iniciando os imports standards")

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


import logging

print(">> Fim dos imports standards")
print(">> Carregando o dataset")

columns_name = ['target', 'lepton_1_pT', 'lepton_1_eta', 'lepton_1_phi', 'lepton_2_pT', 'lepton_2_eta', 
		'lepton_2_phi', 'missing_energy_magnitude', 'missing_energy_phi', 'MET_rel', 'axial_MET',
		'M_R', 'M_TR_2', 'R', 'MT2', 'S_R', 'M_Delta_R', 'dPhi_r_b', 'cos_theta_r1']

df_susy = pd.read_csv('SUSY.csv', names=[name.lower() for name in columns_name], engine='c')

print(">> Dataset carregado com sucesso")
print(">> Imprimindo cabecalho do dataset")
print( df_susy.head() )
print(">> Tamanho do dataset")
print(df_susy.shape)
print(">> Informações do dataset")
print(df_susy.info())
print(">> Inicia a separacao dos dados de treino e teste")

X=df_susy[[name.lower() for name in columns_name[1:]]]
y=df_susy['target']

# Split dataset into training set and test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, shuffle=True, stratify=y)

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

class BaseEnginnering(object):
	def __init__(self):
		pass

	def get_subset(self, X, y, columns):

		df_tmp = X.copy()
		df_tmp.insert(loc=(len(df_tmp.columns)), column='target', value=y)

		df_subset = (df_tmp.loc[:, columns], df_tmp['target'])

		return df_subset

	def arrangement_features(self, features: list,  n_selected: int) -> list:
		from itertools import combinations
		permsList = list(combinations(features, r=n_selected))

		return permsList

print(">> Classe BaseEnginnering declarada com sucesso")
print(">> Declara a classe ClassifierEnginneringForest")

class ClassifierEnginneringForest(BaseEnginnering):

	def __init__(self):
		self.criterion = 'entropy'
		self.splitter='best'
		self.max_depth=None
		self.min_samples_split=2
		self.min_samples_leaf=1
		self.min_weight_fraction_leaf=0
		self.max_features=None
		self.random_state = 200
		self.max_leaf_nodes=None
		self.class_weight=None
		self.presort=False

	def make_base_estimator(self):
		clf = DecisionTreeClassifier(self.criterion)
		return clf

	def make_lote_base_estimator(self):
		pass


print(">> Classe ClassifierEnginneringForest declarada com sucesso")
print(">> Declara a classe EnginneringForest")

class EnginneringForest(ClassifierEnginneringForest):

    def __init__(self, select_features):
        self.estimators_ = []
        self.select_features_ = select_features
        self.group_features_ = []
        self.df_predict_ = pd.DataFrame()
        self.n_features_ = 0
        self.n_samples_ = 0
        self.name_features_ = []
        super().__init__()

    def build(self, features_set):
        """ Cria um vetor com o número de árvores igual ao número de subconjuntos possíveis"""
        self.group_features_ = self.arrangement_features(features=features_set, n_selected=self.select_features_)
        self.estimators_ = [self.make_base_estimator() for gf in self.group_features_]

    def train(self, X, y, group_feature, estimator):
        subset_xdata, subset_ydata = self.get_subset(X, y, group_feature)
        return estimator.fit(subset_xdata, subset_ydata)
        
    def fit(self, X, y):
        self.n_samples_, self.n_features_ = X.shape
        self.name_features_ = X.columns

        # Cria a floresta
        self.build(features_set=self.name_features_)

        # Treina as arvores individualmente
        self.estimators_ = [self.train(X, y, subset_feature, estimator) 
                            for subset_feature, estimator in zip(self.group_features_, self.estimators_)]

    def voting(self):
        final_predict = []
        for i in range(self.df_predict_.shape[0]):
            class_one = list(self.df_predict_.loc[i]).count(1)
            class_zero = list(self.df_predict_.loc[i]).count(0)
            if class_one > class_zero:
                final_predict.append(1)
            else:
                final_predict.append(0)
        return final_predict 
        
    def predict(self, X):
        # É aqui que monto a (matriz nº de amostras x nº de classificadores)
        for subset_feature, estimator in zip(self.group_features_, self.estimators_):
            subset_test = X.loc[:, subset_feature]
            num_columns = len(self.df_predict_.columns)
            pattern_name_column = 'cls_' + str(num_columns)
            cls_predict = estimator.predict(subset_test)
            self.df_predict_.insert(loc=num_columns, column=pattern_name_column, value=cls_predict)
            
        return self.voting()


print(">> Classe EnginneringForest declarada com sucesso")
print(">> Cria o ambiente de log para salvar os dados de acuracia e matriz de confusao")

reset_logger('logger_accuracy_dataset_susy.log')
reset_logger('logger_matrix_confusion_dataset_susy.log')
logger_accuracy_eg = setup_logger('accuracy_eg', 'logger_accuracy_dataset_susy.log')
logger_matrix_confusion_eg = setup_logger('matrix_confusion_eg', 'logger_matrix_confusion_dataset_susy.log')

print(">> Ambiente dos logs criados com sucesso")
print(">> Inicia o treinamento de cada conjunto de arvores")

for n_tree in range(df_susy.shape[1]-1):
    print('>>> Iniciando execucao - ', n_tree)
    print('>> Cria o modelo')
    model_eg = EnginneringForest(select_features=n_tree+1)
    print('>> Treina o modelo')
    model_eg.fit(X_train, y_train)
    print('>> Testa p modelo')
    y_pred = model_eg.predict(X_test)
    print('>> Calcula a acuracia')
    mac = accuracy_score(y_test, y_pred)
    logger_accuracy_eg.info(str(mac))
    print('>> Calcula a matriz de confusao')
    mcm = confusion_matrix(y_test,y_pred)
    logger_matrix_confusion_eg.info(str(mcm))
    print(">> N. de atributos %i, Acuracia = %d, N. de Arvores = %i".format((n_tree+1), mac, len(model_eg.estimators_)))
    print('>>> Fim da execucao')

print("Fim do treinamento de cada conjunto de arvores")
print(">>>> Fim do script")



