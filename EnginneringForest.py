from ClassifierEnginneringForest import ClassifierEnginneringForest
from LoggerEnginnering import LoggerEnginnering
from pandas import DataFrame, Series
import numpy as np
import time

class EnginneringForest(ClassifierEnginneringForest):
    
    __slots__ = ('estimators_', 'select_features_', 'group_features_', 
                 'df_predict_', 'n_features_', 'n_samples_', 'name_features_',
                 'prefix_column_predict', 'logger')
    
    def __init__(self, select_features: int):
        if type(select_features) != int:
            raise TypeError('Expectd value int in select_features')
        
        self.estimators_ = []
        self.select_features_ = select_features
        self.group_features_ = []
        self.df_predict_ = DataFrame()
        self.n_features_ = 0
        self.n_samples_ = 0
        self.name_features_ = []
        self.prefix_column_predict = 'cls'
        self.logger = LoggerEnginnering(name='enginnering', log_file='enginnering.log')
        super().__init__()
        
    def __del__(self):
        del self.estimators_
        del self.select_features_
        del self.group_features_
        del self.df_predict_
        del self.n_features_
        del self.n_samples_
        del self.name_features_
        del self.prefix_column_predict
        # del self.logger

    def build(self, features_set: list) -> None:
        """ Cria um vetor com o número de árvores igual ao número de 
            subconjuntos possíveis """
        self.group_features_ = self.get_arrangement_features(features_set,
                                                             self.select_features_)
        self.group_features_ = self.get_pack_nparray(self.group_features_)
        n_estimator = len(self.group_features_)
        self.estimators_ = self.make_lote_base_estimator(n_estimator)
        self.estimators_ = self.get_pack_nparray(self.estimators_)

    def train(self, group_feature: list, estimator):
        msg = 'Training subset = {0}, Timing = {1}'
        start_train = time.time()
        
        subset_xdata, subset_ydata = self.get_subset(group_feature)
        fit_ = estimator.fit(subset_xdata, subset_ydata)
        del subset_xdata
        del subset_ydata
        
        end_train = time.time()
        self.logger.add('debug',msg.format(group_feature, (end_train - start_train)))
        
        return fit_

    def fit(self, X, y) -> None:
        if not isinstance(X, DataFrame):
            raise TypeError('Expected value should descend from pandas.core.frame.DataFrame')
        if not isinstance(y, Series):
            raise TypeError('Expected value should descend from pandas.core.frame.DataFrame')
            
        self.n_samples_, self.n_features_ = X.shape
        self.name_features_ = X.columns
        
        self.train_X = X
        self.train_y = y
        
        # Cria a floresta
        self.build(features_set=self.name_features_)

        # Treina as arvores individualmente
        self.estimators_ = [self.train(subset_feature, estimator) 
                            for subset_feature, 
                                estimator in zip(self.group_features_, 
                                                 self.estimators_)]
        self.estimators_ = self.get_pack_nparray(self.estimators_)
        
        del self.train_X
        del self.train_y

    def voting(self) -> list:
        final_predict = []
        for i in range(self.df_predict_.shape[0]):
            class_one = list(self.df_predict_.loc[i]).count(1)
            class_zero = list(self.df_predict_.loc[i]).count(0)
            if class_one > class_zero:
                final_predict.append(1)
            else:
                final_predict.append(0)
        return final_predict 
        
    def predict_old(self, X) -> list:
        if not isinstance(X, DataFrame):
            raise TypeError('Expected value should descend from pandas.core.frame.DataFrame')
        # Este novo código se baseia em lidar com um volume muito grande para predição
        self.predict_X = X.reset_index()
        del X
        self.logger.add('debug','Size predict = {}'.format(self.predict_X.shape))
        
        # Determina se o dataset é muito grande para escolher qual estratégia usar
        # Percorre o dataset em pequenas linhas
        # A ideia e dividir o dataset em pequenas tabelas
        # cada tabela contém todos os atributos
        # depois cada árvoreé usada com a minitabela
        #self.chunck = 128
        num_columns = len(self.df_predict_.columns)
        pattern_name_column = "{0}{1}".format(self.prefix_column_predict, 
                                              num_columns)
            
        for x_, y_ in self.get_block_fit():
            self.logger.add('debug','Block Limit = ({}, {})'.format(x_, y_))
            
            dfsub = self.predict_X.loc[x_, y_]
            for subset_feature, estimator in zip(self.group_features_, self.estimators_):
                estimator.predict(block_instances)
        
    # Código antigo para predição
    def predict(self, X) -> list:
        if not isinstance(X, DataFrame):
            raise TypeError('Expected value should descend from pandas.core.frame.DataFrame')
        
        self.predict_X = X.reset_index()
        
        self.logger.add('debug','Size predict = {}'.format(self.predict_X.shape))
        
        # É aqui que monto a (matriz nº de amostras x nº de classificadores)
        for subset_feature, estimator in zip(self.group_features_, self.estimators_):
            # monta o nome da coluna do dataframe de predições
            num_columns = len(self.df_predict_.columns)
            pattern_name_column = "{0}{1}".format(self.prefix_column_predict, num_columns)
            
            self.logger.add('debug','>>> Predicting subset = {0}'.format(subset_feature))
            start_train = time.time()
            # Prepara para o treinamento com o subconjunto
            subset_test = self.predict_X.loc[:, subset_feature]
            cls_predict = []
            for item in self.get_block_fit():
                
                self.logger.add('debug','>>>> Block instances for subset = {0}'.format(item))
                
                block_instances = subset_test.loc[item[0]:item[1]]
                cls_predict.extend(estimator.predict(block_instances))
                
            
            end_train = time.time()
            self.logger.add('debug','>>>> Time predicting = {0}'.format((end_train - start_train)))
            
            # Adiciona o vetor de predições como uma coluna no dataframe de predições
            self.df_predict_.insert(loc=num_columns, 
                                    column=pattern_name_column, 
                                    value=cls_predict)
            del cls_predict
            del subset_feature
            del estimator

        del X
        del self.predict_X
        
        return self.voting()