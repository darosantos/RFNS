from grimoire.ClassifierEnginneringForest import ClassifierEnginneringForest
from grimoire.LoggerEnginnering import LoggerEnginnering
from pandas import DataFrame, Series
import numpy as np
import time

class EnginneringForest(ClassifierEnginneringForest):
    
    __slots__ = ('estimators_', 'select_features_', 'group_features_', 
                 'df_predict_', 'n_features_', 'n_samples_', 'name_features_',
                 'prefix_column_predict', 'logger', 'autoclean')
    
    def __init__(self, select_features: int, reset_log=False, name_log='enginnering.log'):
        if type(select_features) != int:
            raise TypeError('Expectd value int in select_features')
        
        self.estimators_ = []
        self.select_features_ = select_features
        self.group_features_ = []
        self.df_predict_ = []
        self.n_features_ = 0
        self.n_samples_ = 0
        self.name_features_ = []
        self.logger = LoggerEnginnering(name='enginnering', 
                                        log_file=name_log,
                                        drop_old=reset_log)
        self.autoclean = False
        super().__init__()
        
    def __del__(self):
        del self.estimators_
        del self.select_features_
        del self.group_features_
        del self.df_predict_
        del self.n_features_
        del self.n_samples_
        del self.name_features_
        del self.autoclean

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
        msg = 'Training subset = {0}, Timing = {1}, Size (Kb) = {2}'
        start_train = time.time()
        
        subset_xdata, subset_ydata = self.get_subset(group_feature)
        fit_ = estimator.fit(subset_xdata, subset_ydata)
        
        end_train = time.time()
        self.logger.add('debug',msg.format(group_feature, 
                                           (end_train - start_train),
                                           self.get_size_estimator(fit_)))
        
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
        
        self.build(features_set=self.name_features_)

        self.estimators_ = [self.train(subset_feature, estimator) 
                            for subset_feature, 
                                estimator in zip(self.group_features_, 
                                                 self.estimators_)]
        self.estimators_ = self.get_pack_nparray(self.estimators_)
        
        if self.autoclean:
            del self.train_X
            del self.train_y
    
    def predict(self, X) -> list:
        if not isinstance(X, DataFrame):
            raise TypeError('Expected value should descend from pandas.core.frame.DataFrame')
        
        self.predict_X = X
        
        self.logger.add('debug',
                        'Size predict = {0}, N estimators = {1}'.format(self.predict_X.shape, 
                                                                        len(self.estimators_)))
        
        for x_, y_ in self.get_block_fit():
            self.logger.add('debug','Block Limit = ({}, {})'.format(x_, y_))
            
            dfsub = self.predict_X.iloc[x_:y_]
            block_predict = []
            
            for subset_feature, estimator in zip(self.group_features_, self.estimators_):
                self.logger.add('debug', 'Subset predict = {0}'.format(subset_feature))
                subset_test = dfsub.loc[:, subset_feature]
                block_predict.append(estimator.predict(subset_test))
            
            block_predict = np.matrix(block_predict)
            
            self.logger.add('debug', "Shape One = {0}".format(block_predict.shape))
            
            block_predict = block_predict.T
            np.savetxt(fname='block_predict.log', X=block_predict,
                       fmt='%d', delimiter=',')
            self.logger.add('debug', "Shape Two = {0}".format(block_predict.shape))
            self.logger.add('debug', "Block predict \n{0}".format(block_predict))
            
            block_voting = self.voting(block_predict)
            self.logger.add('debug', "Block voting data \n{0}".format(str(block_voting)))
            self.logger.add('debug', "Block voting len {0}".format(len(block_voting)))
            self.df_predict_.extend(block_voting)
        
        return self.df_predict_
    
    def voting(self, data) -> list:
        final_predict = []
        for instance in data:
            cz = instance.tolist()[0].count(0)
            co = instance.tolist()[0].count(1)
            marjotiry = (co > cz) and 1 or 0
            final_predict.append(marjotiry)
        return final_predict