# -*- coding: utf-8 -*-
from grimoire.ClassifierEnginneringForest import ClassifierEnginneringForest

from pandas import DataFrame, Series
from numpy import matrix
import time


class EnginneringForest(ClassifierEnginneringForest):

    __slots__ = ('estimators_', 'select_features_', 'group_features_',
                 'vector_predict_', 'n_features_', 'n_samples_',
                 'name_features_', 'classes_', 'estrategy_trainning',
                 'is_data_categorical')

    # Const values - don't modify
    ESTRATEGY_TRAINNING_SINGLE = 0
    ESTRATEGY_TRAINNING_BLOCK = 1

    def __init__(self, select_features: int):
        if type(select_features) != int:
            raise TypeError('Expectd value int in select_features')
        super().__init__()
        self.estimators_ = []
        self.select_features_ = select_features
        self.group_features_ = []
        self.vector_predict_ = []
        self.n_features_ = 0
        self.n_samples_ = 0
        self.name_features_ = []
        self.estrategy_trainning = self.ESTRATEGY_TRAINNING_SINGLE
        self.is_data_categorical = False

    def __del__(self):
        del self.estimators_
        del self.select_features_
        del self.group_features_
        del self.vector_predict_
        del self.n_features_
        del self.n_samples_
        del self.name_features_
        del self.estrategy_trainning

    def build(self, features_set: list) -> None:
        """ Cria um vetor com o número de árvores igual ao número de
            subconjuntos possíveis """
        sf = self.select_features_
        self.group_features_ = self.get_arrangement_features(features_set, sf)
        self.group_features_ = self.get_pack_nparray(self.group_features_)
        n_estimator = len(self.group_features_)
        self.estimators_ = self.make_lote_base_estimator(n_estimator)
        self.estimators_ = self.get_pack_nparray(self.estimators_)

    def train(self, group_feature: list, estimator):
        msg = 'Training subset = {0}, Timing = {1}, Size (Kb) = {2}'
        start_train = time.time()

        features_list = self.get_in_list(group_feature)
        subset_xdata, subset_ydata = self.get_subset(features_list)
        fit_ = estimator.fit(subset_xdata, subset_ydata)

        end_train = time.time()
        self.logger.add('debug', msg.format(group_feature,
                                            (end_train - start_train),
                                            self.get_size_estimator(fit_)))

        return fit_

    def fit(self, X, y) -> None:
        if not isinstance(X, DataFrame):
            msg = 'Expected value should descend from Pandas DataFrame'
            raise TypeError(msg)
        if not isinstance(y, Series):
            msg = 'Expected value should descend from Pandas Series'
            raise TypeError(msg)

        self.train_X = X
        self.train_y = y
        # Define os parâmetros de acordo com a estratégia de treinamento
        # Somente em caso de dados categóricos presentes
        if self.is_data_categorical is False:
            self.n_samples_, self.n_features_ = X.shape
            self.name_features_ = X.columns
        else:
            # Normaliza e transforma os dados
            self.get_transform()
            self.get_normalize()
            # Prepara o número de amostra de acordo com a estratégia
            mode_train = self.estrategy_trainning
            if mode_train == self.ESTRATEGY_TRAINNING_SINGLE:
                self.n_samples_, self.n_features_ = self.train_X.shape
                self.name_features_ = self.train_X.columns
            elif mode_train == self.ESTRATEGY_TRAINNING_BLOCK:
                self.n_samples_ = self.train_X.shape[0]
                for key_ef in self.encoder_feature:
                    if type(self.encoder_feature[key_ef]) is list:
                        block = ['{0}_{1}'.format(key_ef, value)
                                 for value in self.encoder_feature[key_ef]]
                        self.name_features_.append(tuple(block))
                    else:
                        self.name_features_.append(key_ef)
            else:
                raise TypeError('Expected estrategy trainning value')

        self.get_classes()

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
            msg = 'Expected value should descend from Pandas DataFrame'
            raise TypeError(msg)

        self.predict_X = X

        # Codifica o X_test
        if self.is_data_categorical is True:
            self.get_transform_predict()
            self.get_normalize_predict()

        msg = 'Size predict = {0}, N estimators = {1}'
        self.logger.add('debug',
                        msg.format(self.predict_X.shape,
                                   len(self.estimators_)))

        for x_, y_ in self.get_block_fit():
            self.logger.add('debug', 'Block Limit = ({}, {})'.format(x_, y_))

            dfsub = self.predict_X.iloc[x_:y_]
            block_predict = []

            for subset_feature, estimator in zip(self.group_features_,
                                                 self.estimators_):
                msg = 'Subset predict = {0}'
                self.logger.add('debug', msg.format(subset_feature))
                subset_feature = self.get_in_list(subset_feature)
                # Aqui deve ser inserido a mostragem nos dados
                # Por enquanto está usando o conjunto todo
                subset_test = dfsub.loc[:, subset_feature]
                block_predict.append(estimator.predict(subset_test))

            block_predict = matrix(block_predict)
            self.run_save_predict(block_predict)

            msg = "Shape One = {0}"
            self.logger.add('debug', msg.format(block_predict.shape))

            block_predict = block_predict.T
            msg = "Shape Two = {0}"
            self.logger.add('debug', msg.format(block_predict.shape))
            msg = "Block predict \n{0}"
            self.logger.add('debug', msg.format(block_predict))

            block_voting = self.voting(block_predict)
            msg = "Block voting data \n{0}"
            self.logger.add('debug', msg.format(str(block_voting)))
            msg = "Block voting len {0}"
            self.logger.add('debug', msg.format(len(block_voting)))
            self.vector_predict_.extend(block_voting)

        return self.vector_predict_

    def voting(self, data) -> list:
        final_predict = []
        for instance in data:
            cz = instance.tolist()[0].count(0)
            co = instance.tolist()[0].count(1)
            marjotiry = (co > cz) and 1 or 0
            final_predict.append(marjotiry)
        return final_predict
