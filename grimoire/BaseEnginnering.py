# -*- coding: utf-8 -*-
from grimoire.ConfigurationEnginnering import ConfigurationEnginnering

from pandas import DataFrame, Series


class BaseEnginnering(ConfigurationEnginnering):

    __slots__ = ('train_X', 'train_y', 'predict_X', 'classes_')

    def __init__(self):
        super().__init__()
        self.train_X = []
        self.train_y = []
        self.predict_X = []
        self.classes_ = {}

    def __del__(self):
        del self.train_X
        del self.train_y
        del self.predict_X
        del self.classes_

    def get_subset(self, columns) -> tuple:
        df_subset_x = self.train_X.loc[:, columns]
        df_subset_y = self.train_y.loc[df_subset_x.index]

        return (df_subset_x, df_subset_y)

    def get_param_value(self, param_name: str):
        if param_name in self.__slots__:
            return self.__slots__[param_name]
        elif param_name in self.__dict__:
            return self.__dict__[param_name]
        else:
            raise TypeError('Access property is invalid')

    def get_arrangement_features(self, features, n_selected):
        from itertools import combinations

        if type(n_selected) != int:
            raise TypeError('Expected value integer in n_selected')

        permsList = list(combinations(features, r=n_selected))

        return permsList

    def get_pack_nparray(self, elements: list):
        import numpy as np
        return np.array(elements, np.object)

    def get_size_estimator(self, estimator):
        from sys import getsizeof

        return (getsizeof(estimator) / 1024)

    def get_block_fit(self):
        from sys import getsizeof
        from math import ceil

        # in bytes
        df_sizeof = getsizeof(self.predict_X)
        # number of instances
        n_instances = self.predict_X.shape[0]
        # size in bytes of instances
        instance_sizeof = df_sizeof / n_instances
        # number of instance per block
        n_per_block = ceil((1024 * 4 * self.chunck) / instance_sizeof)

        if n_per_block >= n_instances:
            (yield (0, n_instances))
        else:
            pair_blocks = [((y - n_per_block), (y-1))
                           for y in range(n_per_block,
                                          n_instances, n_per_block)]

            if (pair_blocks[-1][1] < n_instances):
                e = ((pair_blocks[-1][1] + 1), (n_instances-1))
                pair_blocks.append(e)

            for item in pair_blocks:
                (yield (item))

    def get_transform(self, data_encoder_type=1, target_encoder_type=0):
        condition = [self.encoder_enable,
                     self.encoder_data,
                     (self.encoder_flag[0] == 0)]
        if all(condition):
            self.run_encoder_data(data_encoder_type)
            encoder_df = DataFrame(index=self.train_X.index)
            for col in self.train_X.columns:
                index_first_element = self.train_X.index[0]
                first_element = self.train_X[col][index_first_element]
                if type(first_element) in self.encoder_not_type:
                    encoder_df.insert(loc=encoder_df.shape[1],
                                      column=col,
                                      value=self.train_X[col])
                    self.encoder_feature[col] = type(first_element)
                else:
                    df_col = self.train_X.loc[:, [col]]
                    # reverse list of unique values
                    # convertendo para um list para facilitar nos
                    # próximos procedimentos
                    unique_categories = list(df_col[col].unique()[::-1])
                    self.encoder_feature[col] = unique_categories
                    df_tmp = self.encoder_X.fit_transform(df_col)
                    if (len(df_tmp.shape) == 1):
                        col_name = '{0}_all'.format(col)
                        encoder_df.insert(loc=encoder_df.shape[1],
                                          column=col_name,
                                          value=df_tmp)
                        self.encoder_categorical_columns.append(col_name)
                    else:
                        index_shape = range(df_tmp.shape[1])
                        for i, c in zip(index_shape, unique_categories):
                            col_name = '{0}_{1}'.format(col, c)
                            encoder_df.insert(loc=encoder_df.shape[1],
                                              column=col_name,
                                              value=df_tmp[:, i])
                            self.encoder_categorical_columns.append(col_name)
            del self.train_X
            self.train_X = encoder_df.copy()
            del encoder_df
            self.encoder_flag[0] = 1

        condition = [self.encoder_enable,
                     self.encoder_target,
                     (self.encoder_flag[1] == 0)]
        if all(condition):
            self.run_encoder_target(target_encoder_type)
            encoder_index = self.train_y.index
            encoder_values = self.encoder_y.fit_transform(self.train_y)
            self.train_y = Series(data=encoder_values, index=encoder_index)
            self.encoder_flag[1] = 1

    def get_normalize(self, scaler_type=0):
        if self.normalize_enable & (self.normalize_flag == 0):
            column_numerical = [col for col in self.train_X
                                if col not in self.encoder_categorical_columns]
            df_tmp = self.train_X.loc[:, column_numerical]
            self.run_scaler_data(scaler_type)
            normal_values = self.normalize_scaler.fit_transform(df_tmp)
            index_shape = range(normal_values.shape[1])
            for i, c in zip(index_shape, column_numerical):
                self.train_X[c] = normal_values[:, i]
            self.normalize_flag = 1

    def get_transform_predict(self):
        condition = [self.encoder_enable,
                     self.encoder_data,
                     (self.encoder_flag[0] == 1)]
        if all(condition):
            encoder_df = DataFrame(index=self.predict_X.index)
            for col in self.predict_X.columns:
                index_first_element = self.predict_X.index[0]
                first_element = self.predict_X[col][index_first_element]
                if type(first_element) in self.encoder_not_type:
                    encoder_df.insert(loc=encoder_df.shape[1],
                                      column=col,
                                      value=self.predict_X[col])
                else:
                    df_col = self.predict_X.loc[:, [col]]
                    unique_categories = list(df_col[col].unique())
                    # reverse list of unique values
                    # convertendo para um list para facilitar nos
                    # próximos procedimentos
                    df_tmp = self.encoder_X.transform(df_col)
                    if (len(df_tmp.shape) == 1):
                        col_name = '{0}_all'.format(col)
                        encoder_df.insert(loc=encoder_df.shape[1],
                                          column=col_name,
                                          value=df_tmp)
                    else:
                        index_shape = range(df_tmp.shape[1])
                        for i, c in zip(index_shape, unique_categories):
                            col_name = '{0}_{1}'.format(col, c)
                            encoder_df.insert(loc=encoder_df.shape[1],
                                              column=col_name,
                                              value=df_tmp[:, i])
            del self.predict_X
            self.predict_X = encoder_df.copy()
            del encoder_df

    def get_normalize_predict(self):
        if self.normalize_enable & (self.normalize_flag == 1):
            column_numerical = [col for col in self.predict_X
                                if col not in self.encoder_categorical_columns]
            df_tmp = self.predict_X.loc[:, column_numerical]
            normal_values = self.normalize_scaler.transform(df_tmp)
            index_shape = range(normal_values.shape[1])
            for i, c in zip(index_shape, column_numerical):
                self.predict_X[c] = normal_values[:, i]

    def get_preprocessing(self, data_encoder_type=1,
                          target_encoder_type=0, scaler_type=0):
        if self.preprocessing_enable:
            self.encoder_enable = True
            self.encoder_data = True
            self.encoder_target = True
            self.normalize_enable = True
            self.run_preprocessing(data_encoder_type, target_encoder_type,
                                   scaler_type)
            self.get_transform()
            self.get_normalize()

    def get_in_list(self, l):
        my_result = []
        for i in l:
            if type(i) is tuple:
                my_result.extend(i)
            else:
                my_result.append(i)
        return my_result

    def get_classes(self):
        condition = [self.encoder_enable,
                     self.encoder_target,
                     (self.encoder_flag[1] == 1)]
        if all(condition):
            for i, c in enumerate(self.encoder_y.classes_):
                self.classes_[i] = c
        else:
            for i, c in enumerate(set(self.train_y)):
                self.classes_[i] = c
