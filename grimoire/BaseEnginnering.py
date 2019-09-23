from grimoire.ConfigurationEnginnering import ConfigurationEnginnering

class BaseEnginnering(ConfigurationEnginnering):

    __slots__ = ('train_X', 'train_y', 'predict_X')

    def __init__(self):
        self.train_X = []
        self.train_y = []
        self.predict_X = []

        super().__init__()

    def __del__(self):
        del self.train_X
        del self.train_y
        del self.predict_X

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

# Implementar no método para que a transformação sejap progressiva
    def get_transform(self):
        if self.encoder_enable & self.encoder_data:
            self.run_encoder_data(1)
            #self.train_X = self.encoder_X.fit_transform(self.train_X)
            for col in self.train_X.columns:
                if type(self.train_X[col][0]) in self.encoder_not_type:
                    self.encoder_df.insert(loc=self.encoder_df.shape[1],
                                           column=col,
                                           value=self.train_X[col])
                else:
                    df_col = self.train_X.loc[:, [col]]
                    # reverse list of unique values
                    unique_categories = df_col[col].unique()[::-1]
                    self.encoder_feature[col] = unique_categories
                    df_tmp = self.encoder_X.fit_transform(df_col)
                    if (len(df_tmp.shape) == 1):
                        self.encoder_df.insert(loc=self.encoder_df.shape[1],
                                               column='{0}_all'.format(col), 
                                               value=df_tmp)
                    else:
                        index_shape = range(df_tmp.shape[1])
                        for i, c in zip(index_shape, unique_categories):
                            self.encoder_df.insert(loc=self.encoder_df.shape[1],
                                                   column='{0}_{1}'.format(col, c), 
                                                   value=df_tmp[:,i])
            del self.train_X
            self.train_X = self.encoder_df
            self.encoder_df = None

            
        if self.encoder_enable & self.encoder_target:
            self.run_encoder_target(0)
            self.train_y = self.encoder_y.fit_transform(self.train_y)

    def get_normalize(self):
        pass

# Adicionar na etapa de preprocessamento um código que verifica a integridade do dataset
    def get_preprocessing(self):
        pass
