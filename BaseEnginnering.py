class BaseEnginnering(object):
    
    __slots__ = ('train_X', 'train_y')
    
    def __init__(self):
        self.train_X = []
        self.train_y = []
        
    def __del__(self):
        del self.train_X
        del self.train_y
        
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
    
    def get_arrangement_features(self, features: list,  n_selected: int) -> list:
        from itertools import combinations
        
        if type(n_selected) != int:
            raise TypeError('Expected value integer in n_selected')
            
        permsList = list(combinations(features, r=n_selected))
        
        return permsList
    
    def get_pack_nparray(self, elements: list):
        import numpy as np
        return np.array(elements, np.object)
    
    def get_df_split(self, df, chunck=1):
        # pagina de memória tamanho padrão é 4 kb
        # pegar o tamanho do dataset
        # dividir o tamanho do dataset pelo número de registros
        # multiplicar o número de registro por 4k para descborir quantos precisa
        # com o número de registros implementar um interador
        # retornar bloco a bloco com loc do dataframe
        #chunck
        from sys import getsizeof
        from math import ceil
        
        #df_sizeof = int((getsizeof(df) / 1024) + 1) # in kb , arredonda para cima
        #df_instances = df.shape[0]
        #n_blocks = int(df_instances / df_sizeof)
        #pair_blocks = [(x, y+n_blocks) for x in range(n_interators) for y in range(n_interators)]
        
        # in bytes
        df_sizeof = getsizeof(df)
        # number of instances
        n_instances = df.shape[0]
        # size in bytes of instances
        instance_sizeof = df_sizeof / n_instances
        # number of intance per block
        n_blocks = ceil((1024 * chunck) / instance_sizeof)
        # mount list blocks
        pair_blocks = []
        x = 0
        for y in range(n_blocks, n_instances, n_blocks):
            pair_blocks.append((x, y))
            x = y+1
        # add diff
        if ( (n_instances % n_blocks) > 0 ):
            y = (x + (n_instances % n_blocks)) - 2
            pair_blocks.append((x, y))
        
        return pair_blocks
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


