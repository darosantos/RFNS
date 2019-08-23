class BaseEnginnering(object):
    
    __slots__ = ('train_X', 'train_y', 'predict_X', 'chunck')
    
    def __init__(self):
        self.train_X = []
        self.train_y = []
        self.predict_X = []
        self.chunck = 32
        
    def __del__(self):
        del self.train_X
        del self.train_y
        del self.predict_X
        del self.chunck
        
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
        n_per_block = ceil((1024 * self.chunck) / instance_sizeof)
        print('N instances = ', n_instances)
        print('N per block = ', n_per_block)
        
        if n_per_block >= n_instances:
            (yield (0,(n_instances-1)))
        else:
            # mount list blocks
            #pair_blocks = []
            #x = 0
            #for y in range(n_per_block, n_instances, n_per_block):
            #    pair_blocks.append((x, y))
            #    x = y+1
            pair_blocks = [((y - n_per_block), (y-1)) 
                           for y in range(n_per_block, 
                                          n_instances, 
                                          n_per_block)]
            
            print('Y = ', pair_blocks[-1])
            
            if (pair_blocks[-1][1] < n_instances):
                e = ((pair_blocks[-1][1] + 1), (n_instances-1))
                pair_blocks.append(e)
                
            for item in pair_blocks:
                (yield (item))

    #@deprecated
    def get_df_split(self):
        from sys import getsizeof
        from math import ceil
        
        # in bytes
        df_sizeof = getsizeof(self.predict_X)
        # number of instances
        n_instances = self.predict_X.shape[0]
        # size in bytes of instances
        instance_sizeof = df_sizeof / n_instances
        # number of intance per block
        n_blocks = ceil((1024 * self.chunck) / instance_sizeof)
        # mount list blocks
        pair_blocks = []
        x = 0
        for y in range(n_blocks, n_instances, n_blocks):
            pair_blocks.append((x, y))
            x = y+1
        # add diff
        if ( (n_instances % n_blocks) > 1 ):
            y = (x + (n_instances % n_blocks)) - 2
            pair_blocks.append((x, y))
        
        for item in pair_blocks:
            (yield item)
