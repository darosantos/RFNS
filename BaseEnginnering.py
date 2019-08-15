class BaseEnginnering(object):
    
    __slots__ = ('train_X', 'train_y')
    
    def __init__(self):
        self.train_X = []
        self.train_y = []
        
    def __del__(self):
        del self.train_X
        del self.train_y

    def get_subset(self, columns: list) -> tuple:
        if type(columns) != list:
            raise TypeError('Expectd value list in columns')
        
        df_subset_x = self.train_X.loc[:, columns]
        df_subset_y = self.train_y.loc[df_subset_x.index]

        return (df_subset_x, df_subset_y)

    def arrangement_features(self, features: list,  n_selected: int) -> list:
        from itertools import combinations
        
        if type(n_selected) != int:
            raise TypeError('Expected value integer in n_selected')
            
        permsList = list(combinations(features, r=n_selected))

        return permsList