class BaseEnginnering(object):
	def __init__(self):
		pass

	def get_subset(self, X, y, columns: list) -> tuple:
		from pandas import DataFrame

		if not isinstance(X, DataFrame):
			raise TypeError('Expected value should descend from pandas.core.frame.DataFrame')
		if not isinstance(y, DataFrame):
			raise TypeError('Expected value should descend from pandas.core.frame.DataFrame')
		if type(columns) != list:
			raise TypeError('Expectd value list in columns')
			
		df_subset = (X.loc[:, columns], y.loc[:,:])

		return df_subset

	def arrangement_features(self, features: list,  n_selected: int) -> list:
		from itertools import combinations
		
		if type(n_selected) != int:
			raise TypeError('Expected value integer in n_selected')
			
		permsList = list(combinations(features, r=n_selected))

		return permsList