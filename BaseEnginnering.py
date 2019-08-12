class BaseEnginnering(object):
	def __init__(self):
		pass

	def get_subset(self, X, y, columns):
		from pandas import DataFrame

		#if not isinstance(self, X, DataFrame):
			#raise TypeError('Expected value should descend from pandas.core.frame.DataFrame')

		df_tmp = X.copy()
		df_tmp.insert(loc=(len(df_tmp.columns)), column='target', value=y)

		#df_subset = (df_tmp[df_tmp.columns in columns], df_tmp['target'])
		df_subset = (df_tmp.loc[:, columns], df_tmp['target'])

		return df_subset

	def arrangement_features(self, features: list,  n_selected: int) -> list:
		from itertools import combinations
		permsList = list(combinations(features, r=n_selected))

		return permsList