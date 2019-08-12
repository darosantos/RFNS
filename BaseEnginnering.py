class BaseEnginnering:
	def get_subset(X, y, columns: list):
		from pandas import DataFrame
			
		if not isinstance(X, DataFrame):
			raise TypeError('Expected value should descend from pandas.core.frame.DataFrame')
		
		df_tmp = X.copy()
		df_tmp.insert(loc=(len(df_tmp.columns)), column='target', value=y)
		
		df_subset = (df_tmp[columns].copy(), df_tmp['target'].copy())
		
		return df_subset
		
	def arrangement_features(features: list,  n_selected: int) -> list:
		from itertools import combinations
		permsList = list(combinations(features, r=n_selected))
		return permsList