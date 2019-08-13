class EnginneringForest(ClassifierEnginneringForest):

	__slots__ = [estimators_, select_features_, group_features_, df_predict_, 
				 n_features_, n_samples_, n_samples_, name_features_]
	
	def __init__(self, select_features: int):
		if type(select_features) != int:
			raise TypeError('Expectd value int in select_features')
			
		self.estimators_ = []
		self.select_features_ = select_features
		self.group_features_ = []
		self.df_predict_ = pd.DataFrame()
		self.n_features_ = 0
		self.n_samples_ = 0
		self.name_features_ = []
		self.df_prefix_column_predict = 'cls'
		super().__init__()
		
	def __del__(self):
		del self.estimators_
		del self.select_features_
		del self.group_features_
		del self.df_predict_
		del self.n_features_
		del self.n_samples_
		del self.name_features_

	def build(self, features_set: list) -> None:
		""" Cria um vetor com o número de árvores igual ao número de subconjuntos possíveis"""
		self.group_features_ = self.arrangement_features(features=features_set, n_selected=self.select_features_)
		self.estimators_ = (self.make_base_estimator() for gf in self.group_features_)

	def train(self, X, y, group_feature: list, estimator: list):
		subset_xdata, subset_ydata = self.get_subset(X, y, group_feature)
		return estimator.fit(subset_xdata, subset_ydata)

	def fit(self, X, y) -> None:
		if not isinstance(X, DataFrame):
			raise TypeError('Expected value should descend from pandas.core.frame.DataFrame')
		if not isinstance(y, Series):
			raise TypeError('Expected value should descend from pandas.core.frame.DataFrame')
			
		self.n_samples_, self.n_features_ = X.shape
		self.name_features_ = X.columns

		# Cria a floresta
		self.build(features_set=self.name_features_)

		# Treina as arvores individualmente
		self.estimators_ = (self.train(X, y, subset_feature, estimator) 
							for subset_feature, estimator in zip(self.group_features_, self.estimators_))

	def voting(self) -> list:
		final_predict = []
		for i in range(self.df_predict_.shape[0]):
			class_one = list(self.df_predict_.loc[i]).count(1)
			class_zero = list(self.df_predict_.loc[i]).count(0)
		if class_one > class_zero:
			final_predict.append(1)
		else:
			final_predict.append(0)
		return final_predict 

	def classifier(self, group_feature: list, estimator):
			subset_test = X.loc[:, group_feature]
			num_columns = len(self.df_predict_.columns)
			pattern_name_column = "{1}{2}".format(self.df_prefix_column_predict, num_columns)
			cls_predict = estimator.predict(subset_test)
			self.df_predict_.insert(loc=num_columns, column=pattern_name_column, value=cls_predict)
			
	def predict(self, X) -> list:
		if not isinstance(X, DataFrame):
			raise TypeError('Expected value should descend from pandas.core.frame.DataFrame')
			
		# É aqui que monto a (matriz nº de amostras x nº de classificadores)
		(self.classifier(group_feature=subset_feature,estimator=estimator) 
		 for subset_feature, estimator in zip(self.group_features_, self.estimators_))
		#for subset_feature, estimator in zip(self.group_features_, self.estimators_):
		#	subset_test = X.loc[:, subset_feature]
		#	num_columns = len(self.df_predict_.columns)
		#	pattern_name_column = 'cls_' + str(num_columns)
		#	cls_predict = estimator.predict(subset_test)
		#	self.df_predict_.insert(loc=num_columns, column=pattern_name_column, value=cls_predict)

		return self.voting()