class EnginneringForest(ClassifierEnginneringForest):

	def __init__(self, select_features):
		self.estimators_ = []
		self.select_features_ = select_features
		self.group_features_ = []
		self.df_predict_ = pd.DataFrame()
		self.n_features_ = 0
		self.n_samples_ = 0
		self.name_features_ = []
		super().__init__()

	def build(self, features_set):
		""" Cria um vetor com o número de árvores igual ao número de subconjuntos possíveis"""
		self.group_features_ = self.arrangement_features(features=features_set, n_selected=self.select_features_)
		self.estimators_ = [self.make_base_estimator() for gf in self.group_features_]

	def train(self, X, y, group_feature, estimator):
		subset_xdata, subset_ydata = self.get_subset(X, y, group_feature)
		return estimator.fit(subset_xdata, subset_ydata)

	def fit(self, X, y):
		self.n_samples_, self.n_features_ = X.shape
		self.name_features_ = X.columns

		# Cria a floresta
		self.build(features_set=self.name_features_)

		# Treina as arvores individualmente
		self.estimators_ = [self.train(X, y, subset_feature, estimator) 
							for subset_feature, estimator in zip(self.group_features_, self.estimators_)]

	def voting(self):
		final_predict = []
		for i in range(self.df_predict_.shape[0]):
			class_one = list(self.df_predict_.loc[i]).count(1)
			class_zero = list(self.df_predict_.loc[i]).count(0)
		if class_one > class_zero:
			final_predict.append(1)
		else:
			final_predict.append(0)
		return final_predict 

	def predict(self, X):
		# É aqui que monto a (matriz nº de amostras x nº de classificadores)
		for subset_feature, estimator in zip(self.group_features_, self.estimators_):
			subset_test = X.loc[:, subset_feature]
			num_columns = len(self.df_predict_.columns)
			pattern_name_column = 'cls_' + str(num_columns)
			cls_predict = estimator.predict(subset_test)
			self.df_predict_.insert(loc=num_columns, column=pattern_name_column, value=cls_predict)

		return self.voting()