class EnginneringForest(ClassifierEnginneringForest):

	def __init__(self, select_features):
		# Global variables
		self.estimators_ = []
		self.select_features_ = select_features
		self.group_features = []
		self.vector_predict = []
		super().__init__()

	def build(self, features_set):
		# Aqui vai o código para criar a floresta
		# cria o vetor dos subconjuntos
		# cria as instâncias das arvores
		self.group_features = self.arrangement_features(features=features_set, n_selected=self.select_features_)
		#self.estimators_ = [self.make_base_estimator() for item_set in self.group_features]
		for i in self.group_features:
			self.estimators_.append(self.make_base_estimator())

	def voting(self):
		final_predict = []
		for i in range(len(self.vector_predict[0])):
			column_predict = []
		for j in range(len(self.estimators_)):
			column_predict.append(self.vector_predict[j][i])
		if column_predict.count(1) > column_predict.count(0):
			final_predict.append(1)
		else:
			final_predict.append(0)
		return final_predict


	def fit(self, X, y):
		# Determine output settings
		n_samples, self.n_features_ = X.shape
		name_features = X.columns

		# Cria a floresta
		self.build(features_set=name_features)

		# Treina as arvores individualmente
		for subset_feature, estimators in zip(self.group_features, self.estimators_):
			subset_xdata, subset_ydata = self.get_subset(X, y, subset_feature)
			estimators = estimators.fit(subset_xdata, subset_ydata)

		# Treina a floresta
		#for item_set, clf in zip(self.group_features, self.estimators_):
		#	subset_xdata, subset_ydata = self.get_subset(X, y, item_set)
		#	clf = clf.fit(subset_xdata, subset_ydata)
		#for subset_feature in subsets_features:
		#	subset_xdata, subset_ydata = self.get_subset(X, y, subset_feature)
		#	clf = self.build()
		#	clf = clf.fit(subset_xdata, subset_ydata)
		#	self.estimators_.append(clf)

	def predict(self, X):
		for subset_feature, estimators in zip(self.group_features, self.estimators_):
			subset_test = X.loc[:, subset_feature]
			self.vector_predict.append(estimators.predict(subset_test))
		return self.voting()