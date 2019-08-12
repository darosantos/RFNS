class EnginneringForest(ClassifierEnginneringForest):

	def __init__(self, select_features):
		self.estimators_ = []
		self.select_features_ = select_features
		self.group_features = []
		self.vector_predict = []
		super().__init__()

	def build(self, features_set):
		if (type(self.select_features_) == tuple):
			self.group_features = [[],[]]
			self.group_features[0] = self.arrangement_features(features=features_set, n_selected=self.select_features_[0])
			self.group_features[1] = self.arrangement_features(features=features_set, n_selected=self.select_features_[1])

		self.estimators_ = [[],[]]
		for i in self.group_features[0]:
			self.estimators_[0].append(self.make_base_estimator())

		for i in self.group_features[1]:
			self.estimators_[1].append(self.make_base_estimator())
		else:
			self.group_features = self.arrangement_features(features=features_set, n_selected=self.select_features_)

		for i in self.group_features:
			self.estimators_.append(self.make_base_estimator())


	def voting(self):
		final_predict = []
		if (type(self.select_features_) == tuple):
			for i in range(len(self.vector_predict[0][0])):
				column_predict = []
			for j in range(len(self.estimators_[0])):
				column_predict.append(self.vector_predict[0][j][i])
			for j in range(len(self.estimators_[1])):
				column_predict.append(self.vector_predict[1][j][i])
			if column_predict.count(1) > column_predict.count(0):
				final_predict.append(1)
			else:
				final_predict.append(0)
		else:
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
		n_samples, self.n_features_ = X.shape
		name_features = X.columns

		# Cria a floresta
		self.build(features_set=name_features)

		# Treina as arvores individualmente
		if (type(self.select_features_) == tuple):
			for group_features_, estimators_ in zip(self.group_features, self.estimators_):
				for subset_feature, estimators in zip(group_features_, estimators_):
					subset_xdata, subset_ydata = self.get_subset(X, y, subset_feature)
					estimators = estimators.fit(subset_xdata, subset_ydata)
		else:
			for subset_feature, estimators in zip(self.group_features, self.estimators_):
				subset_xdata, subset_ydata = self.get_subset(X, y, subset_feature)
				estimators = estimators.fit(subset_xdata, subset_ydata)

	def predict(self, X):
		if (type(self.select_features_) == tuple):
			self.vector_predict = [[],[]]
			for subset_feature, estimators in zip(self.group_features[0], self.estimators_[0]):
				subset_test = X.loc[:, subset_feature]
				self.vector_predict[0].append(estimators.predict(subset_test))
			for subset_feature, estimators in zip(self.group_features[1], self.estimators_[1]):
				subset_test = X.loc[:, subset_feature]
				self.vector_predict[1].append(estimators.predict(subset_test))
		else:
			for subset_feature, estimators in zip(self.group_features, self.estimators_):
				subset_test = X.loc[:, subset_feature]
				self.vector_predict.append(estimators.predict(subset_test))

	return self.voting()