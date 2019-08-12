class EnginneringForest(BaseEnginnering):
	
	def __init__(self, select_features):
		# Global variables
		self.estimators_ = []
		self.select_features_ = select_features
		
	
	def build(self, X, y):
		pass
		
		
	def predict(self, X):
		pass
		
	def fit(self, X, y):
		# local variables
		features = X.columns
		subsets_features = self.arrangement_features(features, self.select_features_)
		
		for subset_feature in subsets_features:
			subset_xdata, subset_ydata = self.get_subset(X, y, subset_feature)
			clf = DecisionTreeClassifier(criterion = 'entropy',
										 splitter='best',
										 max_depth=None,
										 min_samples_split=2,
										 min_samples_leaf=1,
										 min_weight_fraction_leaf=0,
										 max_features=None,
										 random_state = 200,
										 max_leaf_nodes=None,
										 #min_impurity_decrease=0,
										 #min_impurity_split=1e-7,
										 class_weight=None,
										 presort=False)
			clf = clf.fit(subset_xdata, subset_ydata)
			self.estimators_.append(clf)