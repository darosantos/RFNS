class ClassifierEnginneringForest(BaseEnginnering):

	def __init__(self):
		self.criterion = 'entropy'
		self.splitter='best'
		self.max_depth=None
		self.min_samples_split=2
		self.min_samples_leaf=1
		self.min_weight_fraction_leaf=0
		self.max_features=None
		self.random_state = 200
		self.max_leaf_nodes=None
		self.class_weight=None
		self.presort=False

	def make_base_estimator(self):
		from sklearn.tree import DecisionTreeClassifier
		
		clf = DecisionTreeClassifier(self.criterion)
		return clf

	def make_lote_base_estimator(self):
		pass