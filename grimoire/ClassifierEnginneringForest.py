from grimoire.BaseEnginnering import BaseEnginnering


class ClassifierEnginneringForest(BaseEnginnering):

    __slots__ = ('criterion', 'splitter', 'max_depth', 'min_samples_split',
                 'min_samples_leaf', 'min_weight_fraction_leaf',
                 'max_features', 'random_state', 'max_leaf_nodes',
                 'class_weight', 'presort')

    def __init__(self):
        super().__init__()
        self.criterion = 'entropy'
        self.splitter = 'best'
        self.max_depth = None
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.min_weight_fraction_leaf = 0
        self.max_features = None
        self.random_state = 200
        self.max_leaf_nodes = None
        self.class_weight = None
        self.presort = False

    def __del__(self):
        del self.criterion
        del self.splitter
        del self.max_depth
        del self.min_samples_split
        del self.min_samples_leaf
        del self.min_weight_fraction_leaf
        del self.max_features
        del self.random_state
        del self.max_leaf_nodes
        del self.class_weight
        del self.presort

    def make_base_estimator(self):
        from sklearn.tree import DecisionTreeClassifier

        clf = DecisionTreeClassifier(self.criterion)
        return clf

    def make_lote_base_estimator(self, n_estimators):
        estimators_ = [self.make_base_estimator() for i in range(n_estimators)]
        estimators_ = self.get_pack_nparray(estimators_)
        return estimators_
