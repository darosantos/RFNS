import pandas as pd

import numpy as np

import logging

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import seaborn as sns

import logging

from EnginneringForest import EnginneringForest

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(name, log_file, level=logging.INFO):
	"""Function setup as many loggers as you want"""

	handler = logging.FileHandler(log_file)
	handler.setFormatter(formatter)

	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)

	return logger

def reset_logger(name):
	from os import remove
	from os.path import isfile
	if isfile(name):
		remove(name)

df_heart = pd.read_csv('heart.csv', engine='c')

X=df_heart[['age', 'sex', 'cp', 'trestbps',  'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
# Labels
y=df_heart['target']

# Split dataset into training set and test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, shuffle=True, stratify=y)

logger_accuracy_eg = setup_logger('accuracy_eg', 'logger_accuracy_eg.log')
logger_matrix_confusion_eg = setup_logger('matrix_confusion_eg', 'logger_matrix_confusion_eg.log')

for n_tree in [1,2,3,4,5,6,7]:
	model_eg = EnginneringForest(select_features=n_tree+1)
	model_eg.fit(X_train, y_train)
	y_pred = model_eg.predict(X_test)

	mac = metrics.accuracy_score(y_test, y_pred)
	logger_accuracy_eg.info(str(mac))

	mcm = confusion_matrix(y_test,y_pred)
	logger_matrix_confusion_eg.info(str(mcm))

	del model_eg