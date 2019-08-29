print(">>>> Iniciando a execução do script")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from EnginneringForest import EnginneringForest
#from LoggerEnginnering import LoggerEnginnering

print(">> Carregando o dataset")

df_heart = pd.read_csv('heart.csv', engine='c')

X=df_heart[['age', 'sex', 'cp', 'trestbps',  'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
# Labels
y=df_heart['target']

# Split dataset into training set and test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100, shuffle=True, stratify=y)

print(">> Dataset carregado com sucesso")
print(">> Simula o código do modelo")

model_eg = EnginneringForest(select_features=2, reset_log=True)
model_eg.fit(X_train, y_train)
y_pred = model_eg.predict(X_test)

mac = accuracy_score(y_test, y_pred)
print('Acuracia = {0}'.format(mac))

mcm = confusion_matrix(y_test,y_pred)
print("Matriz de confusao \n{0}".format(str(mcm)))

print(">> Fim da simulacao")
print(">> Fim do script de teste")