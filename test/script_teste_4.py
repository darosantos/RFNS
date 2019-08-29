print(">>>> Iniciando a execução do script")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from EnginneringForest import EnginneringForest
#from LoggerEnginnering import LoggerEnginnering

print(">> Carregando o dataset")

columns_name = ['target', 'lepton_1_pT', 'lepton_1_eta', 'lepton_1_phi', 'lepton_2_pT', 'lepton_2_eta', 
        'lepton_2_phi', 'missing_energy_magnitude', 'missing_energy_phi', 'MET_rel', 'axial_MET',
        'M_R', 'M_TR_2', 'R', 'MT2', 'S_R', 'M_Delta_R', 'dPhi_r_b', 'cos_theta_r1']

# columns_dtype = {name: np.float64 for name in columns_name}

df_susy = pd.read_csv('SUSY.csv', 
                      names=[name.lower() for name in columns_name], 
                      engine='c', 
                      memory_map=True, 
                      low_memory=True)
                      
X=df_susy[[name.lower() for name in columns_name[1:]]]
y=df_susy['target']

# Split dataset into training set and test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, shuffle=True, stratify=y)

print(">> Dataset carregado com sucesso")
print(">> Simula o código do modelo")

model_eg = EnginneringForest(select_features=2)
model_eg.chunck = 128
# model_eg.fit(X_train, y_train)
y_pred = model_eg.predict(X_test)

print(">> Fim da simulacao")
print(">> Fim do script de teste")