import os
import pandas as pd
import numpy as np
import skops.io as sio
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Carregar as fonte de dados
x_train = pd.read_csv("./data/birds_x_train.csv", header=None)
y_train = np.ravel(pd.read_csv("./data/birds_y_train.csv", header=None))
x_test = pd.read_csv("./data/birds_x_test.csv", header=None)
y_test = np.ravel(pd.read_csv("./data/birds_y_test.csv", header=None))

# Implementa a multi-layer perceptron (MLP) com o algoritmo para treinamento com Backpropagation
clf = MLPClassifier(solver='lbfgs', 
                    alpha=1e-5,
                    hidden_layer_sizes=(250,),
                    random_state=84)

# Treinar a rede neural para ser capaz de realizar a predição
print('Iniciando treinamento do modelo')
clf.fit(x_train, y_train)

y_prediction = clf.predict(x_test)

# Gerar visualização da previsão 
print(classification_report(y_test, y_prediction, zero_division=0.0))

# Serializar o modelo treinado
if not os.path.exists('model'):
    os.makedirs('model')

sio.dump(clf, "./model/cls.skops")