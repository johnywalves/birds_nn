import pandas as pd
from sklearn.neural_network import MLPClassifier

# Carregar as fonte de dados
x_train = pd.read_csv('birds_x_train')
x_test = pd.read_csv('birds_x_test')
y_train = pd.read_csv('birds_y_train')
x_test = pd.read_csv('birds_y_test')

# Implementa a multi-layer perceptron (MLP) com o algoritmo de Backpropagation
clf = MLPClassifier(solver='lbfgs', 
                    alpha=1e-5,
                    hidden_layer_sizes=(5, 2), 
                    random_state=84)

clf.fit(x_train, y_train)

prediction = clf.predict(x_test)