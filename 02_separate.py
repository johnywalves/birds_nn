import pandas as pd
from sklearn.model_selection import train_test_split

# Carregar a fonte de dados
birds_data = pd.read_csv("birds.csv")

# Separar os dados entre features e labels
X = birds_data.drop(columns=[0], axis=1)
y = birds_data[0]

# Dividir a fonte de dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Salvar a fontes de dados em arquivos
x_train.to_csv('birds_x_train')
x_test.to_csv('birds_x_test')
y_train.to_csv('birds_y_train')
x_test.to_csv('birds_y_test')
