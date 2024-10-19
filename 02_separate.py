import pandas as pd
from sklearn.model_selection import train_test_split

# Carregar a fonte de dados
birds_data = pd.read_csv("./data/birds.csv", header=None)

# Separar os dados entre features e labels
x = birds_data.drop(columns=[0])
y = birds_data[0]

# Dividir a fonte de dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Salvar a fontes de dados em arquivos
x_train.to_csv('./data/birds_x_train.csv', header=None, index=None)
x_test.to_csv('./data/birds_x_test.csv', header=None, index=None)
y_train.to_csv('./data/birds_y_train.csv', header=None, index=None)
y_test.to_csv('./data/birds_y_test.csv', header=None, index=None)
