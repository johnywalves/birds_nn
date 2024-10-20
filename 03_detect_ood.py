import pandas as pd
import tensorflow as tf
from functions import get_database_for_test

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

test_gen, f_series = get_database_for_test()

cnn_model = load_model('./model/bird_species.keras')

predictions = cnn_model.predict(test_gen)

# ==========================================================
# Calcular a entropia das previsões
# ==========================================================
def entropy(p):
    return -tf.reduce_sum(p * K.log(p + 1e-6), axis=-1)

ent = entropy(predictions)

entropies = pd.Series(ent, name='entropies')

result = pd.concat([f_series, entropies], axis=1)

print(result)

# # Se a entropia for alta, é OOD
# ood_threshold = 2.5
# if ent > ood_threshold:
#     print("Amostra OOD detectada pela entropia")
# else:
#     print("Amostra dentro da distribuição")



