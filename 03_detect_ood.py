import pandas as pd
import tensorflow as tf
from functions import get_database_for_test, calc_entropy

from tensorflow.keras.models import load_model

test_gen, f_series = get_database_for_test()

cnn_model = load_model('./model/original_bird_species.keras')

predictions = cnn_model.predict(test_gen)

# ==========================================================
# Calcular a entropia das previsões
# ==========================================================
entropy = calc_entropy(f_series, predictions)

print(entropy)
