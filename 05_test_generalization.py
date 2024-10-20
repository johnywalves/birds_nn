import pandas as pd
from functions import get_database_for_test

from tensorflow.keras.models import load_model

test_gen, f_series = get_database_for_test()

cnn_model = load_model('./model/generalization_bird_species.keras')

predictions = cnn_model.predict(test_gen)

# ==========================================================
# Apresentação de resultados
# ==========================================================
classes = pd.read_csv("./data/classes.csv", header=None)

pd.options.display.float_format = '{:,.6f}'.format
df_predictions = pd.DataFrame(predictions) \
                    .rename(classes[0], axis='columns')

result = pd.concat([f_series, df_predictions], axis=1)

print(result)
