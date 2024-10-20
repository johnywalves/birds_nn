import pandas as pd
pd.options.display.float_format = '{:,.6f}'.format

from functions import get_database_for_test, calc_entropy

from tensorflow.keras.models import load_model

test_gen, f_series = get_database_for_test()

classes = pd.read_csv("./data/classes.csv", header=None)

model_original = load_model('./model/original_bird_species.keras')
predictions_original = model_original.predict(test_gen)
df_predictions_original = pd.DataFrame(predictions_original) \
                    .rename(classes[0], axis='columns')

model_generalization = load_model('./model/generalization_bird_species.keras')
predictions_generalization = model_generalization.predict(test_gen)
df_predictions_generalization = pd.DataFrame(predictions_generalization) \
                    .rename(classes[0], axis='columns')


print('\n\n==========================================================')
print('Original')
print('==========================================================')

results_original = pd.concat([f_series, df_predictions_original], axis=1)
print(results_original)

entropy_original = calc_entropy(f_series, predictions_original)
print(entropy_original)

print('\n==========================================================')
print('Generalização')
print('==========================================================')

results_generalization = pd.concat([f_series, df_predictions_generalization], axis=1)
print(results_generalization)

entropy_generalization = calc_entropy(f_series, predictions_generalization)
print(entropy_generalization)


