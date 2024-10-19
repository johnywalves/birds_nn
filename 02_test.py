import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Concatenate data paths with labels into one dataframe
filepaths = ['./test/penguin.jpg', './test/lake.jpg', './test/buoy.jpg']
f_series = pd.Series(filepaths, name='filepaths')
labels = ['', '', '']
l_series = pd.Series(labels, name='labels')
df = pd.concat([f_series, l_series], axis=1)

test_gen = ImageDataGenerator() \
        .flow_from_dataframe(
            df,
            x_col='filepaths',
            y_col='labels',
            target_size=(224, 224),
            class_mode='categorical',
            color_mode='rgb',
            shuffle=False,
            batch_size=8
        )

classes = pd.read_csv("./data/classes.csv", header=None)
cnn_model = load_model('./model/bird_species.keras')

predictions = cnn_model.predict(test_gen)

pd.options.display.float_format = '{:,.6f}'.format
df_predictions = pd.DataFrame(predictions).rename(classes[0], axis='columns')

result = pd.concat([f_series, df_predictions], axis=1)

print(result)
