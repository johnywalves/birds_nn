import os
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_database_for_test():
    filepaths = []
    labels = []

    folder_original = 'test'

    for filename in os.listdir(folder_original):
        img_path = os.path.join(folder_original, filename)

        filepaths.append(img_path)
        labels.append('')

    # Concatenate data paths with labels into one dataframe
    f_series = pd.Series(filepaths, name='filepaths')
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

    return test_gen, f_series