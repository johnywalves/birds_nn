import os
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_database_for_training():
    data_dir = 'original'
    filepaths, image_list, label_list = [], [], []

    folders = os.listdir(data_dir)

    for fold in folders:
        fold_path = os.path.join(data_dir, fold)
        f_list = os.listdir(fold_path)
        for f in f_list:
            fpath = os.path.join(fold_path, f)
            filepaths.append(fpath)
            label_list.append(fold)
            
    for file in filepaths:
        image = cv2.imread(file)
        image = img_to_array(image)
        image_list.append(image)
        
    # Concatenate data paths with labels into one dataframe
    f_series = pd.Series(filepaths, name='filepaths')
    l_series = pd.Series(label_list, name='labels')
    df = pd.concat([f_series, l_series], axis=1)

    # Storing number of classes
    num_classes = len(df['labels'].unique())

    return df, num_classes


def separate_database_for_training(df):
    # Splitting dataset
    stratify = df['labels']
    train_df, dummy_df = train_test_split(
            df, 
            test_size=.3, 
            shuffle=True, 
            stratify=stratify, 
            random_state=48
        )

    # Valid and test dataframe
    stratify = dummy_df['labels']
    valid_df, test_df = train_test_split(
            dummy_df,
            test_size=.5,
            shuffle=True,
            stratify=stratify,
            random_state=123
        )

    print('\n\nNumbers')
    print(f"Number of Training dataset: {len(train_df)}\nNumber of Validation dataset: {len(valid_df)}\nNumber of Testing dataset: {len(test_df)}")
    print('\n\n')

    # Cropped image size
    batch_size = 8
    img_size = (224, 224)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)

    tr_gen = ImageDataGenerator()
    ts_gen = ImageDataGenerator()

    train_gen = tr_gen.flow_from_dataframe(
            train_df,
            x_col='filepaths',
            y_col='labels',
            target_size=img_size,
            class_mode='categorical', 
            color_mode='rgb',
            shuffle= True, 
            batch_size=batch_size
        )

    valid_gen = ts_gen.flow_from_dataframe(
            valid_df, 
            x_col='filepaths',
            y_col='labels',
            target_size=img_size,
            class_mode='categorical',
            color_mode='rgb',
            shuffle=True,
            batch_size=batch_size
        )

    test_gen = ts_gen.flow_from_dataframe(
            test_df,
            x_col='filepaths',
            y_col='labels',
            target_size=img_size,
            class_mode='categorical',
            color_mode='rgb',
            shuffle=False,
            batch_size=batch_size
        )
    
    return img_shape, train_gen, valid_gen, test_gen


def generate_report(prefix, history, model, test_dataset):
    # ==========================================================
    # Statistics to accuracy and loss
    # ==========================================================

    # Define needed variables
    tr_acc = history.history['acc']
    tr_loss = history.history['loss']
    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']

    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]

    epochs = [i+1 for i in range(len(tr_acc))]
    loss_label = f'best epoch= {str(index_loss + 1)}'
    acc_label = f'best epoch= {str(index_acc + 1)}'

    if not os.path.exists('figs'):
        os.makedirs('figs')

    # ==========================================================
    # Training and Validation Loss
    # ==========================================================

    # Plot Training and Validation Loss
    plt.figure(figsize= (20, 8))
    plt.style.use('fivethirtyeight')

    plt.plot(epochs, tr_loss, 'purple', label= 'Training loss')
    plt.plot(epochs, val_loss, 'gold', label= 'Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'darkblue', label= loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./figs/{prefix}_training_and_loss.jpg')

    # ==========================================================
    # Training and Validation Accuracy
    # ==========================================================

    # Plot Training and Validation Accuracy
    plt.figure(figsize= (20, 8))
    plt.style.use('fivethirtyeight')

    plt.plot(epochs, tr_acc, 'purple', label= 'Training Accuracy')
    plt.plot(epochs, val_acc, 'gold', label= 'Validation Accuracy')
    plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'darkblue', label= acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'./figs/{prefix}_training_and_accuracy.jpg')

    # ==========================================================
    # Confusion matrix
    # ==========================================================

    # Get Predictions
    predictions = model.predict(test_dataset)
    y_pred = np.argmax(predictions, axis=1)

    # Get Classes
    g_dict = test_dataset.class_indices
    classes = list(g_dict.keys())

    if not os.path.exists('data'):
        os.makedirs('data')

    df_classes = pd.DataFrame(classes, columns=['classes'])
    df_classes.to_csv('./data/classes.csv', header=None, index=None)

    # Confusion matrix
    cm = confusion_matrix(test_dataset.classes, y_pred)

    # Create a heat map
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues', 
                xticklabels=classes, 
                yticklabels=classes)

    # Add a title and label the axes
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Classes')
    plt.ylabel('True Classes')

    # Modify the rotation of axis labels
    plt.xticks(rotation=45)  # Rotation of x-axis labels
    plt.yticks(rotation=0)   # Rotation of y-axis labels

    plt.savefig(f'./figs/{prefix}_confusion_matrix.jpg')

    # ==========================================================
    # Classification report
    # ==========================================================

    # Print report
    print('\n\nClassification report')
    print(classification_report(test_dataset.classes, y_pred, target_names=classes))

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

def calc_entropy(f_series, predictions):
    # Distância de Mahalanobis ou Máxima Entropia
    def entropy(p):
        return -tf.reduce_sum(p * K.log(p + 1e-6), axis=-1)

    ent = entropy(predictions)

    entropies = pd.Series(ent, name='entropies')

    result = pd.concat([f_series, entropies], axis=1)

    return result 

    # # Se a entropia for alta, é OOD
    # ood_threshold = 2.5
    # if ent > ood_threshold:
    #     print("Amostra OOD detectada pela entropia")
    # else:
    #     print("Amostra dentro da distribuição")