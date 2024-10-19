# import system libraries
import os
from PIL import Image
import cv2

# import data handling tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# import Deep Learning Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings('ignore')

data_dir = 'preprocessed'
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

# Set the number of images to display per species
num_images = 6

# Get unique bird species labels
bird_species = df['labels'].unique()

# ==========================================================
# Visualize dataset 
# ==========================================================

# Set up the plot
plt.figure(figsize=(20, 20))

# Loop through each bird species
for idx, bird in enumerate(bird_species):
    # Filter the DataFrame to get file paths for this bird species
    bird_df = df[df['labels'] == bird].sample(num_images)  # Get a random sample of 16 images
    
    # Loop through the 16 images and plot them
    for i, file in enumerate(bird_df['filepaths'].values):
        plt.subplot(len(bird_species), num_images, idx * num_images + i + 1)
        img = Image.open(file)
        plt.imshow(img)
        plt.axis('off')
        plt.title(bird)

# Show the plot
if not os.path.exists('figs'):
    os.makedirs('figs')

plt.tight_layout()
plt.savefig('./figs/visualize_birds_species.jpg')

# ==========================================================
# Splitting training and test dataset
# ==========================================================

# Splitting dataset
stratify = df['labels']
train_df, dummy_df = train_test_split(df, test_size=.3, shuffle=True, stratify=stratify, random_state=123)

# Valid and test dataframe
stratify = dummy_df['labels']
valid_df, test_df = train_test_split(dummy_df, test_size=.5, shuffle=True, stratify=stratify, random_state=123)

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

train_gen = tr_gen.flow_from_dataframe(train_df,
                                       x_col='filepaths',
                                       y_col='labels',
                                       target_size=img_size,
                                       class_mode='categorical', 
                                       color_mode='rgb',
                                       shuffle= True, 
                                       batch_size=batch_size)

valid_gen = ts_gen.flow_from_dataframe(valid_df, 
                                       x_col='filepaths',
                                       y_col='labels',
                                       target_size=img_size,
                                       class_mode='categorical',
                                       color_mode='rgb',
                                       shuffle=True,
                                       batch_size=batch_size)

test_gen = ts_gen.flow_from_dataframe(test_df,
                                      x_col='filepaths',
                                      y_col='labels',
                                      target_size=img_size,
                                      class_mode='categorical',
                                      color_mode='rgb',
                                      shuffle=False,
                                      batch_size=batch_size)

# ==========================================================
# Create and Training Model
# ==========================================================

# Create Model Structure
class_count = len(list(train_gen.class_indices.keys()))

def load_model():
    model = Sequential([
        Conv2D(8, (3,3), activation='relu', padding='same', input_shape=img_shape),
        MaxPooling2D((3,3)),
        
        Conv2D(16, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        
        Conv2D(32, (3,3), activation='relu', padding='same'),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        
        Conv2D(64, (3,3), activation='relu', padding='same'),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        
        Flatten(),

        Dense(128, activation='relu'),
        Dense(256, activation='relu'),
        Dense(class_count, activation='softmax')
    ])

    model.compile(Adamax(learning_rate=.0005), loss = 'categorical_crossentropy', metrics=['acc'])

    return model

early_stopping = EarlyStopping(
    patience=10,
    min_delta=0,
    mode=min,
    monitor='val_loss',
    verbose=0,
    restore_best_weights=True,
    baseline=None
)

plateau = ReduceLROnPlateau(
    patience=4,
    mode=min,
    monitor='val_loss',
    factor=.2,
    verbose=0
)

# Create CNN model
cnn_model = load_model()
history = cnn_model.fit(x= train_gen, 
                        verbose=1, 
                        validation_data=valid_gen,
                        epochs=1000, 
                        callbacks=[early_stopping, plateau])

# Save the model
if not os.path.exists('model'):
    os.makedirs('model')

cnn_model.save('./model/bird_species.keras')

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

Epochs = [i+1 for i in range(len(tr_acc))]
loss_label = f'best epoch= {str(index_loss + 1)}'
acc_label = f'best epoch= {str(index_acc + 1)}'

# ==========================================================
# Training and Validation Loss
# ==========================================================

# Plot Training and Validation Loss
plt.figure(figsize= (20, 8))
plt.style.use('fivethirtyeight')

plt.plot(Epochs, tr_loss, 'purple', label= 'Training loss')
plt.plot(Epochs, val_loss, 'gold', label= 'Validation loss')
plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'darkblue', label= loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('./figs/training_and_loss.jpg')

# ==========================================================
# Training and Validation Accuracy
# ==========================================================

# Plot Training and Validation Accuracy
plt.figure(figsize= (20, 8))
plt.style.use('fivethirtyeight')

plt.plot(Epochs, tr_acc, 'purple', label= 'Training Accuracy')
plt.plot(Epochs, val_acc, 'gold', label= 'Validation Accuracy')
plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'darkblue', label= acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('./figs/training_and_accuracy.jpg')

# ==========================================================
# Confusion matrix
# ==========================================================

# Get Predictions
predictions = cnn_model.predict(test_gen)
y_pred = np.argmax(predictions, axis=1)

# Get Classes
g_dict = test_gen.class_indices
classes = list(g_dict.keys())

if not os.path.exists('data'):
    os.makedirs('data')

df_classes = pd.DataFrame(classes, columns=['classes'])
df_classes.to_csv('./data/classes.csv', header=None, index=None)

# Confusion matrix
cm = confusion_matrix(test_gen.classes, y_pred)

# Create a heat map
plt.figure(figsize=(10, 8))
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

plt.savefig('./figs/confusion_matrix.jpg')

# ==========================================================
# Classification report
# ==========================================================

# Print report
print('\n\nClassification report')
print(classification_report(test_gen.classes, y_pred, target_names=classes))
