# import system libraries
import os
from PIL import Image
from functions import get_database_for_training, separate_database_for_training, generate_report

# import data handling tools
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

# import Deep Learning Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adamax

import warnings
warnings.filterwarnings('ignore')

df, num_classes = get_database_for_training()

# ==========================================================
# Visualize dataset 
# ==========================================================

# Set the number of images to display per species
num_images = 6

# Get unique bird species labels
bird_species = df['labels'].unique()

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

img_shape, train_gen, valid_gen, test_gen = separate_database_for_training(df)

# ==========================================================
# Create and Training Model
# ==========================================================

# Create Model Structure
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
        Dense(num_classes, activation='softmax')
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
history = cnn_model.fit(
    x=train_gen, 
    verbose=1, 
    validation_data=valid_gen,
    epochs=1000, 
    callbacks=[early_stopping, plateau]
)

# Save the model
if not os.path.exists('model'):
    os.makedirs('model')

cnn_model.save('./model/original_bird_species.keras')

generate_report('original', history, cnn_model, test_gen)