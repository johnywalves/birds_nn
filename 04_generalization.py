# import system libraries
import os
from PIL import Image
from functions import get_database_for_training, separate_database_for_training, generate_report

# import Deep Learning Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.regularizers import l2

import warnings
warnings.filterwarnings('ignore')

df, num_classes = get_database_for_training()

# ==========================================================
# Splitting training and test dataset
# ==========================================================

img_shape, train_gen, valid_gen, test_gen = separate_database_for_training(df)

# ==========================================================
# Create and Training Model
# ==========================================================

WEIGHT_DECAY = 0.0005

# Create Model Structure
def load_model():
    model = Sequential([
        Conv2D(8, (3,3), activation='relu', padding='same', input_shape=img_shape),
        MaxPooling2D((3,3)),

        Conv2D(16, (3,3), activation='relu', padding='same', use_bias=False, kernel_regularizer=l2(WEIGHT_DECAY)),
        MaxPooling2D((2,2)),
        
        Conv2D(32, (3,3), activation='relu', padding='same', use_bias=False, kernel_regularizer=l2(WEIGHT_DECAY)),
        Conv2D(32, (3,3), activation='relu', padding='same', use_bias=False, kernel_regularizer=l2(WEIGHT_DECAY)),
        MaxPooling2D((2,2)),
        
        Conv2D(64, (3,3), activation='relu', padding='same', use_bias=False, kernel_regularizer=l2(WEIGHT_DECAY)),
        Conv2D(64, (3,3), activation='relu', padding='same', use_bias=False, kernel_regularizer=l2(WEIGHT_DECAY)),
        Conv2D(64, (3,3), activation='relu', padding='same', use_bias=False, kernel_regularizer=l2(WEIGHT_DECAY)),
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
history = cnn_model.fit(x= train_gen, 
                        verbose=1, 
                        validation_data=valid_gen,
                        epochs=1000, 
                        callbacks=[early_stopping, plateau])

# Save the model
if not os.path.exists('model'):
    os.makedirs('model')

cnn_model.save('./model/generalization_bird_species.keras')

generate_report('generalization', history, cnn_model, test_gen)