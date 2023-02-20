import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the directories for the training and validation datasets
train_dir = "path/to/training/dataset"
validation_dir = "path/to/validation/dataset"

# Define some image preprocessing options
# These are applied to the training dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Rescale pixel values to be between 0 and 1
    rotation_range=40,  # Rotate images randomly up to 40 degrees
    width_shift_range=0.2,  # Shift images horizontally up to 20%
    height_shift_range=0.2,  # Shift images vertically up to 20%
    shear_range=0.2,  # Shear images up to 20%
    zoom_range=0.2,  # Zoom in on images up to 20%
    horizontal_flip=True,  # Flip images horizontally
    fill_mode='nearest')  # Fill in missing pixels with nearby pixels

# These options are applied to the validation dataset, but no data augmentation is performed
validation_datagen = ImageDataGenerator(rescale=1./255)

# Define the batch size and image dimensions
batch_size = 32
img_height = 150
img_width = 150

# Create the training dataset
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# Create the validation dataset
validation_data = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# Define the neural network architecture
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model with an appropriate loss function and optimizer
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

# Train the model on the training data, and use the validation data to evaluate performance
history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // batch_size,
    epochs=30,
    validation_data=validation_data,
    validation_steps=validation_data.samples // batch_size)

# Evaluate the model on the validation data
model.evaluate(validation_data)

# Save the trained model to disk
model.save("cats_vs_dogs.h5")
