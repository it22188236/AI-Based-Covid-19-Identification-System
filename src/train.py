# src/train.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Remove oneDNN message

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.model import build_model
from src.utils import plot_training_history

# -------------------------- Paths --------------------------
processed_dir = 'data/processed/'
model_save_path = 'models/trained_model.h5'
plot_save_path = 'results/plots/training_history.png'

os.makedirs('models', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

# -------------------------- Parameters --------------------------
img_size = 224
batch_size = 32
epochs = 25  # You can change to 20 if you want

# -------------------------- Data Generators --------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    processed_dir + 'train/',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    processed_dir + 'val/',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

print(f"Classes found: {train_generator.class_indices}")

# -------------------------- Build & Compile Model --------------------------
num_classes = len(train_generator.class_indices)
model = build_model(num_classes=num_classes)

model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Lower LR is better for transfer learning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------- Callbacks --------------------------
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(model_save_path, save_best_only=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]

# -------------------------- Training --------------------------
print("\nStarting training...\n")

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# -------------------------- Save Model & Plot --------------------------
model.save(model_save_path)
print(f"\nModel saved to {model_save_path}")

plot_training_history(history, plot_save_path)
print(f"Training plots saved to {plot_save_path}")

print("\nTraining completed successfully!")