"""
COVID-19 Lung Segmentation - Fixed Training
Compatible with NumPy 1.24.3
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
import matplotlib.pyplot as plt

print("=" * 80)
print("COVID-19 LUNG SEGMENTATION - TRAINING")
print("Compatible with NumPy 1.24.3")
print("=" * 80)

# Check NumPy version
print(f"NumPy Version: {np.__version__}")
print(f"TensorFlow Version: {tf.__version__}")

# ==================== LOAD DATASET ====================
def load_dataset():
    """Load dataset from files"""
    print("\nLoading dataset...")
    
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    # Load training data
    train_img_dir = 'datasets/train/images'
    train_mask_dir = 'datasets/train/masks'
    
    train_files = sorted([f for f in os.listdir(train_img_dir) if f.endswith('.png')])
    
    print(f"Found {len(train_files)} training samples")
    
    for i, filename in enumerate(train_files[:50]):  # Load first 50 for speed
        # Load image
        img_path = os.path.join(train_img_dir, filename)
        img = plt.imread(img_path)
        if len(img.shape) == 3:  # If RGB, convert to grayscale
            img = np.mean(img, axis=2)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)  # Add channel
        
        # Load mask
        mask_path = os.path.join(train_mask_dir, filename)
        mask = plt.imread(mask_path)
        if len(mask.shape) == 3:
            mask = np.mean(mask, axis=2)
        mask = (mask > 0.5).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        
        X_train.append(img)
        y_train.append(mask)
    
    # Load validation data
    val_img_dir = 'datasets/val/images'
    val_mask_dir = 'datasets/val/masks'
    
    val_files = sorted([f for f in os.listdir(val_img_dir) if f.endswith('.png')])
    
    for i, filename in enumerate(val_files[:10]):  # First 10
        img_path = os.path.join(val_img_dir, filename)
        img = plt.imread(img_path)
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        
        mask_path = os.path.join(val_mask_dir, filename)
        mask = plt.imread(mask_path)
        if len(mask.shape) == 3:
            mask = np.mean(mask, axis=2)
        mask = (mask > 0.5).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        
        X_val.append(img)
        y_val.append(mask)
    
    # Load test data
    test_img_dir = 'datasets/test/images'
    test_mask_dir = 'datasets/test/masks'
    
    test_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.png')])
    
    for i, filename in enumerate(test_files[:10]):  # First 10
        img_path = os.path.join(test_img_dir, filename)
        img = plt.imread(img_path)
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        
        mask_path = os.path.join(test_mask_dir, filename)
        mask = plt.imread(mask_path)
        if len(mask.shape) == 3:
            mask = np.mean(mask, axis=2)
        mask = (mask > 0.5).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        
        X_test.append(img)
        y_test.append(mask)
    
    # Convert to arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    print(f"\nDataset loaded:")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ==================== BUILD MODEL ====================
def build_model():
    """Build simple U-Net model"""
    print("\nBuilding U-Net model...")
    
    inputs = tf.keras.Input(shape=(256, 256, 1))
    
    # Encoder
    conv1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(16, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPool2D(2)(conv1)
    
    conv2 = layers.Conv2D(32, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPool2D(2)(conv2)
    
    # Bottleneck
    conv3 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv3)
    
    # Decoder
    up4 = layers.UpSampling2D(2)(conv3)
    up4 = layers.concatenate([up4, conv2])
    conv4 = layers.Conv2D(32, 3, activation='relu', padding='same')(up4)
    conv4 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv4)
    
    up5 = layers.UpSampling2D(2)(conv4)
    up5 = layers.concatenate([up5, conv1])
    conv5 = layers.Conv2D(16, 3, activation='relu', padding='same')(up5)
    conv5 = layers.Conv2D(16, 3, activation='relu', padding='same')(conv5)
    
    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv5)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# ==================== MAIN TRAINING ====================
def main():
    # Create directories
    os.makedirs('models/trained', exist_ok=True)
    os.makedirs('training_results', exist_ok=True)
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset()
    
    # Build and compile model
    model = build_model()
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n" + "=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    model.summary()
    
    # Train model
    print("\n" + "=" * 50)
    print("STARTING TRAINING (10 epochs)")
    print("=" * 50)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=4,
        verbose=1
    )
    
    # Save model
    model.save('models/trained/lung_unet.h5')
    print("\n💾 Model saved: models/trained/lung_unet.h5")
    
    # Evaluate
    print("\n" + "=" * 50)
    print("EVALUATION")
    print("=" * 50)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Make predictions
    predictions = model.predict(X_test[:3], verbose=0)
    
    # Visualize results
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    for i in range(3):
        # Original
        axes[i, 0].imshow(X_test[i].squeeze(), cmap='gray')
        axes[i, 0].set_title(f"Sample {i+1} - Input")
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(y_test[i].squeeze(), cmap='gray')
        axes[i, 1].set_title(f"Sample {i+1} - Ground Truth")
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(predictions[i].squeeze(), cmap='gray')
        axes[i, 2].set_title(f"Sample {i+1} - Prediction")
        axes[i, 2].axis('off')
    
    plt.suptitle("U-Net Lung Segmentation Results", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_results/segmentation_results.png', dpi=150, bbox_inches='tight')
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle('Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_results/training_history.png', dpi=150, bbox_inches='tight')
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    print("\n📁 RESULTS SAVED:")
    print("- models/trained/lung_unet.h5 (Trained model)")
    print("- training_results/segmentation_results.png (Predictions)")
    print("- training_results/training_history.png (Training curves)")
    
    print(f"\n📊 FINAL METRICS:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Create report
    report = f"""
======================================================
COVID-19 LUNG SEGMENTATION - TRAINING REPORT
======================================================

Training Completed Successfully!

MODEL INFORMATION:
- Architecture: U-Net
- Input Shape: 256x256x1
- Parameters: {model.count_params():,}
- Epochs: 10
- Batch Size: 4

DATASET:
- Training Samples: {len(X_train)}
- Validation Samples: {len(X_val)}
- Test Samples: {len(X_test)}

RESULTS:
- Test Loss: {test_loss:.4f}
- Test Accuracy: {test_acc:.4f}

FILES GENERATED:
1. models/trained/lung_unet.h5 - Trained model
2. training_results/segmentation_results.png - Predictions
3. training_results/training_history.png - Training curves

NEXT STEPS:
1. Use the model for lung segmentation
2. Integrate with COVID-19 classification
3. Create GUI application

======================================================
"""
    
    with open('training_results/training_report.txt', 'w') as f:
        f.write(report)
    
    print("\n📄 Report saved: training_results/training_report.txt")
    print("\n🎯 READY FOR PROGRESS PRESENTATION!")

if __name__ == "__main__":
    main()