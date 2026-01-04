# src/utils.py
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

def plot_training_history(history, save_path='results/plots/training_history.png'):
    """
    Plot accuracy and loss curves from training history.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")

def display_prediction_with_heatmap(original_img_path, heatmap, prediction, confidence, save_path=None):
    """
    Display original X-ray, heatmap, and overlay side-by-side with prediction text.
    """
    img = cv2.imread(original_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(img)
    axs[0].set_title('Original X-ray')
    axs[0].axis('off')
    
    axs[1].imshow(heatmap_resized, cmap='jet')
    axs[1].set_title('Grad-CAM Heatmap')
    axs[1].axis('off')
    
    axs[2].imshow(overlay)
    axs[2].set_title(f'Overlay\nPrediction: {prediction} ({confidence:.2f}%)')
    axs[2].axis('off')
    
    plt.suptitle('Model Explanation using Grad-CAM', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

def evaluate_and_report(model, test_generator, class_names, save_dir='results/reports/'):
    """
    Generate classification report and confusion matrix.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Predictions
    y_pred = np.argmax(model.predict(test_generator), axis=1)
    y_true = test_generator.classes
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(save_dir, 'classification_report.csv'))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    print(f"Evaluation artifacts saved to {save_dir}")

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """
    Load an image and preprocess it for model input.
    """
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array, img  # Return both array (for model) and PIL image (for display)