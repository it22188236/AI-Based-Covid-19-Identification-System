# src/evaluate.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'      # Suppress TensorFlow info/warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'    # Disable oneDNN custom ops message

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from src.model import grad_cam
from src.utils import display_prediction_with_heatmap

# -------------------------- Paths --------------------------
model_path = 'models/trained_model.h5'
test_dir = 'data/processed/test/'
results_dir = 'results/heatmaps/'
os.makedirs(results_dir, exist_ok=True)

# -------------------------- Settings --------------------------
class_names = ['COVID', 'Normal', 'Pneumonia']
num_samples_per_class = 5   # Change this number to get more/fewer heatmaps per class

# -------------------------- Load Model --------------------------
model = load_model(model_path)
print("Model loaded successfully!\n")

# -------------------------- Evaluation Loop --------------------------
total_processed = 0

for cls in class_names:
    class_path = os.path.join(test_dir, cls)
    
    if not os.path.exists(class_path):
        print(f"Warning: Folder not found: {class_path}")
        continue
    
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(images) == 0:
        print(f"No images found in {cls} test folder. Skipping.\n")
        continue
    
    # Select random samples (or all if fewer than requested)
    selected_images = random.sample(images, min(len(images), num_samples_per_class))
    
    print(f"Processing {len(selected_images)} sample(s) from class '{cls}'...\n")
    
    for img_file in selected_images:
        img_path = os.path.join(class_path, img_file)
        
        # Load and preprocess image
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Shape: (1, 224, 224, 3)
        
        # Predict
        pred_probs = model.predict(img_array, verbose=0)[0]
        pred_idx = np.argmax(pred_probs)
        pred_class = class_names[pred_idx]
        confidence = pred_probs[pred_idx] * 100
        
        # Generate Grad-CAM heatmap
        heatmap = grad_cam(model, img_array)
        
        # Save visualization
        base_name = os.path.splitext(img_file)[0]  # Remove extension
        save_path = os.path.join(results_dir, f'{cls}_{base_name}_gradcam.png')
        
        display_prediction_with_heatmap(
            original_img_path=img_path,
            heatmap=heatmap,
            prediction=pred_class,
            confidence=confidence,
            save_path=save_path
        )
        
        correct = "✓" if pred_class == cls else "✗"
        print(f"  {correct} {img_file}")
        print(f"     True: {cls} | Predicted: {pred_class} ({confidence:.2f}%)\n")
        
        total_processed += 1

# -------------------------- Summary --------------------------
print("="*60)
print("EVALUATION COMPLETE!")
print(f"Generated {total_processed} Grad-CAM visualizations")
print(f"Saved to: {os.path.abspath(results_dir)}")
print("\nOpen the PNG files to see:")
print("   • Left:   Original X-ray")
print("   • Middle: Grad-CAM Heatmap (red = high attention)")
print("   • Right:  Overlay + Prediction & Confidence")
print("\nFor your research paper:")
print("   → Select best COVID examples where heatmap highlights bilateral lower zones / ground-glass opacities")
print("   → Normal cases should show minimal or diffuse activation")
print("   → Pneumonia cases should focus on consolidation areas")
print("="*60)