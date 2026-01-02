# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import os
# import shutil
# from sklearn.model_selection import train_test_split

# # Define paths
# raw_dir = 'data/raw/'
# processed_dir = 'data/processed/'

# # Split data (70/15/15)
# classes = ['COVID', 'Normal', 'Pneumonia']
# for cls in classes:
#     images = os.listdir(os.path.join(raw_dir, cls))
#     train_imgs, temp_imgs = train_test_split(images, test_size=0.3)
#     val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5)
    
#     # Copy to processed folders
#     for split, imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
#         os.makedirs(os.path.join(processed_dir, split, cls), exist_ok=True)
#         for img in imgs:
#             shutil.copy(os.path.join(raw_dir, cls, img), os.path.join(processed_dir, split, cls, img))

# # Data generators with augmentation
# train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
# val_gen = ImageDataGenerator(rescale=1./255)
# # Load: train_gen.flow_from_directory(processed_dir + 'train/', target_size=(224,224), batch_size=32, class_mode='categorical')

# src/data_preparation.py
import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
raw_dir = 'data/raw/'
processed_dir = 'data/processed/'

# Classes (must match your folder names exactly)
classes = ['COVID', 'Normal', 'Pneumonia']

print("Starting data preparation...\n")
print(f"Raw data directory: {os.path.abspath(raw_dir)}\n")

for cls in classes:
    class_raw_dir = os.path.join(raw_dir, cls)
    
    if not os.path.exists(class_raw_dir):
        print(f"ERROR: Folder {class_raw_dir} does not exist!")
        continue
    
    images = [f for f in os.listdir(class_raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(images)
    
    print(f"Class '{cls}': {num_images} images found")
    
    if num_images == 0:
        print(f"  → No images in {cls}. Skipping.\n")
        continue
    
    # Create directories for train/val/test
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(processed_dir, split, cls), exist_ok=True)
    
    # If too few images, put all in train and skip val/test for this class
    if num_images < 10:
        print(f"  → Too few images ({num_images}). All placed in 'train' folder.")
        for img in images:
            shutil.copy(os.path.join(class_raw_dir, img), os.path.join(processed_dir, 'train', cls, img))
        print(f"  → Completed {cls}\n")
        continue
    
    # Standard split: 70% train, 15% val, 15% test
    train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42, stratify=None)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42, stratify=None)
    
    print(f"  → Split: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")
    
    # Copy files
    for img in train_imgs:
        shutil.copy(os.path.join(class_raw_dir, img), os.path.join(processed_dir, 'train', cls, img))
    for img in val_imgs:
        shutil.copy(os.path.join(class_raw_dir, img), os.path.join(processed_dir, 'val', cls, img))
    for img in test_imgs:
        shutil.copy(os.path.join(class_raw_dir, img), os.path.join(processed_dir, 'test', cls, img))
    
    print(f"  → Completed {cls}\n")

print("Data preparation finished!")
print(f"Processed data saved to: {os.path.abspath(processed_dir)}")
print("\nYou can now proceed to training the model.")