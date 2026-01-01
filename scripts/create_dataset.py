"""
COVID-19 Lung Segmentation - Dataset Creation
Creates dummy dataset for training when real data is not available
"""

import numpy as np
import cv2
import os
from tqdm import tqdm

print("=" * 80)
print("CREATING DUMMY DATASET FOR TRAINING")
print("=" * 80)

def create_covid_image():
    """Create COVID-19 positive CXR image"""
    img = np.zeros((256, 256), dtype=np.uint8)
    
    # Lungs with COVID patterns
    cv2.ellipse(img, (100, 128), (80, 110), 0, 0, 360, 180, -1)  # Left lung
    cv2.ellipse(img, (156, 128), (80, 110), 0, 0, 360, 180, -1)  # Right lung
    
    # COVID-19 patterns (ground glass opacities)
    for _ in range(15):
        x = np.random.randint(40, 216)
        y = np.random.randint(40, 216)
        radius = np.random.randint(8, 25)
        intensity = np.random.randint(160, 220)
        cv2.circle(img, (x, y), radius, intensity, -1)
    
    # Add some consolidation areas
    for _ in range(5):
        x = np.random.randint(60, 196)
        y = np.random.randint(60, 196)
        w = np.random.randint(20, 40)
        h = np.random.randint(20, 40)
        intensity = np.random.randint(140, 180)
        cv2.rectangle(img, (x, y), (x+w, y+h), intensity, -1)
    
    return img

def create_normal_image():
    """Create normal CXR image"""
    img = np.zeros((256, 256), dtype=np.uint8)
    
    # Clear lungs
    cv2.ellipse(img, (100, 128), (80, 110), 0, 0, 360, 200, -1)  # Left lung
    cv2.ellipse(img, (156, 128), (80, 110), 0, 0, 360, 200, -1)  # Right lung
    
    # Clear background
    img[img == 0] = np.random.randint(30, 60)
    
    return img

def create_lung_mask(image):
    """Create lung mask for segmentation"""
    _, mask = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    
    # Clean mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

def create_dataset(num_samples=200):
    """Create complete dataset"""
    print(f"\nCreating {num_samples} samples...")
    
    # Create directories
    base_dirs = [
        'datasets/train/images',
        'datasets/train/masks',
        'datasets/train/labels',
        'datasets/val/images', 
        'datasets/val/masks',
        'datasets/val/labels',
        'datasets/test/images',
        'datasets/test/masks',
        'datasets/test/labels'
    ]
    
    for dir_path in base_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Create samples
    train_count = int(num_samples * 0.7)
    val_count = int(num_samples * 0.15)
    test_count = num_samples - train_count - val_count
    
    print(f"Train: {train_count}, Val: {val_count}, Test: {test_count}")
    
    sample_id = 0
    
    # Create training set
    print("\nCreating training set...")
    for i in tqdm(range(train_count)):
        # Alternate between COVID and normal
        if i % 2 == 0:
            img = create_covid_image()
            label = 1  # COVID
        else:
            img = create_normal_image()
            label = 0  # Normal
        
        mask = create_lung_mask(img)
        
        # Save files
        cv2.imwrite(f'datasets/train/images/sample_{sample_id:04d}.png', img)
        cv2.imwrite(f'datasets/train/masks/sample_{sample_id:04d}.png', mask)
        np.save(f'datasets/train/labels/sample_{sample_id:04d}.npy', label)
        
        sample_id += 1
    
    # Create validation set
    print("\nCreating validation set...")
    for i in tqdm(range(val_count)):
        if i % 2 == 0:
            img = create_covid_image()
            label = 1
        else:
            img = create_normal_image()
            label = 0
        
        mask = create_lung_mask(img)
        
        cv2.imwrite(f'datasets/val/images/sample_{sample_id:04d}.png', img)
        cv2.imwrite(f'datasets/val/masks/sample_{sample_id:04d}.png', mask)
        np.save(f'datasets/val/labels/sample_{sample_id:04d}.npy', label)
        
        sample_id += 1
    
    # Create test set
    print("\nCreating test set...")
    for i in tqdm(range(test_count)):
        if i % 2 == 0:
            img = create_covid_image()
            label = 1
        else:
            img = create_normal_image()
            label = 0
        
        mask = create_lung_mask(img)
        
        cv2.imwrite(f'datasets/test/images/sample_{sample_id:04d}.png', img)
        cv2.imwrite(f'datasets/test/masks/sample_{sample_id:04d}.png', mask)
        np.save(f'datasets/test/labels/sample_{sample_id:04d}.npy', label)
        
        sample_id += 1
    
    return sample_id

def main():
    # Create dataset
    total_samples = create_dataset(num_samples=200)
    
    # Create dataset info file
    info = f"""
    ============================================
    COVID-19 LUNG SEGMENTATION DATASET
    ============================================
    
    Total Samples: {total_samples}
    Created: {np.datetime64('now')}
    
    Dataset Structure:
    datasets/
    ├── train/
    │   ├── images/     - CXR images (256x256)
    │   ├── masks/      - Lung masks
    │   └── labels/     - 0=Normal, 1=COVID
    ├── val/
    │   ├── images/
    │   ├── masks/
    │   └── labels/
    └── test/
        ├── images/
        ├── masks/
        └── labels/
    
    Image Format: PNG (256x256 grayscale)
    Mask Format: PNG (256x256 binary)
    Label Format: NumPy array (0 or 1)
    
    Class Distribution:
    - COVID-19 Positive: {total_samples // 2} samples
    - Normal: {total_samples // 2} samples
    
    Usage:
    1. For segmentation: Use images and masks
    2. For classification: Use images and labels
    3. For combined: Use all three
    
    ============================================
    """
    
    with open('datasets/dataset_info.txt', 'w') as f:
        f.write(info)
    
    print("\n" + "=" * 80)
    print(f"✅ DATASET CREATED SUCCESSFULLY!")
    print(f"   Total samples: {total_samples}")
    print(f"   Location: datasets/")
    print("=" * 80)
    
    # Show sample images
    print("\n📊 SAMPLE STATISTICS:")
    train_images = len(os.listdir('datasets/train/images'))
    val_images = len(os.listdir('datasets/val/images'))
    test_images = len(os.listdir('datasets/test/images'))
    
    print(f"Training images: {train_images}")
    print(f"Validation images: {val_images}")
    print(f"Test images: {test_images}")
    
    # Create visualization
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Show first COVID sample
    covid_img = cv2.imread('datasets/train/images/sample_0000.png', cv2.IMREAD_GRAYSCALE)
    covid_mask = cv2.imread('datasets/train/masks/sample_0000.png', cv2.IMREAD_GRAYSCALE)
    covid_label = np.load('datasets/train/labels/sample_0000.npy')
    
    axes[0,0].imshow(covid_img, cmap='gray')
    axes[0,0].set_title(f"COVID Sample (Label: {covid_label})")
    axes[0,0].axis('off')
    
    axes[0,1].imshow(covid_mask, cmap='gray')
    axes[0,1].set_title("Lung Mask")
    axes[0,1].axis('off')
    
    axes[0,2].imshow(cv2.bitwise_and(covid_img, covid_img, mask=covid_mask), cmap='gray')
    axes[0,2].set_title("Lung Region")
    axes[0,2].axis('off')
    
    # Show first Normal sample
    normal_img = cv2.imread('datasets/train/images/sample_0001.png', cv2.IMREAD_GRAYSCALE)
    normal_mask = cv2.imread('datasets/train/masks/sample_0001.png', cv2.IMREAD_GRAYSCALE)
    normal_label = np.load('datasets/train/labels/sample_0001.npy')
    
    axes[1,0].imshow(normal_img, cmap='gray')
    axes[1,0].set_title(f"Normal Sample (Label: {normal_label})")
    axes[1,0].axis('off')
    
    axes[1,1].imshow(normal_mask, cmap='gray')
    axes[1,1].set_title("Lung Mask")
    axes[1,1].axis('off')
    
    axes[1,2].imshow(cv2.bitwise_and(normal_img, normal_img, mask=normal_mask), cmap='gray')
    axes[1,2].set_title("Lung Region")
    axes[1,2].axis('off')
    
    plt.suptitle("Dataset Samples - COVID-19 vs Normal", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('datasets/dataset_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()