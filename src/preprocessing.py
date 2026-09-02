"""
src/preprocessing.py
--------------------
Unit 1: Introduction to Computer Vision & Vision Systems
  - Dataset loading and class distribution visualization.
Unit 2: Preprocessing
  - Image resizing to 32x32 fixed dimensions.
  - Pixel normalization (scaling to [0, 1]).
  - Histogram Equalization (CLAHE) comparison on sample images.
  - Dataset splitting into Train, Validation, and Test sets (80/10/10).
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

CLASS_NAMES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing >3.5t', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles',
    16: 'Vehicles >3.5t prohibited', 17: 'No entry', 18: 'General caution',
    19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve',
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians',
    28: 'Children crossing', 29: 'Bicycles crossing', 30: 'Beware of ice/snow',
    31: 'Wild animals crossing', 32: 'End speed & passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing', 42: 'End no passing >3.5t'
}

RAW_DIR = os.path.join('data', 'raw')
TRAIN_DIR = os.path.join('data', 'train')
TEST_DIR = os.path.join('data', 'test')
RESULTS_DIR = 'results'

IMG_SIZE = (32, 32)


def plot_class_distribution(df):
    """
    Unit 1: Plot and save dataset class distribution bar chart.
    """
    print("[Unit 1] Generating class distribution bar chart...")
    class_counts = df['ClassId'].value_counts().sort_index()

    plt.figure(figsize=(14, 6), facecolor='#111827')
    ax = plt.gca()
    ax.set_facecolor('#1f2937')

    palette = sns.color_palette("plasma", n_colors=43)
    bars = plt.bar(class_counts.index, class_counts.values, color=palette, edgecolor='#374151', linewidth=1)

    plt.title('GTSRB Dataset Class Distribution (Unit 1: Dataset Overview)', fontsize=14, fontweight='bold', color='white', pad=15)
    plt.xlabel('Traffic Sign Class ID (0 - 42)', fontsize=12, color='white', labelpad=10)
    plt.ylabel('Number of Samples', fontsize=12, color='white', labelpad=10)
    plt.xticks(range(0, 43, 2), color='white', fontsize=10)
    plt.yticks(color='white', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.3, color='#4b5563')

    plt.text(0.98, 0.95, f'Total Samples: {len(df):,}', transform=ax.transAxes,
             fontsize=12, fontweight='bold', color='#38bdf8', ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#111827', edgecolor='#38bdf8', alpha=0.8))

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'class_distribution.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to {out_path}")


def apply_histogram_equalization(image_rgb):
    """
    Unit 2: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on Y channel of YCrCb.
    """
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return equalized


def plot_histogram_comparison(samples_orig, samples_eq, sample_labels):
    """
    Unit 2: Visualize before and after Histogram Equalization comparison.
    """
    print("[Unit 2] Generating Histogram Equalization comparison plot...")
    n_samples = len(samples_orig)
    fig, axes = plt.subplots(n_samples, 4, figsize=(12, 3 * n_samples), facecolor='#111827')

    for i in range(n_samples):
        label_str = CLASS_NAMES.get(sample_labels[i], f"Class {sample_labels[i]}")
        
        axes[i, 0].imshow(samples_orig[i])
        axes[i, 0].set_title(f"Original\n({label_str})", color='white', fontsize=10)
        axes[i, 0].axis('off')

        gray_orig = cv2.cvtColor(samples_orig[i], cv2.COLOR_RGB2GRAY)
        axes[i, 1].hist(gray_orig.ravel(), 256, [0, 256], color='#38bdf8', alpha=0.8)
        axes[i, 1].set_title("Original Hist", color='white', fontsize=10)
        axes[i, 1].set_facecolor('#1f2937')
        axes[i, 1].tick_params(colors='white')
        axes[i, 1].grid(alpha=0.2)

        axes[i, 2].imshow(samples_eq[i])
        axes[i, 2].set_title("Equalized (CLAHE)", color='#4ade80', fontsize=10, fontweight='bold')
        axes[i, 2].axis('off')

        gray_eq = cv2.cvtColor(samples_eq[i], cv2.COLOR_RGB2GRAY)
        axes[i, 3].hist(gray_eq.ravel(), 256, [0, 256], color='#4ade80', alpha=0.8)
        axes[i, 3].set_title("Equalized Hist", color='white', fontsize=10)
        axes[i, 3].set_facecolor('#1f2937')
        axes[i, 3].tick_params(colors='white')
        axes[i, 3].grid(alpha=0.2)

    plt.suptitle("Unit 2: Preprocessing - Histogram Equalization Comparison", color='white', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'histogram_eq_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison to {out_path}")


def _load_single_image(row_data):
    rel_path, class_id = row_data
    img_path = os.path.join(RAW_DIR, rel_path)
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, IMG_SIZE)
    return img_resized, class_id


def load_and_preprocess_dataset():
    """
    Load raw images with ThreadPoolExecutor, resize to 32x32, normalize, and split 80/10/10.
    """
    train_csv_path = os.path.join(RAW_DIR, 'Train.csv')
    test_csv_path = os.path.join(RAW_DIR, 'Test.csv')

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    plot_class_distribution(train_df)

    print("[Unit 2] Loading & Resizing (32x32) images in parallel...")

    train_rows = list(zip(train_df['Path'], train_df['ClassId']))
    
    images = []
    labels = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = executor.map(_load_single_image, train_rows)
        for img, cid in results:
            if img is not None:
                images.append(img)
                labels.append(cid)

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)

    print(f"  Parallel loading complete: {len(images)} training images loaded.")

    sample_indices = np.random.choice(len(images), 4, replace=False)
    sample_orig = [images[idx] for idx in sample_indices]
    sample_labels = [labels[idx] for idx in sample_indices]
    sample_eq = [apply_histogram_equalization(img) for img in sample_orig]

    plot_histogram_comparison(sample_orig, sample_eq, sample_labels)

    print("[Unit 2] Splitting dataset into Train (80%), Val (10%), Test (10%)...")
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.20, random_state=42, stratify=labels
    )
    X_val, X_test_sub, y_val, y_test_sub = train_test_split(
        X_val, y_val, test_size=0.50, random_state=42, stratify=y_val
    )

    test_rows = list(zip(test_df['Path'], test_df['ClassId']))
    test_images = []
    test_labels = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = executor.map(_load_single_image, test_rows)
        for img, cid in results:
            if img is not None:
                test_images.append(img)
                test_labels.append(cid)

    test_images = np.array(test_images, dtype=np.uint8)
    test_labels = np.array(test_labels, dtype=np.int32)

    # Use fast uncompressed savez
    print("  Saving processed arrays to disk...")
    np.savez(
        os.path.join(TRAIN_DIR, 'train_data.npz'),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val
    )

    np.savez(
        os.path.join(TEST_DIR, 'test_data.npz'),
        X_test=test_images, y_test=test_labels
    )

    np.savez(
        os.path.join('data', 'demo_samples.npz'),
        images=X_test_sub[:50], labels=y_test_sub[:50]
    )

    print(f"  Preprocessing complete!")
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Val set:   {X_val.shape[0]} samples")
    print(f"  Test set:  {test_images.shape[0]} samples")


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    load_and_preprocess_dataset()
