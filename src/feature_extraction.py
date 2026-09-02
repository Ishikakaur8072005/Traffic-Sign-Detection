"""
src/feature_extraction.py
--------------------------
Unit 3: Feature Detection & Extraction
  - Canny Edge Detection visualization (saved to results/edge_detection_samples.png)
  - HOG (Histogram of Oriented Gradients) feature extraction & visualization (saved to results/hog_visualization.png)
  - PCA (Principal Component Analysis) 2D dimensionality reduction & scatter plot (saved to results/pca_visualization.png)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
from sklearn.decomposition import PCA

RESULTS_DIR = 'results'
DEMO_PATH = os.path.join('data', 'demo_samples.npz')
TRAIN_PATH = os.path.join('data', 'train', 'train_data.npz')


def demo_canny_edge_detection(sample_images, sample_labels):
    """
    Unit 3: Apply Canny Edge Detection on sample traffic signs.
    """
    print("[Unit 3] Running Canny Edge Detection demo...")
    n_samples = min(4, len(sample_images))
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 2.5 * n_samples), facecolor='#111827')

    for i in range(n_samples):
        img_rgb = sample_images[i]
        if img_rgb.dtype != np.uint8:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"Original Sign (Class {sample_labels[i]})", color='white', fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title("Grayscale", color='white', fontsize=10)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(edges, cmap='magma')
        axes[i, 2].set_title("Canny Edges", color='#38bdf8', fontsize=10, fontweight='bold')
        axes[i, 2].axis('off')

    plt.suptitle("Unit 3: Feature Detection - Canny Edge Detection", color='white', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'edge_detection_samples.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved Canny edge plot to {out_path}")


def demo_hog_visualization(sample_images, sample_labels):
    """
    Unit 3: Extract and visualize HOG (Histogram of Oriented Gradients).
    """
    print("[Unit 3] Running HOG visualization demo...")
    n_samples = min(4, len(sample_images))
    fig, axes = plt.subplots(n_samples, 2, figsize=(8, 3 * n_samples), facecolor='#111827')

    for i in range(n_samples):
        img_rgb = sample_images[i]
        if img_rgb.dtype != np.uint8:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        features, hog_image = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,
            channel_axis=None
        )

        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"Original Sign (Class {sample_labels[i]})", color='white', fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(hog_image, cmap='inferno')
        axes[i, 1].set_title(f"HOG Descriptor Map\n({len(features)} Features)", color='#4ade80', fontsize=10, fontweight='bold')
        axes[i, 1].axis('off')

    plt.suptitle("Unit 3: Feature Detection - HOG (Histogram of Oriented Gradients)", color='white', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'hog_visualization.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved HOG visualization to {out_path}")


def demo_pca_visualization(X_train, y_train):
    """
    Unit 3: Flatten pixel features, reduce to 2D using PCA, and plot scatter plot.
    """
    print("[Unit 3] Running PCA 2D Projection demo...")
    
    indices = np.random.choice(len(X_train), min(2000, len(X_train)), replace=False)
    X_sub = X_train[indices].astype(np.float32) / 255.0
    y_sub = y_train[indices]

    X_flat = X_sub.reshape(X_sub.shape[0], -1)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_flat)
    var_exp = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 8), facecolor='#111827')
    ax = plt.gca()
    ax.set_facecolor('#1f2937')

    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=y_sub, cmap='turbo', alpha=0.7, s=25, edgecolor='none'
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label('Class ID (0 - 42)', color='white', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

    plt.title(f"Unit 3: Feature Reduction - PCA 2D Projection\nExpl. Var: PC1={var_exp[0]*100:.1f}%, PC2={var_exp[1]*100:.1f}%",
              color='white', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Principal Component 1", color='white', fontsize=11)
    plt.ylabel("Principal Component 2", color='white', fontsize=11)
    plt.tick_params(colors='white')
    plt.grid(alpha=0.2, color='#4b5563')

    out_path = os.path.join(RESULTS_DIR, 'pca_visualization.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved PCA plot to {out_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if os.path.exists(DEMO_PATH):
        demo_data = np.load(DEMO_PATH)
        sample_images = demo_data['images']
        sample_labels = demo_data['labels']
    else:
        raise FileNotFoundError(f"Missing {DEMO_PATH}. Run src/preprocessing.py first.")

    if os.path.exists(TRAIN_PATH):
        train_data = np.load(TRAIN_PATH)
        X_train = train_data['X_train']
        y_train = train_data['y_train']
    else:
        raise FileNotFoundError(f"Missing {TRAIN_PATH}. Run src/preprocessing.py first.")

    demo_canny_edge_detection(sample_images, sample_labels)
    demo_hog_visualization(sample_images, sample_labels)
    demo_pca_visualization(X_train, y_train)


if __name__ == '__main__':
    main()
