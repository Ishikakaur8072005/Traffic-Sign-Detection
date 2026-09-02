"""
src/segmentation.py
-------------------
Unit 6: Image Segmentation
  - Color-based (HSV) thresholding for traffic sign isolation (Red & Blue hue ranges).
  - Region-based Watershed segmentation algorithm for shape boundary extraction.
  - Saves comparison grid to results/segmentation_samples.png.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = 'results'
DEMO_PATH = os.path.join('data', 'demo_samples.npz')


def apply_hsv_color_segmentation(img_rgb):
    """
    Unit 6: Segment red and blue traffic sign regions using HSV thresholding.
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    lower_blue, upper_blue = np.array([100, 70, 50]), np.array([130, 255, 255])

    mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

    color_mask = cv2.bitwise_or(mask_red1, mask_red2)
    color_mask = cv2.bitwise_or(color_mask, mask_blue)

    segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=color_mask)
    return color_mask, segmented


def apply_watershed_segmentation(img_rgb):
    """
    Unit 6: Apply Watershed Region Segmentation to isolate foreground object contours.
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    sure_bg = cv2.dilate(opening, kernel, iterations=2)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_watershed = img_bgr.copy()
    cv2.watershed(img_watershed, markers)

    img_watershed_rgb = cv2.cvtColor(img_watershed, cv2.COLOR_BGR2RGB)
    img_watershed_rgb[markers == -1] = [0, 255, 0]

    return img_watershed_rgb, markers


def demo_segmentation(sample_images, sample_labels):
    """
    Unit 6: Run and save color & region segmentation pipeline.
    """
    print("[Unit 6] Running Image Segmentation demo (HSV + Watershed)...")
    n_samples = min(4, len(sample_images))
    fig, axes = plt.subplots(n_samples, 4, figsize=(12, 3 * n_samples), facecolor='#111827')

    for i in range(n_samples):
        img_rgb = sample_images[i]
        if img_rgb.dtype != np.uint8:
            img_rgb = (img_rgb * 255).astype(np.uint8)

        color_mask, color_segmented = apply_hsv_color_segmentation(img_rgb)
        watershed_out, markers = apply_watershed_segmentation(img_rgb)

        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"Original Sign (Class {sample_labels[i]})", color='white', fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(color_mask, cmap='gray')
        axes[i, 1].set_title("HSV Color Mask\n(Red/Blue Hue)", color='#38bdf8', fontsize=10)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(color_segmented)
        axes[i, 2].set_title("HSV Segmented Region", color='#38bdf8', fontsize=10)
        axes[i, 2].axis('off')

        axes[i, 3].imshow(watershed_out)
        axes[i, 3].set_title("Watershed Boundaries\n(Region Segmentation)", color='#4ade80', fontsize=10, fontweight='bold')
        axes[i, 3].axis('off')

    plt.suptitle("Unit 6: Image Segmentation - HSV Color Thresholding & Watershed", color='white', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'segmentation_samples.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved segmentation samples to {out_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(DEMO_PATH):
        demo_data = np.load(DEMO_PATH)
        sample_images = demo_data['images']
        sample_labels = demo_data['labels']
        demo_segmentation(sample_images, sample_labels)
    else:
        raise FileNotFoundError(f"Missing {DEMO_PATH}. Run src/preprocessing.py first.")


if __name__ == '__main__':
    main()
