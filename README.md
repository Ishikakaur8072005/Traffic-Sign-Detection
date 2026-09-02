# 🚦 Traffic Sign Detection & Classification

A comprehensive Computer Vision and Deep Learning system for traffic sign recognition and localization using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

This repository implements a multi-stage computer vision pipeline combining digital image processing, feature extraction, traditional machine learning (SVM), and deep learning (Convolutional Neural Networks) to detect and classify 43 traffic sign categories.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Computer Vision Pipeline Architecture](#-computer-vision-pipeline-architecture)
4. [Methodology & Technical Components](#-methodology--technical-components)
5. [Presentation & PPT Structure Guide](#-presentation--ppt-structure-guide)
6. [Experimental Results & Evaluation](#-experimental-results--evaluation)
7. [Visual Artifacts Gallery](#-visual-artifacts-gallery)
8. [Repository Structure](#-repository-structure)
9. [Installation & Usage](#-installation--usage)
10. [Viva Voce Q&A Reference](#-viva-voce-qa-reference)

---

## 🎯 Project Overview

Traffic sign recognition is an essential capability for **Autonomous Vehicles (AV)** and **Advanced Driver Assistance Systems (ADAS)**. Real-world conditions introduce challenges such as poor lighting, severe weather, motion blur, and partial occlusions.

This project delivers an end-to-end solution that takes raw road imagery, enhances contrast, extracts spatial and color features, performs region segmentation, localizes sign candidate regions with bounding boxes, and classifies traffic signs with high precision.

### Key Highlights
- **Dataset**: German Traffic Sign Recognition Benchmark (GTSRB)
- **Categories**: 43 distinct traffic sign classes (Speed Limits, Warnings, Mandatory, Prohibitory)
- **Image Resolution**: Normalized to $32 \times 32 \times 3$ RGB pixels
- **CNN Classifier Test Accuracy**: **96.91%**
- **SVM Baseline Test Accuracy**: **75.40%**

---

## ✨ Key Features

- **Automated Data Preprocessing**: Standardized resizing, min-max pixel scaling $[0.0, 1.0]$, and multithreaded dataset processing.
- **Contrast Enhancement**: Contrast Limited Adaptive Histogram Equalization (CLAHE) applied to the luminance channel to handle glare and shadow regions.
- **Feature Extraction & Dimensionality Reduction**: Integrated Canny edge detection, Histogram of Oriented Gradients (HOG) extraction, and 2D Principal Component Analysis (PCA) visualization.
- **Color & Watershed Segmentation**: HSV color space thresholding targeting sign hue ranges combined with Watershed marker-based boundary extraction.
- **Dual Classification Engine**: Comparison between a classical Support Vector Machine (SVM) baseline and a 4-Layer Convolutional Neural Network (CNN).
- **Candidate Region Object Detection**: Contour-based region proposal algorithm with aspect-ratio filtering and CNN verification for bounding box visualization.

---

## 🏗️ Computer Vision Pipeline Architecture

```
[ Input Road Image ]
          │
          ▼
[ Data Preprocessing ] ────────► Resizing (32x32), Normalization [0,1], CLAHE Equalization
          │
          ▼
[ Feature & Segmentation ] ────► Canny Edges, HOG Descriptors, 2D PCA, HSV Masking & Watershed
          │
          ▼
[ Model Classification ] ──────► Classical SVM Baseline (75.40%) vs. 4-Layer Keras CNN (96.91%)
          │
          ▼
[ Object Detection ] ──────────► Contour Region Proposals + CNN Bounding Box Predictions
          │
          ▼
[ Evaluation & Reporting ] ────► Confusion Matrix Heatmap, Classification Metrics, Error Samples
```

---

## 🔬 Methodology & Technical Components

### 1. Data Preprocessing & Contrast Enhancement
Raw traffic images often suffer from uneven illumination. Images are resized to $32 \times 32$ pixels and normalized. To enhance low-contrast inputs, images are converted to the **YCrCb** color space, and **CLAHE** is applied exclusively to the luminance ($Y$) channel:
$$I_{\text{CLAHE}} = \text{CLAHE}(Y), \quad \text{recombined with } Cr, Cb$$

### 2. Feature Extraction (Canny, HOG, PCA)
- **Canny Edge Detection**: Highlights structural sign shapes (triangles, octagons, circles) using dual-threshold gradient intensity checking.
- **Histogram of Oriented Gradients (HOG)**: Computes local gradient magnitude and orientation distributions across $8 \times 8$ cells, creating a robust feature representation.
- **Principal Component Analysis (PCA)**: Projects high-dimensional pixel vectors ($3072$-dimensions) down to $2\text{D}$ space to analyze class distribution and variance.

### 3. Segmentation (HSV & Watershed)
- **HSV Color Segmentation**: Converts images to HSV space and applies color boundaries to isolate red (Hue $[0,10] \cup [170,180]$) and blue (Hue $[100,130]$) sign regions.
- **Watershed Algorithm**: Utilizes distance transforms and morphological markers to separate foreground sign contours from complex background textures.

### 4. Classification Models
- **Classical Baseline**: Support Vector Machine (SVM) trained on extracted HOG descriptors.
- **Deep CNN Architecture**:
  - `Conv2D(32, 3x3)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `Conv2D(32, 3x3)` $\rightarrow$ `MaxPool(2x2)` $\rightarrow$ `Dropout(0.25)`
  - `Conv2D(64, 3x3)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `Conv2D(64, 3x3)` $\rightarrow$ `MaxPool(2x2)` $\rightarrow$ `Dropout(0.30)`
  - `Flatten` $\rightarrow$ `Dense(256)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `Dropout(0.50)` $\rightarrow$ `Dense(43, Softmax)`

### 5. Object Detection & Localization
Combines classical candidate region proposal (HSV color filtering + contour bounding boxes with aspect ratio $0.65 \le w/h \le 1.45$) with the trained CNN classifier to draw labeled bounding boxes and confidence scores over test scenes.

---

## 📺 Presentation & PPT Structure Guide

Use the slide outlines below to prepare your presentation slides directly:

### **Slide 1: Title Slide**
- **Title**: Traffic Sign Detection & Classification using Computer Vision
- **Subtitle**: End-to-End Pipeline Implementation on GTSRB Dataset
- **Focus**: Traditional Image Processing + Deep Learning (CNN)

### **Slide 2: Problem Statement & Significance**
- Real-time traffic sign detection is critical for Autonomous Vehicles (AV) and ADAS.
- Challenges: Variable lighting, shadows, weather conditions, motion blur, and class imbalance.
- Goal: Design an intelligent system capable of detection, segmentation, and classification across 43 categories.

### **Slide 3: System Pipeline Architecture**
- Overview of the 5 key stages: *Preprocessing $\rightarrow$ Feature Extraction $\rightarrow$ Segmentation $\rightarrow$ Classification $\rightarrow$ Object Detection*.
- Comparison between traditional machine learning (HOG + SVM) and deep learning (CNN).

### **Slide 4: Data Preprocessing & CLAHE Enhancement**
- Resizing images to a uniform $32 \times 32 \times 3$ size and normalizing pixel values to $[0, 1]$.
- Applying Contrast Limited Adaptive Histogram Equalization (CLAHE) on the Y channel in YCrCb color space to improve visibility in dark/shadowed images.
- Visual: `results/histogram_eq_comparison.png`

### **Slide 5: Feature Extraction & Dimensionality Reduction**
- **Canny Edge Detection**: Captures geometric contours (circles, octagons, triangles).
- **HOG Descriptors**: Encodes directional gradient distributions in $8 \times 8$ local cell blocks.
- **2D PCA Projection**: Visualizes cluster separation across classes in reduced 2D feature space.
- Visuals: `results/edge_detection_samples.png`, `results/hog_visualization.png`, `results/pca_visualization.png`

### **Slide 6: Image Segmentation (HSV & Watershed)**
- **HSV Thresholding**: Isolates characteristic traffic sign colors (Red and Blue hue channels).
- **Watershed Algorithm**: Uses morphological opening/closing and distance transforms to separate sign boundaries from background elements.
- Visual: `results/segmentation_samples.png`

### **Slide 7: Model Architectures & Training**
- **SVM Baseline**: Linear SVM trained on HOG features (**75.40% Accuracy**).
- **CNN Architecture**: 4 Convolutional layers with Batch Normalization, Max Pooling, and Dropout regularization.
- **CNN Result**: Achieves **96.91% Test Accuracy**.
- Visual: `results/training_curves.png`

### **Slide 8: Object Detection & Candidate Proposals**
- Proposes candidate regions using HSV color masks and contour bounding boxes ($0.65 \le w/h \le 1.45$).
- Passes candidate patches to the trained CNN model for classification and confidence evaluation.
- Displays color-coded bounding boxes with predicted labels and confidence scores.
- Visual: `results/detection_samples.png`

### **Slide 9: Evaluation & Performance Breakdown**
- Multi-class confusion matrix ($43 \times 43$) heatmap showing low off-diagonal confusion.
- Evaluation on test set: High precision, recall, and F1-scores across categories.
- Visuals: `results/confusion_matrix.png`, `results/predictions_samples.png`, `results/summary.txt`

### **Slide 10: Conclusion & Key Learnings**
- Successfully built a complete, robust traffic sign detection and classification pipeline.
- Demonstrated that learned CNN features significantly outperform hand-crafted HOG features (96.91% vs. 75.40%).
- System is lightweight and efficient for CPU inference.

---

## 📊 Experimental Results & Evaluation

| Model | Representation | Test Accuracy | Model File |
| :--- | :--- | :---: | :--- |
| **Linear SVM Baseline** | Hand-crafted HOG Features | **75.40%** | `models/svm_baseline.pkl` |
| **4-Layer CNN** | Learned Convolutional Feature Maps | **96.91%** | `models/traffic_sign_cnn.keras` |

---

## 🖼️ Visual Artifacts Gallery

All generated plots are stored in `results/` using a dark-mode palette suitable for slides:

- `results/class_distribution.png`: Bar chart displaying sample frequency across 43 GTSRB categories.
- `results/histogram_eq_comparison.png`: CLAHE contrast enhancement comparison grid.
- `results/edge_detection_samples.png`: Canny edge detector visual outputs.
- `results/hog_visualization.png`: HOG gradient orientation maps.
- `results/pca_visualization.png`: 2D PCA scatter plot showing class clustering.
- `results/segmentation_samples.png`: HSV color mask and Watershed boundary segmentation.
- `results/training_curves.png`: Training vs. Validation accuracy and loss curves.
- `results/detection_samples.png`: Annotated bounding boxes with predicted class and confidence.
- `results/confusion_matrix.png`: $43 \times 43$ multi-class confusion matrix heatmap.
- `results/predictions_samples.png`: Correct vs. misclassified sample predictions.

---

## 📂 Repository Structure

```
CV_Traffic_Sign_Project/
├── src/                   <- Modular Python scripts
│   ├── preprocessing.py   <- Data loading, resizing, CLAHE contrast enhancement
│   ├── feature_extraction.py <- Canny edge detection, HOG descriptors, PCA 2D scatter plot
│   ├── segmentation.py    <- HSV color thresholding & Watershed segmentation
│   ├── classification.py  <- SVM baseline & Keras CNN model training
│   ├── detection.py       <- Contour region proposal & CNN object detection
│   └── evaluate.py        <- Model evaluation, confusion matrix & metrics
├── models/                <- Saved trained models (.keras and .pkl)
├── results/               <- Generated visual PNG plots and summary text
├── requirements.txt       <- Project dependencies
├── .gitignore             <- Excludes raw dataset binaries and temp files
└── README.md              <- Project documentation & presentation guide
```

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.10+ or Python 3.12
- Required libraries listed in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

### Running Modules
You can execute each script independently:

```bash
# 1. Preprocessing & Data Loading
python src/preprocessing.py

# 2. Feature Extraction (Canny, HOG, PCA)
python src/feature_extraction.py

# 3. Image Segmentation (HSV & Watershed)
python src/segmentation.py

# 4. Model Training (SVM Baseline & CNN)
python src/classification.py

# 5. Object Detection & Localization
python src/detection.py

# 6. Evaluation & Confusion Matrix
python src/evaluate.py
```

---

## 💡 Viva Voce Q&A Reference

**Q1: Why convert images to YCrCb space for CLAHE instead of RGB?**
> *Answer*: RGB channels correlate color and brightness together. In YCrCb, the $Y$ channel represents luminance (brightness), while $Cr$ and $Cb$ store chrominance (color). Applying CLAHE exclusively to $Y$ enhances contrast without distorting color information.

**Q2: How does HOG feature extraction work?**
> *Answer*: HOG divides an image into small connected cells ($8 \times 8$ pixels), calculates gradient magnitude and direction for each pixel, and builds a 9-bin orientation histogram per cell. Neighboring cells are normalized in blocks to improve invariance to lighting changes.

**Q3: Why does the CNN achieve significantly higher accuracy than SVM?**
> *Answer*: SVM relies on hand-crafted HOG features that capture fixed edge gradients. The CNN dynamically learns multi-layer hierarchical representations (low-level edges $\rightarrow$ mid-level shapes $\rightarrow$ high-level sign symbols) directly optimized for the classification task.

**Q4: How does your candidate region proposal detector operate?**
> *Answer*: It converts road images to HSV color space, masks red and blue sign colors, extracts candidate contours filtered by aspect ratio ($0.65 \le w/h \le 1.45$), crops candidate patches, and feeds them into the trained CNN model for confidence verification and labeling.
