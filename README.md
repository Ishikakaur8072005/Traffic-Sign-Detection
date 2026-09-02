# 🚦 Traffic Sign Detection & Classification using Computer Vision

A complete, end-to-end Computer Vision project developed using Python, OpenCV, Scikit-Learn, and TensorFlow/Keras on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

This repository covers **all 6 units** of the Computer Vision course curriculum. Each module is runnable independently and outputs presentation-ready visual plots saved in `results/`.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Syllabus Unit Mapping](#-syllabus-unit-mapping)
3. [Slide-by-Slide PPT Outline & Content](#-slide-by-slide-ppt-outline--content)
4. [Computer Vision Pipeline Architecture](#-computer-vision-pipeline-architecture)
5. [Final Experimental Results](#-final-experimental-results)
6. [Visual Artifacts Gallery](#-visual-artifacts-gallery)
7. [Repository Structure](#-repository-structure)
8. [How to Setup & Run](#-how-to-setup--run)
9. [Viva Voce Q&A Cheat Sheet](#-viva-voce-qa-cheat-sheet)

---

## 🎯 Project Overview

Traffic sign recognition is a vital component of **Autonomous Driving Systems (ADS)** and **Advanced Driver Assistance Systems (ADAS)**. This project demonstrates how classical computer vision techniques (edge detection, color thresholding, HOG descriptors, PCA, watershed segmentation) work alongside deep learning architectures (Convolutional Neural Networks) to solve traffic sign classification and localization.

- **Dataset**: German Traffic Sign Recognition Benchmark (GTSRB)
- **Total Classes**: 43 Traffic Sign Categories (Speed Limits, Warnings, Mandatory Signs, Yield, Stop, etc.)
- **Resolution**: Normalized to $32 \times 32 \times 3$ RGB pixels
- **CNN Test Accuracy**: **96.91%**
- **SVM Baseline Test Accuracy**: **75.40%**

---

## 📚 Syllabus Unit Mapping

| Unit | Course Topic | Technique Implemented | Python Script | Output Visual Artifact |
| :---: | :--- | :--- | :--- | :--- |
| **Unit 1** | Introduction & Vision Systems | Class Distribution Analysis across 43 classes | [`src/preprocessing.py`](src/preprocessing.py) | `results/class_distribution.png` |
| **Unit 2** | Preprocessing | Image Resizing ($32\times32$), $[0,1]$ Normalization, CLAHE Equalization | [`src/preprocessing.py`](src/preprocessing.py) | `results/histogram_eq_comparison.png` |
| **Unit 3** | Feature Detection & Extraction | Canny Edge Detector, HOG Descriptors, 2D PCA Scatter Plot | [`src/feature_extraction.py`](src/feature_extraction.py) | `results/edge_detection_samples.png`<br>`results/hog_visualization.png`<br>`results/pca_visualization.png` |
| **Unit 4** | Image Classification | Classical SVM Baseline vs. 4-Layer Keras CNN Model | [`src/classification.py`](src/classification.py) | `results/training_curves.png`<br>`models/traffic_sign_cnn.keras`<br>`models/svm_baseline.pkl` |
| **Unit 5** | Object Detection | Contour-based Candidate Region Proposal + CNN Classification | [`src/detection.py`](src/detection.py) | `results/detection_samples.png` |
| **Unit 6** | Image Segmentation | HSV Color Thresholding (Red/Blue signs) & Watershed Segmentation | [`src/segmentation.py`](src/segmentation.py) | `results/segmentation_samples.png` |
| **Eval** | Evaluation & Reporting | Multi-class Confusion Matrix Heatmap, F1 Report, Prediction Samples | [`src/evaluate.py`](src/evaluate.py) | `results/confusion_matrix.png`<br>`results/predictions_samples.png`<br>`results/summary.txt` |

---

## 📺 Slide-by-Slide PPT Outline & Content

Use the content below to populate your presentation slides directly:

### **Slide 1: Title Slide**
- **Title**: Traffic Sign Detection & Classification using Computer Vision
- **Subtitle**: A Unit-wise CV Pipeline Implementation on GTSRB Dataset
- **Key Highlight**: Classical Computer Vision Techniques + Deep Convolutional Neural Networks

### **Slide 2: Problem Statement & Motivation**
- **Problem**: Autonomous vehicles require real-time, highly accurate traffic sign recognition under varying illumination, weather, and occlusions.
- **Objective**: Build a modular CV pipeline covering preprocessing, feature detection, segmentation, classical ML, and deep CNN classification across 43 classes.
- **Dataset**: GTSRB dataset containing cropped traffic sign images categorized into 43 distinct classes.

### **Slide 3: Unit 1 - Vision System & Dataset Distribution**
- **Syllabus Topic**: Vision Systems, Image Acquisition & Dataset Overview.
- **Implementation**: Extracted total class frequencies across 43 GTSRB sign categories.
- **Key Observation**: Highly imbalanced dataset — frequent signs (e.g. 50km/h speed limit) have >2,000 samples, whereas rare signs (e.g. 20km/h speed limit) have ~200 samples.
- **Visual Asset**: `results/class_distribution.png`

### **Slide 4: Unit 2 - Image Preprocessing & Contrast Enhancement**
- **Syllabus Topic**: Image Normalization & Histogram Equalization.
- **Techniques Applied**:
  1. **Fixed Resizing**: Uniform $32 \times 32 \times 3$ resolution.
  2. **Min-Max Normalization**: Pixel scaling $I_{\text{norm}} = \frac{I}{255.0} \in [0.0, 1.0]$.
  3. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Enhances local contrast on Y channel of YCrCb color space to handle extreme shadows and bright glare.
- **Visual Asset**: `results/histogram_eq_comparison.png`

### **Slide 5: Unit 3 - Feature Detection & Dimensionality Reduction**
- **Syllabus Topic**: Edges, HOG Descriptors, PCA.
- **Techniques Applied**:
  1. **Canny Edge Detection**: Detects sharp intensity gradients to capture sign geometry (triangles, circles, octagons).
  2. **HOG (Histogram of Oriented Gradients)**: Captures local edge direction distributions across $8 \times 8$ pixel cells.
  3. **PCA (Principal Component Analysis)**: Reduces $3072$-dimensional flattened pixel vectors to $2\text{D}$ space for cluster visualization.
- **Visual Assets**: `results/edge_detection_samples.png`, `results/hog_visualization.png`, `results/pca_visualization.png`

### **Slide 6: Unit 4 - Image Classification (SVM vs. CNN)**
- **Syllabus Topic**: Machine Learning & Deep Learning Classifiers.
- **Classical Baseline**: Linear Support Vector Machine (SVM) trained on HOG features (**75.40% Accuracy**).
- **CNN Architecture**:
  - Conv2D(32, 3x3) $\rightarrow$ BatchNorm $\rightarrow$ Conv2D(32, 3x3) $\rightarrow$ MaxPool(2x2) $\rightarrow$ Dropout(0.25)
  - Conv2D(64, 3x3) $\rightarrow$ BatchNorm $\rightarrow$ Conv2D(64, 3x3) $\rightarrow$ MaxPool(2x2) $\rightarrow$ Dropout(0.30)
  - Flatten $\rightarrow$ Dense(256) $\rightarrow$ BatchNorm $\rightarrow$ Dropout(0.50) $\rightarrow$ Softmax(43)
- **Result**: CNN achieves **96.91% Test Accuracy**.
- **Visual Asset**: `results/training_curves.png`

### **Slide 7: Unit 5 - Object Detection & Bounding Boxes**
- **Syllabus Topic**: Bounding Box Localization & Object Recognition.
- **Methodology**:
  1. **Color-based Candidate Region Proposal**: HSV mask isolates red/blue sign candidate shapes.
  2. **Contour Extraction & Aspect Ratio Filter**: Filters bounding boxes ($0.65 \le w/h \le 1.45$).
  3. **CNN Verification**: Passes candidate crops into the trained CNN for prediction and confidence scoring.
- **Visual Asset**: `results/detection_samples.png`

### **Slide 8: Unit 6 - Image Segmentation**
- **Syllabus Topic**: Thresholding, Color-based Segmentation, Region-based Watershed.
- **Techniques Applied**:
  1. **HSV Thresholding**: Isolates red (Hue [0-10], [170-180]) and blue (Hue [100-130]) sign region masks.
  2. **Watershed Algorithm**: Distance transform + marker-based morphological segmentation to separate foreground sign shapes from background context.
- **Visual Asset**: `results/segmentation_samples.png`

### **Slide 9: Experimental Results & Model Evaluation**
- **Confusion Matrix**: $43 \times 43$ heatmap displaying high diagonal precision.
- **Comparative Metrics**:
  - SVM Accuracy: **75.40%**
  - CNN Accuracy: **96.91%**
- **Visual Assets**: `results/confusion_matrix.png`, `results/predictions_samples.png`, `results/summary.txt`

### **Slide 10: Conclusion & Viva Summary**
- Developed a complete end-to-end Computer Vision pipeline covering Units 1 to 6.
- Demonstrated superiority of learned CNN features (96.91%) over hand-crafted HOG+SVM features (75.40%).
- Successfully generated high-contrast visual artifacts for each processing stage.

---

## 🏗️ Computer Vision Pipeline Architecture

```
[ Camera Image Acquisition ]
             │
             ▼
 [ Unit 1 & 2: Preprocessing ] ────► Resizing (32x32), Normalization [0,1], CLAHE Equalization
             │
             ▼
[ Unit 3 & 6: Feature & Segmentation ] ──► Canny Edges, HOG Descriptors, PCA 2D, HSV Masking & Watershed
             │
             ▼
 [ Unit 4: Image Classification ] ────► Classical SVM Baseline (75.40%) vs 4-Layer CNN (96.91%)
             │
             ▼
[ Unit 5: Object Detection ] ────────► Contour Candidate Proposal + CNN Bounding Box Classification
             │
             ▼
 [ Evaluation & Reporting ] ─────────► 43-Class Confusion Matrix, F1 Report, Prediction Samples
```

---

## 📊 Final Experimental Results

| Model | Technique | Test Accuracy | Output Model File |
| :--- | :--- | :---: | :--- |
| **Classical ML Baseline** | Linear SVM on HOG Descriptors | **75.40%** | `models/svm_baseline.pkl` |
| **Deep Learning CNN** | 4-Layer ConvNet + Dropout + BatchNorm | **96.91%** | `models/traffic_sign_cnn.keras` |

---

## 🖼️ Visual Artifacts Gallery

All plots are stored in `results/` formatted in high-contrast dark-mode theme for slides:

- `results/class_distribution.png`: Class sample frequency bar chart across 43 GTSRB categories.
- `histogram_eq_comparison.png`: CLAHE histogram equalization grid before vs. after.
- `edge_detection_samples.png`: Canny edge maps across traffic sign categories.
- `hog_visualization.png`: HOG gradient orientation maps.
- `pca_visualization.png`: 2D PCA scatter plot showing class clustering.
- `segmentation_samples.png`: HSV color mask & Watershed region segmentation outputs.
- `training_curves.png`: Loss & Accuracy curves over training epochs.
- `detection_samples.png`: Annotated bounding boxes with predicted class name and confidence.
- `confusion_matrix.png`: Multi-class $43 \times 43$ confusion matrix heatmap.
- `predictions_samples.png`: Correct vs. misclassified sign predictions grid.

---

## 📂 Repository Structure

```
CV_Traffic_Sign_Project/
├── src/                   <- Modular Python scripts per unit
│   ├── preprocessing.py   <- Unit 1 & Unit 2 (Class distribution, resizing, CLAHE)
│   ├── feature_extraction.py <- Unit 3 (Canny, HOG, PCA)
│   ├── segmentation.py    <- Unit 6 (HSV color thresholding & Watershed)
│   ├── classification.py  <- Unit 4 (SVM baseline & CNN model training)
│   ├── detection.py       <- Unit 5 (Contour region proposal + CNN detection)
│   └── evaluate.py        <- Evaluation module (Confusion matrix & metrics)
├── models/                <- Trained model artifacts (.keras and .pkl)
├── results/               <- Generated visual PNG plots and summary text
├── requirements.txt       <- Dependencies list
├── .gitignore             <- Excludes dataset binaries and temporary files
└── README.md              <- Project documentation & presentation guide
```

---

## 🚀 How to Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Modules Individually
- **Preprocessing (Unit 1 & 2)**:
  ```bash
  python src/preprocessing.py
  ```
- **Feature Extraction (Unit 3)**:
  ```bash
  python src/feature_extraction.py
  ```
- **Image Segmentation (Unit 6)**:
  ```bash
  python src/segmentation.py
  ```
- **Classification (Unit 4)**:
  ```bash
  python src/classification.py
  ```
- **Detection (Unit 5)**:
  ```bash
  python src/detection.py
  ```
- **Evaluation & Reporting**:
  ```bash
  python src/evaluate.py
  ```

---

## 💡 Viva Voce Q&A Cheat Sheet

**Q1: Why apply CLAHE histogram equalization instead of standard histogram equalization?**
> *Answer*: Standard histogram equalization operates globally and can over-amplify contrast in bright areas. CLAHE (Contrast Limited Adaptive Histogram Equalization) operates on localized $4\times4$ contextual regions and caps contrast enhancement, making it ideal for traffic signs under harsh sunlight or shadows.

**Q2: What is the benefit of HOG descriptors?**
> *Answer*: HOG (Histogram of Oriented Gradients) captures object appearance and shape by calculating edge gradient directions in localized cells. It is invariant to subtle illumination changes.

**Q3: Why does the CNN outperform the SVM classifier?**
> *Answer*: SVM relies on hand-crafted features (HOG) that cannot easily capture complex high-level spatial abstractions. CNN automatically learns multi-level hierarchical representations (edges $\rightarrow$ shapes $\rightarrow$ sign symbols) directly from raw pixel data.

**Q4: How does your simplified object detection pipeline work?**
> *Answer*: We use classical HSV color thresholding (red/blue sign colors) and contour aspect-ratio filtering to propose candidate bounding box regions, which are then passed into our trained CNN model for classification and confidence filtering.
