# 🚦 Traffic Sign Detection & Classification

A comprehensive Computer Vision and Deep Learning system for traffic sign recognition, segmentation, and bounding-box localization using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

This project implements a multi-stage computer vision pipeline combining digital image processing, feature extraction, traditional machine learning (SVM), and deep convolutional neural networks (CNN) to detect and classify 43 traffic sign categories.

---

## 🎯 Project Overview & Objectives

Traffic sign recognition is an essential component of **Autonomous Driving Systems (ADS)** and **Advanced Driver Assistance Systems (ADAS)**. Real-world road conditions present severe challenges including fluctuating outdoor illumination, shadows, adverse weather, motion blur, and partial occlusions.

The objective of this project is to build an end-to-end computer vision system capable of:
1. Enhancing low-contrast road imagery under varied lighting conditions.
2. Extracting structural geometric edges, gradient orientation features, and color segmentations.
3. Classifying traffic sign categories using both traditional machine learning (SVM) and deep learning (CNN).
4. Localizing traffic signs within full road scenes using candidate region proposals and bounding boxes.
5. Evaluating multi-class classification performance using quantitative metrics and visual confusion matrices.

---

## 🛠️ Technology Stack & Tools Used

- **Programming Language**: Python 3.12
- **Computer Vision & Image Processing**: OpenCV (`cv2`), `scikit-image`
- **Deep Learning Framework**: TensorFlow 2.16 / Keras 3.3
- **Machine Learning & Statistics**: `scikit-learn` (SVM, PCA, metrics, dataset splitting)
- **Data Manipulation**: NumPy, Pandas
- **Data Visualization**: Matplotlib, Seaborn

---

## 📊 Dataset Description

- **Dataset**: German Traffic Sign Recognition Benchmark (GTSRB)
- **Total Categories**: 43 traffic sign classes (Speed Limit 20/30/50/60/70/80/100/120 km/h, Yield, Stop, No Entry, Road Work, Pedestrians, Children Crossing, Traffic Signals, Roundabout, etc.)
- **Total Images**: Over 50,000 cropped traffic sign images
- **Input Resolution**: Standardized and resized to $32 \times 32 \times 3$ RGB pixels
- **Data Splits**: 80% Training ($31,367$ samples), 10% Validation ($3,921$ samples), and 10% Testing ($12,630$ samples)

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

## 🔬 Detailed Implementation & Methodology

### 1. Data Preprocessing & Contrast Enhancement (`src/preprocessing.py`)
- **Fixed Resizing**: Standardizes raw input images of varying sizes to a uniform $32 \times 32 \times 3$ resolution.
- **Pixel Normalization**: Scales uint8 pixel values from $[0, 255]$ to float32 values in $[0.0, 1.0]$.
- **CLAHE Contrast Enhancement**: Traffic sign images captured in dark shadows or under bright glare are converted from RGB to the **YCrCb** color space. Contrast Limited Adaptive Histogram Equalization (CLAHE) with a clip limit of $2.0$ and tile grid size of $4 \times 4$ is applied exclusively to the luminance ($Y$) channel before recombining:
  $$I_{\text{CLAHE}} = \text{CLAHE}(Y), \quad \text{recombined with } Cr, Cb$$
- **Dataset Partitioning**: Performs stratified splitting to ensure balanced class distributions across Train, Validation, and Test sets.

### 2. Feature Detection & Extraction (`src/feature_extraction.py`)
- **Canny Edge Detection**: Detects structural contours and boundary geometries (triangles, circles, octagons) by applying Gaussian blurring followed by dual-threshold gradient intensity evaluation.
- **Histogram of Oriented Gradients (HOG)**: Captures local edge direction distributions across $8 \times 8$ pixel cells using 9 orientation bins and $2 \times 2$ block normalization, generating a robust feature vector invariant to minor illumination changes.
- **Principal Component Analysis (PCA)**: Flattens image pixel matrices ($3072$ dimensions) and applies PCA to project features into $2\text{D}$ space, enabling visual scatter plot analysis of class separability and total variance.

### 3. Image Segmentation (`src/segmentation.py`)
- **HSV Color Thresholding**: Converts images to HSV space and applies color boundaries to isolate red (Hue $[0, 10] \cup [170, 180]$) and blue (Hue $[100, 130]$) sign regions.
- **Watershed Segmentation**: Applies Otsu binary thresholding, distance transform, and morphological opening/closing to generate foreground/background markers, utilizing the Watershed algorithm to extract exact sign boundary contours.

### 4. Classification Models (`src/classification.py`)
- **Classical Machine Learning Baseline**: Trains a Linear Support Vector Machine (SVM) on extracted HOG descriptors, serving as a traditional CV benchmark.
- **4-Layer Convolutional Neural Network (CNN)**:
  - **Input Layer**: $32 \times 32 \times 3$ RGB input.
  - **Conv Block 1**: Two $3 \times 3$ Conv2D layers (32 filters, ReLU) + BatchNormalization + MaxPool(2x2) + Dropout(0.25).
  - **Conv Block 2**: Two $3 \times 3$ Conv2D layers (64 filters, ReLU) + BatchNormalization + MaxPool(2x2) + Dropout(0.30).
  - **Dense Classifier**: Flatten + Dense(256, ReLU) + BatchNormalization + Dropout(0.50) + Dense(43, Softmax).
  - **Optimization**: Adam optimizer with Sparse Categorical Crossentropy loss and Early Stopping.

### 5. Object Detection & Localization (`src/detection.py`)
- **Candidate Region Proposal**: Scans road scenes using HSV color masking to detect potential traffic sign regions.
- **Contour Filtering**: Filters candidate bounding boxes based on minimum surface area ($800 \le \text{Area} \le 15,000$) and realistic aspect ratios ($0.65 \le w/h \le 1.45$).
- **CNN Verification & Bounding Box Annotation**: Crops candidate regions, normalizes them, and feeds them into the trained CNN model. Draws bounding boxes with class labels and confidence percentages if confidence exceeds threshold ($>30\%$).

### 6. Evaluation & Reporting (`src/evaluate.py`)
- Computes overall test set accuracy for both CNN and SVM baseline models.
- Generates a $43 \times 43$ multi-class Confusion Matrix heatmap.
- Computes Precision, Recall, and F1-score classification metrics saved to `results/summary.txt`.
- Plots a visual grid comparing correctly classified vs. misclassified test sign predictions.

---

## 📈 Experimental Performance & Results

| Model | Architecture / Representation | Test Accuracy | Output File Location |
| :--- | :--- | :---: | :--- |
| **Linear SVM Baseline** | Hand-crafted HOG Descriptors | **75.40%** | `models/svm_baseline.pkl` |
| **Deep Learning CNN** | 4-Layer ConvNet + BatchNorm + Dropout | **96.91%** | `models/traffic_sign_cnn.keras` |

### Key Performance Insights
- **SVM Baseline (75.40%)**: Demonstrates that hand-crafted gradient features provide a solid baseline but struggle with fine-grained intra-class variation across 43 classes.
- **CNN Model (96.91%)**: Achieves high accuracy due to learned spatial feature hierarchies that dynamically adjust to color, shape, and textual sign patterns.

---

## 🖼️ Visual Artifacts Gallery

All generated visual plots are saved in `results/` using a dark-mode palette:

- `results/class_distribution.png`: Bar chart displaying sample frequency across 43 GTSRB categories.
- `results/histogram_eq_comparison.png`: CLAHE contrast enhancement comparison grid.
- `results/edge_detection_samples.png`: Canny edge detector visual outputs.
- `results/hog_visualization.png`: HOG gradient orientation maps.
- `results/pca_visualization.png`: 2D PCA scatter plot showing class clustering.
- `results/segmentation_samples.png`: HSV color mask & Watershed region segmentation outputs.
- `results/training_curves.png`: Training vs. Validation accuracy and loss curves over epochs.
- `results/detection_samples.png`: Annotated bounding boxes with predicted class and confidence.
- `results/confusion_matrix.png`: Multi-class $43 \times 43$ confusion matrix heatmap.
- `results/predictions_samples.png`: Correct vs. misclassified sample predictions.
- `results/summary.txt`: Precision, Recall, and F1-score metric breakdown.

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
└── README.md              <- Project documentation
```

---

## 🚀 Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Scripts Individually
Each module can be executed independently:

```bash
# 1. Preprocessing & Contrast Enhancement
python src/preprocessing.py

# 2. Feature Extraction (Canny, HOG, PCA)
python src/feature_extraction.py

# 3. Image Segmentation (HSV & Watershed)
python src/segmentation.py

# 4. Model Training (SVM & CNN)
python src/classification.py

# 5. Object Detection & Localization
python src/detection.py

# 6. Model Evaluation & Confusion Matrix
python src/evaluate.py
```
