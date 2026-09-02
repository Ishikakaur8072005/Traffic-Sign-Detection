"""
src/classification.py
---------------------
Unit 4: Image Classification
  - Classical Machine Learning Baseline: SVM (Support Vector Machine) trained on HOG features.
  - Deep Learning Model: 4-Layer Convolutional Neural Network (CNN) in Keras/TensorFlow.
  - Generates training loss & accuracy curves (saved to results/training_curves.png).
  - Saves trained models to models/ directory.
"""

import os
import sys

# Ensure UTF-8 output encoding for Windows terminal compatibility
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

TRAIN_PATH = os.path.join('data', 'train', 'train_data.npz')
TEST_PATH = os.path.join('data', 'test', 'test_data.npz')
RESULTS_DIR = 'results'
MODELS_DIR = 'models'

NUM_CLASSES = 43
IMG_SHAPE = (32, 32, 3)


def train_classical_ml_baseline(X_train, y_train, X_test, y_test):
    """
    Unit 4: Train an SVM baseline classifier using HOG features.
    """
    print("\n[Unit 4] Training Classical ML Baseline (SVM on HOG features)...")

    n_samples = min(5000, len(X_train))
    idx = np.random.choice(len(X_train), n_samples, replace=False)
    X_sub, y_sub = X_train[idx], y_train[idx]

    def extract_hog_features(images):
        hog_features = []
        for img in images:
            img_ubyte = img if img.dtype == np.uint8 else (img * 255).astype(np.uint8)
            gray = cv2.cvtColor(img_ubyte, cv2.COLOR_RGB2GRAY)
            feat = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=None)
            hog_features.append(feat)
        return np.array(hog_features)

    print(f"  Extracting HOG features from {n_samples} training samples...")
    X_train_hog = extract_hog_features(X_sub)
    
    n_test_sub = min(1500, len(X_test))
    test_idx = np.random.choice(len(X_test), n_test_sub, replace=False)
    X_test_hog = extract_hog_features(X_test[test_idx])
    y_test_sub = y_test[test_idx]

    svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    svm.fit(X_train_hog, y_sub)

    y_pred_svm = svm.predict(X_test_hog)
    svm_acc = accuracy_score(y_test_sub, y_pred_svm)
    print(f"  [SVM Baseline] Test Accuracy: {svm_acc * 100:.2f}%")

    model_path = os.path.join(MODELS_DIR, 'svm_baseline.pkl')
    joblib.dump(svm, model_path)
    print(f"  Saved SVM model to {model_path}")
    return svm_acc


def build_cnn_model():
    """
    Unit 4: Build a small, explainable 4-layer Convolutional Neural Network.
    """
    model = models.Sequential([
        layers.Input(shape=IMG_SHAPE),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Dense Classifier Block
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_training_curves(history):
    """
    Unit 4: Plot and save training vs validation accuracy and loss curves.
    """
    print("[Unit 4] Generating CNN training curves plot...")
    epochs = range(1, len(history.history['accuracy']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#111827')

    axes[0].set_facecolor('#1f2937')
    axes[0].plot(epochs, history.history['accuracy'], 'o-', label='Train Accuracy', color='#38bdf8', linewidth=2)
    axes[0].plot(epochs, history.history['val_accuracy'], 's-', label='Val Accuracy', color='#4ade80', linewidth=2)
    axes[0].set_title('CNN Accuracy Curves (Unit 4)', color='white', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epochs', color='white', fontsize=11)
    axes[0].set_ylabel('Accuracy', color='white', fontsize=11)
    axes[0].tick_params(colors='white')
    axes[0].legend(facecolor='#111827', edgecolor='none', labelcolor='white')
    axes[0].grid(alpha=0.2, color='#4b5563')

    axes[1].set_facecolor('#1f2937')
    axes[1].plot(epochs, history.history['loss'], 'o-', label='Train Loss', color='#f43f5e', linewidth=2)
    axes[1].plot(epochs, history.history['val_loss'], 's-', label='Val Loss', color='#fbbf24', linewidth=2)
    axes[1].set_title('CNN Loss Curves (Unit 4)', color='white', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epochs', color='white', fontsize=11)
    axes[1].set_ylabel('Loss', color='white', fontsize=11)
    axes[1].tick_params(colors='white')
    axes[1].legend(facecolor='#111827', edgecolor='none', labelcolor='white')
    axes[1].grid(alpha=0.2, color='#4b5563')

    plt.suptitle("Unit 4: Image Classification - CNN Model Training Metrics", color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'training_curves.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curves to {out_path}")


def train_cnn(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Unit 4: Train CNN model and evaluate on validation/test sets.
    """
    print("\n[Unit 4] Training CNN (Keras / TensorFlow)...")
    cnn_model = build_cnn_model()

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = cnn_model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=2
    )

    plot_training_curves(history)

    test_loss, test_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
    print(f"  [CNN Model] Final Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")

    cnn_model_path = os.path.join(MODELS_DIR, 'traffic_sign_cnn.keras')
    cnn_model.save(cnn_model_path)
    print(f"  Saved trained CNN model to {cnn_model_path}")
    return test_acc


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):
        train_data = np.load(TRAIN_PATH)
        test_data = np.load(TEST_PATH)
        
        X_train = train_data['X_train'].astype(np.float32) / 255.0
        y_train = train_data['y_train']
        X_val = train_data['X_val'].astype(np.float32) / 255.0
        y_val = train_data['y_val']
        X_test = test_data['X_test'].astype(np.float32) / 255.0
        y_test = test_data['y_test']
    else:
        raise FileNotFoundError("Training data not found. Run src/preprocessing.py first.")

    svm_acc = train_classical_ml_baseline(X_train, y_train, X_test, y_test)
    cnn_acc = train_cnn(X_train, y_train, X_val, y_val, X_test, y_test)

    with open(os.path.join(RESULTS_DIR, 'model_acc_summary.txt'), 'w') as f:
        f.write(f"SVM Baseline Accuracy: {svm_acc * 100:.2f}%\n")
        f.write(f"CNN Model Accuracy:    {cnn_acc * 100:.2f}%\n")


if __name__ == '__main__':
    main()
