"""
src/evaluate.py
---------------
Model Evaluation & Reporting Module
  - Evaluates trained CNN model on test split.
  - Generates 43-Class Confusion Matrix plot (saved to results/confusion_matrix.png).
  - Generates Precision, Recall, F1-score classification report (saved to results/summary.txt).
  - Displays correct and incorrect sample predictions (saved to results/predictions_samples.png).
"""

import os
import sys

# Ensure parent project directory is in python path
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from src.preprocessing import CLASS_NAMES

RESULTS_DIR = 'results'
MODELS_DIR = 'models'
TEST_PATH = os.path.join('data', 'test', 'test_data.npz')
CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'traffic_sign_cnn.keras')
SVM_MODEL_PATH = os.path.join(MODELS_DIR, 'svm_baseline.pkl')


def plot_confusion_matrix(y_true, y_pred):
    """
    Generate and save confusion matrix heatmap for test set.
    """
    print("[Evaluation] Plotting Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(14, 12), facecolor='#111827')
    ax = plt.gca()
    ax.set_facecolor('#1f2937')

    sns.heatmap(
        cm, annot=False, fmt='d', cmap='mako', cbar=True,
        xticklabels=range(43), yticklabels=range(43)
    )

    plt.title('Traffic Sign CNN Model - 43-Class Confusion Matrix', color='white', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted Class ID', color='white', fontsize=12, labelpad=10)
    plt.ylabel('True Class ID', color='white', fontsize=12, labelpad=10)
    plt.tick_params(colors='white', labelsize=8)
    
    out_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix to {out_path}")


def plot_predictions_samples(X_test, y_true, y_pred, y_probs):
    """
    Display correctly classified vs incorrectly classified test image samples.
    """
    print("[Evaluation] Generating Prediction Sample visualizations...")

    correct_idx = np.where(y_true == y_pred)[0]
    incorrect_idx = np.where(y_true != y_pred)[0]

    n_show = 3
    fig, axes = plt.subplots(2, n_show, figsize=(12, 7), facecolor='#111827')

    # Row 0: Correct Predictions
    for j in range(n_show):
        idx = correct_idx[j]
        img = X_test[idx]
        true_lbl = CLASS_NAMES.get(y_true[idx], str(y_true[idx]))
        pred_lbl = CLASS_NAMES.get(y_pred[idx], str(y_pred[idx]))
        conf = y_probs[idx][y_pred[idx]] * 100

        axes[0, j].imshow(img)
        axes[0, j].set_title(f"CORRECT (Conf: {conf:.1f}%)\nTrue: {true_lbl[:18]}\nPred: {pred_lbl[:18]}",
                            color='#4ade80', fontsize=9, fontweight='bold')
        axes[0, j].axis('off')

    # Row 1: Incorrect Predictions (if available)
    for j in range(n_show):
        if j < len(incorrect_idx):
            idx = incorrect_idx[j]
            img = X_test[idx]
            true_lbl = CLASS_NAMES.get(y_true[idx], str(y_true[idx]))
            pred_lbl = CLASS_NAMES.get(y_pred[idx], str(y_pred[idx]))
            conf = y_probs[idx][y_pred[idx]] * 100

            axes[1, j].imshow(img)
            axes[1, j].set_title(f"MISCLASSIFIED (Conf: {conf:.1f}%)\nTrue: {true_lbl[:18]}\nPred: {pred_lbl[:18]}",
                                color='#f43f5e', fontsize=9, fontweight='bold')
        else:
            axes[1, j].axis('off')
        axes[1, j].axis('off')

    plt.suptitle("Model Prediction Samples (Top: Correct | Bottom: Misclassified)", color='white', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'predictions_samples.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved predictions sample grid to {out_path}")


def evaluate_models():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Test data missing at {TEST_PATH}. Run src/preprocessing.py first.")

    test_data = np.load(TEST_PATH)
    X_test_raw = test_data['X_test']
    y_test = test_data['y_test']

    X_test = X_test_raw.astype(np.float32) / 255.0

    print(f"\n[Evaluation] Loaded {len(X_test)} test samples.")

    if not os.path.exists(CNN_MODEL_PATH):
        raise FileNotFoundError(f"CNN model missing at {CNN_MODEL_PATH}. Run src/classification.py first.")

    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    y_probs = cnn_model.predict(X_test, verbose=0)
    y_pred_cnn = np.argmax(y_probs, axis=1)

    cnn_acc = accuracy_score(y_test, y_pred_cnn)
    print(f"  [CNN Model] Test Accuracy: {cnn_acc * 100:.2f}%")

    svm_acc_str = "N/A"
    if os.path.exists(os.path.join(RESULTS_DIR, 'model_acc_summary.txt')):
        try:
            with open(os.path.join(RESULTS_DIR, 'model_acc_summary.txt'), 'r') as f:
                content = f.read()
                for line in content.splitlines():
                    if 'SVM Baseline' in line:
                        svm_acc_str = line.split(':')[-1].strip()
        except Exception:
            pass

    plot_confusion_matrix(y_test, y_pred_cnn)
    plot_predictions_samples(X_test, y_test, y_pred_cnn, y_probs)

    report_str = classification_report(
        y_test, y_pred_cnn,
        target_names=[CLASS_NAMES.get(i, str(i)) for i in range(43)],
        digits=4
    )

    summary_content = f"""================================================================================
TRAFFIC SIGN DETECTION & CLASSIFICATION - COMPUTER VISION MINI-PROJECT SUMMARY
================================================================================

1. MODEL OVERALL PERFORMANCE:
   - CNN Model Test Accuracy:      {cnn_acc * 100:.2f}%
   - Classical SVM Baseline Acc:   {svm_acc_str}

2. EVALUATION METRICS BREAKDOWN (CNN MODEL):
Total Test Samples: {len(y_test)}

{report_str}
"""

    summary_path = os.path.join(RESULTS_DIR, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_content)

    print(f"  Saved evaluation report to {summary_path}")
    print("\n================ EVALUATION SUMMARY ================")
    print(f"CNN Model Test Accuracy: {cnn_acc * 100:.2f}%")
    print(f"SVM Baseline Accuracy:  {svm_acc_str}")
    print("====================================================")


if __name__ == '__main__':
    evaluate_models()
