# Hands-On Machine Learning - Chapter 3: Classification

This repository contains my practice, notes, and implementations for **Chapter 3 (Classification)** from Aurélien Géron's *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*.

## 📌 Topics Covered
- **MNIST Dataset Processing**: Loading 70,000 $28\times 28$ grayscale images and understanding dataset dimensions $(70000, 784)$.
- **Binary Classification**: Building a 5-Detector model using `SGDClassifier`.
- **Custom Cross-Validation**: Implementing `StratifiedKFold` manually and analyzing K-Fold results.
- **Accuracy Fallacy**: Understanding why accuracy is misleading on skewed datasets using a dummy `Never5Classifier`.
- **Performance Measures**:
  - **Confusion Matrix**: Generating clean predictions using `cross_val_predict()` and calculating True Negatives, False Positives, False Negatives, and True Positives.
  - **Precision & Recall**: Evaluating positive prediction accuracy vs. sensitivity.
  - **$F_1$ Score**: Computing the harmonic mean of precision and recall.
  - **Precision/Recall Tradeoff**: Accessing decision scores via `decision_function()`, plotting precision-recall curves, and adjusting decision thresholds (e.g., targeting 90% precision).
  - **ROC & ROC AUC**: Plotting TPR vs. FPR curves, evaluating ROC AUC scores, and comparing models (`SGDClassifier` vs. `RandomForestClassifier`).
- **Multiclass Classification**:
  - **OvA & OvO Strategies**: Understanding One-versus-All and One-versus-One strategies for binary-to-multiclass conversion (e.g., using `OneVsOneClassifier`).
  - **Native Multiclass Classifiers**: Evaluating multi-class decision probabilities using `predict_proba()` with `RandomForestClassifier`.
  - **Feature Scaling Impact**: Improving `SGDClassifier` accuracy to over 90% using `StandardScaler`.
- **Error Analysis**:
  - Analyzing normalized confusion matrices to isolate specific misclassifications (e.g., false 8s).
  - Investigating pixel-level errors and understanding linear model limitations on shifted/rotated inputs (e.g., 3 vs. 5 confusion).
- **Multilabel Classification**:
  - Training `KNeighborsClassifier` to output multiple binary tags per instance (e.g., checking if a digit is $\ge 7$ and odd).
  - Evaluating predictions using macro and weighted $F_1$ scores.
- **Multioutput Classification**:
  - Building an image denoising system that takes noisy MNIST images as input and predicts 784 pixel intensity values ($0-255$) simultaneously to reconstruct clean images.

## 🛠️ Requirements
- Python 3.10+
- `scikit-learn`
- `numpy`
- `matplotlib`
- `jupyter`