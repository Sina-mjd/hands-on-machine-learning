# Chapter 5: Support Vector Machines (SVM) 🚀

This repository contains my implementations and notes for **Chapter 5** of *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. It explores the mathematical logic and practical applications of Support Vector Machines for classification tasks.

## 📌 Core Concepts Explored

*   **Large Margin Classification:** Understanding how SVMs find the widest possible "street" to separate classes, making models generalize better to unseen data.
*   **Soft vs. Hard Margin:** Balancing the street width and margin violations using the `C` hyperparameter.
*   **Feature Scaling:** Demonstrating the critical importance of `StandardScaler` in distance-based algorithms like SVM.
*   **Nonlinear Classification:** Handling complex datasets (like the Moons dataset) by projecting data into higher dimensions.
*   **The Kernel Trick:** Implementing Polynomial and Gaussian RBF kernels to achieve high-dimensional mapping without the massive computational cost of adding actual features.

## 🎛️ Hyperparameter Tuning Cheat Sheet

Through experimentation, I analyzed the behavior of key SVM parameters:

| Hyperparameter | Behavior when Increased | Regularization Effect |
| :--- | :--- | :--- |
| **`C`** | Narrower margin, fewer violations | Less Regularization (High Variance) |
| **`gamma` ($\gamma$)** | Narrower RBF bell curve, highly irregular decision boundary | Less Regularization (High Variance) |

> **Rule of Thumb:** If the model is overfitting, decrease `C` and/or `gamma`. If it's underfitting, increase them.

## 🔬 Practical Observations: Global vs. Local Kernels

During testing on the `make_moons` dataset, I inputted an extreme outlier point `[7, 2]` to observe how different kernels handle distant, unseen data:

1.  **Polynomial Kernel (Global Behavior):** Predicted `Class 1`. Polynomial equations extend infinitely, meaning the algorithm divides the entire 2D space with a continuous boundary. Every point, no matter how far, falls on one side of this dividing wall.
2.  **Gaussian RBF Kernel (Local Behavior):** Predicted `Class 0` (the background/default class). RBF uses bell-shaped similarity functions around training instances (landmarks). The point `[7, 2]` was so far from any training data that its similarity score dropped to zero, isolating the learned pattern to a local "island."

## ⏱️ Computational Complexity Comparison

To handle different dataset sizes efficiently, I evaluated Scikit-Learn's SVM classes:

| Class | Time Complexity | Out-of-core Support | Kernel Trick | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| `LinearSVC` | $O(m \times n)$ | No | No | Large datasets with many features |
| `SGDClassifier` | $O(m \times n)$ | Yes | No | Huge datasets (won't fit in RAM) |
| `SVC` | $O(m^2 \times n)$ to $O(m^3 \times n)$ | No | Yes | Small/Medium complex datasets |

## 🛠️ How to Run

1. Activate the virtual environment:
   ```bash
   source ../env/bin/activate