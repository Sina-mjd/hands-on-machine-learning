# Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow 🤖📈

Welcome to my study and implementation repository for the book **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** (by Aurélien Géron). 

This repository serves as a comprehensive workspace containing clear code implementations, mathematical derivations, end-to-end Machine Learning pipelines, and structured notes for each chapter.

---

## 📌 Main Objectives
* **Theory to Practice:** Translating mathematical models (algebra, calculus, optimization) into clean Python code.
* **Scikit-Learn Mastery:** Building robust feature engineering, preprocessing, and training pipelines.
* **Algorithmic Insight:** Implementing core ML algorithms both **from scratch** (NumPy) and using industry-standard libraries.
* **Reproducibility:** Organizing clean code structures and documentation for future reference.

---

## 📂 Repository Structure

Below is an overview of the modules covered in this repository:

| Chapter | Topic | Key Concepts & Models | Status |
| :---: | :--- | :--- | :---: |
| **01** | **The Machine Learning Landscape** | Supervised vs Unsupervised, Overfitting, Generalization | 🟢 Completed |
| **02** | **End-to-End Machine Learning Project** | Pipelines, Feature Scaling, Cross-Validation, GridSearch | 🟢 Completed |
| **03** | **Classification** | MNIST, Precision/Recall Tradeoff, ROC-AUC, Multiclass | 🟢 Completed |
| **04** | **Training Models** | Normal Equation, SVD, BGD, SGD, Mini-batch, Polynomial Reg. | 🟡 In Progress |
| **05** | **Support Vector Machines** | Linear/Non-linear SVM, Soft Margin, Kernel Trick (RBF) | ⏳ Upcoming |
| **06** | **Decision Trees** | CART Algorithm, Gini Impurity, Entropy, Regularization | ⏳ Upcoming |
| **07** | **Ensemble Learning & Random Forests** | Bagging, Boosting (AdaBoost, XGBoost), Stacking | ⏳ Upcoming |
| **08** | **Dimensionality Reduction** | PCA, Kernel PCA, t-SNE, Manifold Learning | ⏳ Upcoming |

---

## 🔬 Chapter 4 Spotlight: Training Models

A deep dive into how linear and non-linear models actually learn parameters:

* **Closed-Form Solutions:** Analytical approaches using **Normal Equation** and **SVD (Pseudoinverse)**.
* **Gradient Descent Optimization:**
  * **Batch GD:** Computing exact gradients over the full dataset ($O(m)$ memory scaling).
  * **Stochastic GD (SGD):** Ultra-fast iteration with 1 instance, handling Out-of-Core learning.
  * **Mini-batch GD:** Matrix hardware acceleration (GPU optimized) balancing stability & speed.
* **Non-linear Data Fitting:** Feature expansion via **Polynomial Regression** and handling combinatorial explosion.
* **Model Diagnosis:** Analyzing **Bias-Variance Tradeoff** and diagnosing model performance using **Learning Curves**.

---

## 🛠️ Tech Stack & Tools

* **Language:** Python 3.x
* **Numerical & Data Processing:** `NumPy`, `Pandas`
* **Machine Learning & Preprocessing:** `Scikit-Learn`
* **Visualization:** `Matplotlib`, `Seaborn`
* **Environment:** Jupyter Notebooks / PyCharm / VS Code

---

## 🚀 How to Use This Repository

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/hands-on-machine-learning.git](https://github.com/YOUR_USERNAME/hands-on-machine-learning.git)
   cd hands-on-machine-learning
