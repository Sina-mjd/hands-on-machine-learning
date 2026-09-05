# Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow 🤖📈

Welcome to my study and implementation repository for the book **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** (by Aurélien Géron). 

This repository serves as a comprehensive workspace containing clear code implementations, mathematical derivations, end-to-end Machine Learning pipelines, and structured notes for each chapter. It is designed to showcase practical AI engineering skills, from foundational ML algorithms to advanced Deep Learning models.

---

## 📌 Main Objectives
* **Theory to Practice:** Translating mathematical models (algebra, calculus, optimization) into clean Python code.
* **Scikit-Learn Mastery:** Building robust feature engineering, preprocessing, and training pipelines.
* **Algorithmic Insight:** Implementing core ML algorithms both **from scratch** (NumPy) and using industry-standard libraries.
* **Reproducibility:** Organizing clean code structures and documentation for future reference and portfolio building.

---

## 📂 Repository Structure

Below is an overview of the modules covered in this repository:

| Chapter | Topic | Key Concepts & Models | Status |
| :---: | :--- | :--- | :---: |
| **01** | **The Machine Learning Landscape** | Supervised vs Unsupervised, Overfitting, Generalization | 🟢 Completed |
| **02** | **End-to-End Machine Learning Project** | Pipelines, Feature Scaling, Cross-Validation, GridSearch | 🟢 Completed |
| **03** | **Classification** | MNIST, Precision/Recall Tradeoff, ROC-AUC, Multiclass | 🟢 Completed |
| **04** | **Training Models** | Normal Equation, SVD, BGD, SGD, Mini-batch, Polynomial Reg. | 🟢 Completed |
| **05** | **Support Vector Machines** | Linear/Non-linear SVM, Soft Margin, Kernel Trick (RBF) | 🟢 Completed |
| **06** | **Decision Trees** | CART Algorithm, Gini/Entropy, Regularization, Regression Trees | 🟢 Completed |
| **07** | **Ensemble Learning & Random Forests** | Bagging, Boosting (AdaBoost, XGBoost), Stacking | ⏳ Upcoming |
| **08** | **Dimensionality Reduction** | PCA, Kernel PCA, t-SNE, Manifold Learning | ⏳ Upcoming |

---

## 🔬 Latest Spotlight: Chapter 6 - Decision Trees

A deep dive into building interpretable models for classification and regression tasks:

* **White-Box Interpretability:** Visualizing decision rules step-by-step without requiring feature scaling or centering.
* **The CART Algorithm:** Understanding how Scikit-Learn builds binary trees using a greedy approach to minimize Gini Impurity and Mean Squared Error.
* **Regularization Strategies:** Preventing severe overfitting by constraining degrees of freedom using hyperparameters like `max_depth` and `min_samples_leaf`.
* **Addressing Instability:** Handling the model's sensitivity to rotation and high variance by utilizing PCA pipelines and preparing for Ensemble combinations.

---

## 🛠️ Tech Stack & Tools

* **Language:** Python 3.x
* **Numerical & Data Processing:** `NumPy`, `Pandas`
* **Machine Learning:** `Scikit-Learn`
* **Deep Learning Frameworks:** `TensorFlow`, `Keras`, `PyTorch`
* **Computer Vision:** `OpenCV`
* **Visualization:** `Matplotlib`, `Seaborn`
* **Environment & Databases:** Jupyter Notebooks, Docker, SQL, MongoDB, VS Code

---

## 🚀 How to Use This Repository

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/hands-on-machine-learning.git](https://github.com/YOUR_USERNAME/hands-on-machine-learning.git)
   cd hands-on-machine-learning
