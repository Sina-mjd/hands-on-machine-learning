# Chapter 7: Ensemble Learning and Random Forests

This repository contains comprehensive implementations, theoretical notes, and experimentation scripts for **Chapter 7: Ensemble Learning and Random Forests** from Aurélien Géron's *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"*. The focus centers on building robust predictive models by combining multiple weak learners into powerful, high-performance meta-predictors.

## Core Algorithms Implemented

| Algorithm / Technique | Key Concept & Hyperparameters | Primary Use Case |
| :--- | :--- | :--- |
| **Voting Classifiers** | Combines diverse models via Hard (majority rule) or Soft (probability-based) voting. | Baseline ensemble performance enhancement. |
| **Bagging & Pasting** | Bootstrap sampling (Bagging) vs. disjoint sampling (Pasting) with Out-Of-Bag (OOB) evaluation. | Reducing model variance and preventing overfitting. |
| **Random Patches & Subspaces** | Simultaneous instance and feature sampling (`max_features`, `bootstrap_features`). | High-dimensional data and computer vision inputs. |
| **Random Forests & Extra-Trees** | Ensembles of decision trees with feature subsetting and randomized split thresholds. | Robust non-linear classification and feature selection. |
| **AdaBoost** | Sequential training focusing iteratively on misclassified instance weights. | Adaptive sequential error correction. |
| **Gradient Boosting (GBRT)** | Fits new predictors sequentially to the residual errors of their predecessors. | Regression tasks and structured tabular optimization. |
| **XGBoost** | Highly scalable, optimized gradient boosting framework with built-in early stopping. | High-performance and competitive machine learning. |
| **Stacking (Blending)** | Multi-layer architecture using a meta-learner (blender) to aggregate base predictions. | Maximizing predictive power via stacked generalization. |

## Key Technical Highlights

* **API Modernization & Version Control:** Addressed and resolved modern Scikit-Learn and XGBoost API updates, such as migrating `early_stopping_rounds` directly into the `XGBoost` constructor and removing legacy parameters.
* **Early Stopping Strategies:** Implemented retrospective validation evaluation (`staged_predict`) alongside incremental training loops (`warm_start=True`) to prevent gradient boosting overfitting.
* **Feature Importance Profiling:** Extracted and visualized impurity-based feature importances (`feature_importances_`) using Pandas and Matplotlib to execute data-driven feature selection.

## Repository Structure

```text
├── voting_classifiers.py       # Hard and Soft voting implementations
├── bagging_oob.py              # Bagging, Pasting, and OOB evaluation
├── random_forests.py           # Random Forests and Feature Importance plots
├── boosting_adaboost.py        # AdaBoost with Decision Stumps
├── gradient_boosting_early.py  # GBRT, Warm-Start, and XGBoost integration
└── stacking_blender.py         # Manual multi-layer stacking architecture