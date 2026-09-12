# Chapter 8: Dimensionality Reduction

This repository contains the implementation, experimentation, and theoretical notes for **Chapter 8: Dimensionality Reduction** from Aurélien Géron's *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"*.

## Overview

High-dimensional datasets introduce severe challenges in machine learning, commonly referred to as the **Curse of Dimensionality**. This phenomenon causes data sparsity, increases the risk of extreme overfitting, and exponentially slows down model training. This chapter focuses on transforming intractable high-dimensional problems into tractable ones while preserving the maximum amount of underlying information.

## Core Concepts Covered

* **The Curse of Dimensionality:** Understanding spatial sparsity and the geometric behavior of high-dimensional hypercubes.
* **Dimensionality Reduction Approaches:**
  * **Projection:** Projecting data onto lower-dimensional hyperplanes (effective for linear subspaces).
  * **Manifold Learning:** Unrolling complex, twisted datasets (e.g., the Swiss Roll dataset) based on the Manifold Hypothesis.

## Principal Component Analysis (PCA)

Implemented the most popular dimensionality reduction algorithm, **PCA**, from scratch using Linear Algebra and via Scikit-Learn's optimized API.

### Key Implementations & Techniques

1. **Mathematical Core (NumPy):** 
   * Centering the dataset algebraically.
   * Executing **Singular Value Decomposition (SVD)** to extract orthogonal Principal Components (PCs).
   * Calculating matrix dot products to project 3D data down to a 2D subspace.
2. **Scikit-Learn Integration:** 
   * Utilizing `sklearn.decomposition.PCA` for automated centering and projection (`fit_transform`).
3. **Variance Analysis:** 
   * Extracting and interpreting the `explained_variance_ratio_` to quantify information retention.
4. **Dynamic Dimension Selection:** 
   * Automating the selection of dimensions to preserve a specific variance threshold (e.g., `n_components=0.95`).
   * Visualizing the **Intrinsic Dimensionality** using cumulative explained variance plots (the "Elbow Curve").

## Project Structure

* `manual_pca_svd.py`: Raw NumPy implementation of SVD-based PCA and geometric projection.
* `sklearn_pca.py`: Streamlined PCA implementation using Scikit-Learn.
* `pca_variance_analysis.py`: Scripts for determining optimal component counts and plotting variance preservation curves.

## Requirements

```bash
pip install numpy scikit-learn matplotlib