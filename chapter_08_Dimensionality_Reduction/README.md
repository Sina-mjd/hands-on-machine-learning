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

## Advanced Dimensionality Reduction & Manifold Learning

The second part of this chapter explores techniques for handling massive datasets that don't fit in memory, as well as Nonlinear Dimensionality Reduction (NLDR) algorithms designed to unroll complex, twisted manifolds.

### Core Concepts Covered

* **PCA for Compression:** Utilizing PCA to compress datasets (e.g., reducing MNIST to 154 dimensions while retaining 95% variance) and measuring the **Reconstruction Error** via inverse transformation.
* **Algorithmic Optimizations:**
  * **Randomized PCA:** A stochastic algorithm that drastically speeds up computation when the target dimensionality $d$ is much smaller than $n$.
  * **Incremental PCA (IPCA):** Solving the Out-of-Core learning problem. Implemented mini-batch processing using `np.array_split` and memory-mapped files (`np.memmap`) for datasets larger than the available RAM.
* **Nonlinear Dimensionality Reduction (NLDR):**
  * **Kernel PCA (kPCA):** Applying the Kernel Trick to perform complex nonlinear projections. 
  * **Hyperparameter Tuning for kPCA:** Implemented automated tuning using `GridSearchCV` and pipelines, focusing on the **Reconstruction Pre-image Error** for unsupervised evaluation.
  * **Locally Linear Embedding (LLE):** A manifold learning technique that unrolls datasets (like the Swiss Roll) by modeling and preserving local neighborhood relationships rather than global variance.
* **Other Techniques Explored:** A theoretical overview of MDS, Isomap, t-SNE (highly effective for cluster visualization), and LDA (a classification algorithm used for discriminative projection).

### Project Structure (Part 2)

* `pca_compression.py`: Image compression and reconstruction using PCA on the MNIST dataset.
* `incremental_pca.py`: Out-of-core PCA implementation using mini-batches and `np.memmap`.
* `kernel_pca.py`: Nonlinear dimensionality reduction using RBF and Sigmoid kernels.
* `kpca_tuning.py`: Pipeline integration and hyperparameter tuning using Grid Search and pre-image error calculation.
* `lle_manifold.py`: Unrolling the 3D Swiss Roll dataset into a 2D space using Locally Linear Embedding.

## Requirements

```bash
pip install numpy scikit-learn matplotlib