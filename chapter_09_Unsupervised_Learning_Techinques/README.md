## Chapter 9: Unsupervised Learning Techniques (Part 1)

This section marks the transition from supervised learning to the challenges of raw, unlabeled data, exploring the foundations of Unsupervised Learning.

### Core Concepts Explored
* **The "Cake" Analogy:** Understanding the immense potential and necessity of unsupervised learning in real-world scenarios where labeling is expensive.
* **Clustering vs. Classification:** Implementing algorithms to discover hidden structures and groupings in data without predefined labels.
* **K-Means Algorithm:**
  * **Core Mechanics:** Centroid initialization, instance assignment, and iterative centroid updates.
  * **Hard vs. Soft Clustering:** Using `predict()` for cluster assignment vs. `transform()` for non-linear dimensionality reduction based on distances to all centroids.
  * **Initialization Strategies:** Mitigating sub-optimal convergence risks using the **K-Means++** algorithm.
* **Mini-Batch K-Means:** Implementing Sculley's algorithm to accelerate clustering on large datasets (Out-of-Core learning capabilities).
* **Evaluating Cluster Quality:** 
  * Calculating **Inertia** (sum of squared distances).
  * Implementing the **Elbow Rule** to estimate the optimal number of clusters ($k$).
  * Computing the **Silhouette Score** for a more mathematically rigorous evaluation of cluster density and separation.

### Project Files
* `kmeans_basics.py`: Synthetic dataset generation (`make_blobs`) and core K-Means implementation with visualizations.
* `minibatch_kmeans.py`: Accelerated clustering for large-scale datasets.
* `cluster_evaluation.py`: Mathematical evaluation of models using Inertia curves and Silhouette scores to find the optimal $k$.