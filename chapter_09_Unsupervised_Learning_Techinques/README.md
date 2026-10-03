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

## Advanced Applications of K-Means Clustering

This section focuses on practical, real-world applications of the K-Means algorithm, demonstrating its versatility beyond simple data grouping.

### Core Implementations

* **Image Segmentation (Color-Based):**
  * Transformed 3D image arrays (RGB) into 2D pixel lists to apply K-Means.
  * Clustered pixels by color intensity and reconstructed the image using cluster centroids.
  * Demonstrated the algorithm's sensitivity to cluster sizes (e.g., small flashy objects losing their distinct clusters at low $k$ values).

* **Clustering as a Preprocessing Step:**
  * Integrated `KMeans` into a Scikit-Learn `Pipeline` as a feature extractor before a `LogisticRegression` classifier.
  * Converted raw pixel features (Digits dataset) into non-linear distance features to cluster centroids.
  * Addressed `ConvergenceWarning` and scale disparities by introducing `StandardScaler` into the pipeline.
  * Utilized `GridSearchCV` to automatically find the optimal number of clusters ($k$) based on downstream classification accuracy, significantly outperforming the baseline model.

* **Semi-Supervised Learning (Solving the Missing Label Problem):**
  * Tackled a scenario with only 50 labeled instances out of a large dataset.
  * **Representative Instances:** Clustered the data and identified the instances closest to the 50 centroids to act as human-labeled representatives.
  * **Label Propagation:** Automatically propagated the labels from the representatives to the rest of the dataset.
  * **Partial Propagation (Noise Reduction):** Improved accuracy to 94.2% (matching fully supervised performance) by propagating labels *only* to the 20% of instances closest to the centroids, filtering out noisy boundary data.
  * **Active Learning:** Explored the theoretical concept of Uncertainty Sampling to iteratively query human experts for the hardest-to-classify instances.

### Project Files
* `image_segmentation.py`: Image pixel flattening, color clustering, and image reconstruction.
* `preprocessing_pipeline.py`: K-Means integration with `StandardScaler`, `LogisticRegression`, and hyperparameter tuning via `GridSearchCV`.
* `semi_supervised.py`: Representative instance extraction, label propagation algorithms, and partial propagation filtering.

## Advanced Clustering: Density and Probabilistic Models

This section explores advanced clustering algorithms designed to overcome the limitations of K-Means, particularly its assumption of spherical clusters and uniform cluster sizes.

### Core Algorithms Explored

* **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**
  * **Mechanism:** Identifies clusters as continuous regions of high density, defined by a neighborhood radius (`eps`) and a minimum number of points (`min_samples`).
  * **Strengths:** Capable of finding clusters of arbitrary shapes (e.g., the Moons dataset) and highly robust to outliers/noise (labeled as `-1`).
  * **Implementation Note:** DBSCAN lacks a `predict()` method. Implemented a workaround by training a `KNeighborsClassifier` solely on the identified core instances to predict cluster assignments for new data points.
  * **Limitations:** Struggles with datasets containing clusters of varying densities.

* **Gaussian Mixture Models (GMM):**
  * **Mechanism:** A probabilistic, generative model that assumes data is generated from a mixture of several Gaussian distributions.
  * **Expectation-Maximization (EM):** Uses the EM algorithm for soft clustering, estimating the probability (responsibilities) of each instance belonging to each cluster, and iteratively updating cluster parameters (mean, covariance, weight).
  * **Flexibility:** Unlike K-Means, GMMs can model ellipsoidal clusters of different sizes, densities, and orientations.
  * **Generative Capabilities:** Demonstrated the ability to sample entirely new, synthetic instances (`sample()`) from the learned distributions.
  * **Density Estimation:** Utilized `score_samples()` to estimate the Probability Density Function (PDF) at any given location.
  * **Complexity Control:** Explored how restricting the `covariance_type` (e.g., "spherical", "diag", "tied") can reduce computational complexity and prevent the model from overfitting or struggling to converge on high-dimensional data.

### Project Files
* `dbscan_clustering.py`: Implementation of DBSCAN on non-spherical data and integration with KNN for predicting new instances.
* `gaussian_mixtures.py`: EM algorithm implementation, soft clustering predictions, generative sampling, and covariance type constraints.