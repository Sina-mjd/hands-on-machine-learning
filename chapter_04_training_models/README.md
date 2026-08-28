# Chapter 4: Training Models 🚀

This folder contains notes, mathematical foundations, and code implementations of training linear models from **Chapter 4 of Hands-On Machine Learning**.

---

## 1. Linear Regression

A linear model makes predictions by computing a weighted sum of the input features, plus a constant term called the bias term (or intercept).

### 📐 Prediction Formula (Equations 4-1 & 4-2)
* **Expanded Form:**
  $$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n$$

* **Vectorized Form:**
  $$\hat{y} = h_\theta(\mathbf{x}) = \boldsymbol{\theta}^T \mathbf{x}$$

### 🎯 Cost Function (MSE)
To train the model, the parameter vector $\boldsymbol{\theta}$ is tuned to minimize the Mean Squared Error (MSE) over the training set:
$$\text{MSE}(\boldsymbol{\theta}) = \frac{1}{m} \sum_{i=1}^{m} \left( \boldsymbol{\theta}^T \mathbf{x}^{(i)} - y^{(i)} \right)^2$$

---

## 2. Closed-Form Solutions

### 🔹 Normal Equation
A direct mathematical approach to find the optimal values of $\boldsymbol{\theta}$:
$$\hat{\boldsymbol{\theta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

### 🔹 Pseudoinverse and SVD (Scikit-Learn Approach)
Scikit-Learn’s `LinearRegression` class uses Singular Value Decomposition (SVD) to compute the Moore-Penrose pseudoinverse ($\mathbf{X}^+$):
$$\hat{\boldsymbol{\theta}} = \mathbf{X}^+ \mathbf{y}$$
* **Key Advantage:** If the matrix $\mathbf{X}^T \mathbf{X}$ is not invertible (e.g., when $m < n$ or when features are redundant), the SVD method safely computes the optimal solution without raising errors.

---

## 3. Computational Complexity

| Method / Algorithm | Time Complexity wrt Features ($n$) | Time Complexity wrt Instances ($m$) | Memory Requirement (RAM) |
| :--- | :--- | :--- | :--- |
| **Normal Equation** | $O(n^{2.4})$ to $O(n^3)$ | $O(m)$ (Linear) | Requires full dataset in RAM |
| **SVD (Scikit-Learn)** | $O(n^2)$ | $O(m)$ (Linear) | Requires full dataset in RAM |

### 📌 Key Takeaways:
* **Number of Features ($n$):** Closed-form methods become extremely slow as the number of features grows large (e.g., $n > 100,000$).
* **Number of Instances ($m$):** Training time grows linearly with the number of instances, but the main limitation is RAM capacity.
* **Prediction Speed:** Once trained, making predictions is extremely fast and scales linearly $O(m \times n)$ with both instances and features.
* **Why Move to Gradient Descent?** Direct methods fail or become impractically slow when dealing with high-dimensional feature spaces ($n$ is very large) or datasets that exceed system RAM.