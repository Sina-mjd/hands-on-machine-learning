# Chapter 4: Training Models 🚀

This folder contains notes, mathematical foundations, and code implementations of training linear and non-linear models from **Chapter 4 of Hands-On Machine Learning**.

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
* **Key Advantage:** Handles non-invertible matrices safely (e.g., when $m < n$ or when features are redundant).

---

## 3. Gradient Descent Algorithms

Optimization algorithms that iteratively tweak parameters to minimize the cost function:
$$\boldsymbol{\theta}^{(\text{next step})} = \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} \text{MSE}(\boldsymbol{\theta})$$

* **Batch Gradient Descent (BGD):** Computes the gradient over the full training set at every step using $\nabla_{\boldsymbol{\theta}} \text{MSE}(\boldsymbol{\theta}) = \frac{2}{m} \mathbf{X}^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$. High-precision, but slow on large datasets.
* **Stochastic Gradient Descent (SGD):** Picks a random instance at each step to compute the gradient. Extremely fast, supports Out-of-core learning, escapes local minima, but oscillates around the minimum.
* **Mini-batch Gradient Descent:** Computes gradients on small random subsets of instances. Leverages GPU matrix operations and provides smoother convergence than SGD.

---

## 4. Polynomial Regression

Used to fit non-linear data using linear models by adding powers of each feature as new features via `PolynomialFeatures`.

* **Feature Combinations:** Automatically includes interaction terms between features (e.g., $a^2, b^2, ab$).
* **Combinatorial Explosion:** Be cautious with high degrees ($d$) or many features ($n$), as total features explode to:
  $$\frac{(n + d)!}{d! \, n!}$$

---

## 5. Comparison of Linear Regression Algorithms

| Method / Algorithm | Large $m$ (Instances) | Out-of-core Support | Large $n$ (Features) | Hyperparameters | Scaling Required | Scikit-Learn Class |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Normal Equation** | Fast | ❌ No | Slow | 0 | ❌ No | N/A |
| **SVD** | Fast | ❌ No | Slow | 0 | ❌ No | `LinearRegression` |
| **Batch GD** | Slow | ❌ No | Fast | 2 ($\eta$, iterations) | ✅ Yes | `SGDRegressor` / Custom |
| **Stochastic GD** | Fast | ✅ Yes | Fast | $\ge 2$ ($\eta$, schedule) | ✅ Yes | `SGDRegressor` |
| **Mini-batch GD** | Fast | ✅ Yes | Fast | $\ge 2$ (batch size, $\eta$) | ✅ Yes | `SGDRegressor` / Custom |

### 📌 Key Takeaways:
* **Feature Scaling:** All Gradient Descent variants strictly require feature scaling (e.g., via `StandardScaler`) for fast convergence.
* **Prediction Uniformity:** Once trained, all methods produce identical models with equal prediction speed $O(m \times n)$.