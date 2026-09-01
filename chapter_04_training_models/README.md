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

## 4. Polynomial Regression & Learning Curves

Used to fit non-linear data using linear models by adding powers of each feature as new features via `PolynomialFeatures`.

* **Combinatorial Explosion:** Be cautious with high degrees ($d$) or many features ($n$), as total features explode to:
  $$\frac{(n + d)!}{d! \, n!}$$

### 📈 Learning Curves & The Bias/Variance Tradeoff
Plots of model performance on training vs. validation sets as a function of training set size:
* **Underfitting (High Bias):** Both training and validation errors are high and reach a plateau close together. Adding more training data does not help.
* **Overfitting (High Variance):** Large gap between training error (low) and validation error (high). Can be fixed by gathering more data or regularizing the model.

---

## 5. Regularized Linear Models

Regularization constrains model weights to reduce overfitting.

### 🔹 Ridge Regression ($L_2$ Penalty)
Adds a squared magnitude penalty to the cost function:
$$J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + \frac{\alpha}{2} \sum_{i=1}^{n} \theta_i^2$$
* Forces weights to stay as small as possible without driving them completely to zero.

### 🔹 Lasso Regression ($L_1$ Penalty)
Adds an absolute magnitude penalty to the cost function:
$$J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + \alpha \sum_{i=1}^{n} |\theta_i|$$
* Tends to eliminate weights of least important features ($\theta_i \to 0$), performing automatic feature selection.

### 🔹 Elastic Net
Combines both $L_1$ and $L_2$ penalties with a mix ratio $r$ (`l1_ratio`):
$$J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + r \alpha \sum_{i=1}^{n} |\theta_i| + \frac{1 - r}{2} \alpha \sum_{i=1}^{n} \theta_i^2$$
* Preferred over pure Lasso when $p > n$ (more features than instances) or when features are strongly correlated.

### 🔹 Early Stopping
A regularization technique for iterative algorithms (GD) where training is stopped as soon as validation error reaches a minimum.

---

## 6. Logistic Regression & Softmax Regression

### 🔹 Logistic Regression (Binary Classification)
Estimates the probability $p = h_{\boldsymbol{\theta}}(\mathbf{x})$ that an instance belongs to a class using the Sigmoid function $\sigma(t) = \frac{1}{1 + e^{-t}}$:
$$\hat{p} = \sigma(\boldsymbol{\theta}^T \mathbf{x})$$

* **Log Loss Cost Function:**
  $$J(\boldsymbol{\theta}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right]$$

### 🔹 Softmax Regression (Multiclass Classification)
Generalizes Logistic Regression to handle multiple classes without needing multiple binary classifiers.

1. **Score Calculation for Class $k$:** $s_k(\mathbf{x}) = \boldsymbol{\theta}_k^T \mathbf{x}$
2. **Softmax Function:**
   $$\hat{p}_k = \sigma(\mathbf{s}(\mathbf{x}))_k = \frac{\exp(s_k(\mathbf{x}))}{\sum_{j=1}^{K} \exp(s_j(\mathbf{x}))}$$
3. **Cross Entropy Cost Function:**
   $$J(\mathbf{\Theta}) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log\left(\hat{p}_k^{(i)}\right)$$

---

## 7. Comparison of Model Selection Strategy

| Model / Algorithm | Primary Use Case | Regularization Type | Feature Selection? | Handles Multicollinearity? | Scikit-Learn Class |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Linear Regression** | Baseline continuous prediction | None | ❌ No | ❌ Poor | `LinearRegression` |
| **Ridge Regression** | Default for continuous prediction | $L_2$ Penalty | ❌ No | ✅ Good | `Ridge` / `SGDRegressor` |
| **Lasso Regression** | Sparse feature sets | $L_1$ Penalty | ✅ Yes | ❌ Unstable | `Lasso` / `SGDRegressor` |
| **Elastic Net** | $p > n$ or correlated features | Combined $L_1 + L_2$ | ✅ Yes | ✅ Good | `ElasticNet` |
| **Logistic Regression**| Binary classification | $L_1$ / $L_2$ / ElasticNet | Optional | Dependent on Penalty | `LogisticRegression` |
| **Softmax Regression** | Multiclass classification | $L_1$ / $L_2$ / ElasticNet | Optional | Dependent on Penalty | `LogisticRegression(multi_class="multinomial")` |

---

### 📌 Key Takeaways:
* **Feature Scaling:** Strictly required for all Gradient Descent implementations, Lasso, Ridge, and Logistic Regression with regularization.
* **Regularization Choice:** Avoid plain Linear Regression. Default to Ridge, but use Elastic Net if you suspect correlated features or high dimensionality.
* **Multiclass Strategy:** Prefer Softmax Regression over One-versus-Rest (OvR) for mutually exclusive classes.