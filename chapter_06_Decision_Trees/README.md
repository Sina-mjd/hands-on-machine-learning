# Chapter 6: Decision Trees 🌲

This repository contains my notes and implementations for **Chapter 6** of *Hands-On Machine Learning*. Decision Trees are versatile, highly interpretable algorithms capable of both classification and regression, and they serve as the fundamental building blocks for Random Forests.

## 📌 Core Characteristics
*   **White Box Model:** Unlike black-box models (e.g., Neural Networks), Decision Trees provide intuitive, easy-to-explain decision rules.
*   **Zero Data Preparation:** Decision Trees do not require feature scaling or centering. They split data using orthogonal decision boundaries based on raw thresholds.
*   **Class Probabilities:** The model estimates the probability of a class by computing the ratio of training instances of that class in the corresponding leaf node.

## 🧠 The CART Training Algorithm
Scikit-Learn uses the **Classification And Regression Tree (CART)** algorithm, which exclusively builds *binary trees* (each node has exactly two children).

*   **Greedy Approach:** CART searches for the best feature $k$ and threshold $t_k$ to split the data at the *current* node to minimize impurity. It does not look ahead to see if this split leads to an optimal tree overall.
*   **NP-Complete:** Finding the absolute optimal Decision Tree requires $O(\exp(m))$ time. Thus, CART settles for a "reasonably good" (local optimum) solution.

## 📐 Measuring Impurity: Gini Index
To evaluate the quality of a split, the algorithm measures the **Gini Impurity** of the nodes. 
*   A node is considered **pure** (`gini = 0`) if all training instances within it belong to exactly one class.
*   The CART cost function tries to minimize the size-weighted sum of the Gini impurities of the left and right subsets.

## ⏱️ Computational Complexity
| Phase | Time Complexity | Note |
| :--- | :--- | :--- |
| **Prediction** | $O(\log_2(m))$ | Extremely fast; depends only on tree depth, independent of the number of features. |
| **Training** | $O(n \times m \log_2(m))$ | Slower; the algorithm compares all features across all instances at each node. |

*(Note: $m$ = number of instances, $n$ = number of features)*

## ⚠️ Limitations & Instability
1. **Orthogonal Boundaries:** Decision Trees split data strictly perpendicular to the axes. They struggle with rotated or diagonal datasets, often creating convoluted "staircase" boundaries. (Mitigation: Use PCA to rotate data first).
2. **High Variance:** Trees are extremely sensitive to minute variations in the training data. Removing or adding a single instance can radically alter the entire tree structure.
3. **Stochastic Nature:** Scikit-Learn evaluates a random subset of features at each split. Always set `random_state` for reproducible results.

*Ultimate Solution:* **Random Forests (Chapter 7)** solve this inherent instability by training many trees and averaging their predictions.