# California Housing Price Prediction 🏡📊

This repository contains my implementation and detailed notes for the California Housing Prices project from the renowned book **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** by Aurélien Géron (Chapter 2). 

The goal of this project is to build an end-to-end Machine Learning pipeline to predict median housing prices in California districts using various demographic, economic, and geographic features.

---

## 🚀 Project Workflow & Pipeline

### 1. Data Ingestion & Setup
* Automated data downloading and extraction pipelines.
* Implemented robust data loading mechanisms using Pandas.

### 2. Stratified Sampling 🎯
* Analyzed income distributions to avoid pure random sampling bias.
* Leveraged Scikit-Learn's `StratifiedShuffleSplit` based on income categories to ensure the test set accurately represents the overall population.

### 3. Exploratory Data Analysis & Feature Engineering 🔍
* Visualized geographical data and analyzed feature correlations using the Pearson correlation matrix.
* Created engineered ratios (e.g., `rooms_per_hhold`, `pop_per_hhold`, `bedrooms_per_room`) to extract richer signals for the model.

### 4. Automated Preprocessing Pipeline 🛠️
* **Data Cleaning:** Handled missing values dynamically using Scikit-Learn's `SimpleImputer` (median strategy).
* **Categorical Encoding:** Converted the categorical attribute `ocean_proximity` into binary vectors using `OneHotEncoder`.
* **Custom Transformers:** Developed a custom transformer class (`CombinedAttributesAdder`) inheriting from `BaseEstimator` and `TransformerMixin` to seamlessly integrate custom feature engineering into the pipeline.
* **Feature Scaling:** Applied `StandardScaler` to standardize all numerical features.
* **Unified Pipeline:** Combined numerical and categorical pipelines into a clean, reusable `ColumnTransformer` (`full_pipeline`).

### 5. Model Training & Fine-Tuning ⚙️
* Evaluated baseline models including **Linear Regression**, **Decision Tree**, and **Random Forest Regressor**.
* Utilized **K-Fold Cross-Validation** to accurately measure model performance without data leakage.
* Performed hyperparameter tuning using **`GridSearchCV`** to optimize the best-performing `RandomForestRegressor`.

### 6. Final Evaluation & Results 📉
* Evaluated the fine-tuned model on the unseen test set using `transform()` (avoiding data leakage).
* **Final RMSE:** ~ $47,730
* **95% Confidence Interval:** Computed using `scipy.stats` to estimate prediction error bound ($45,685 to $49,691).

---

## 💡 Key Insights & Feature Importance
* **Primary Predictor:** `median_income` proved to be the single most influential feature (~36% relative importance) in predicting district housing values.
* **Geographical Impact:** The `INLAND` category from `ocean_proximity` was the second most significant factor (~16% importance).

---

## 🛠️ Tech Stack
* **Language:** Python 🐍
* **Data Processing & ML:** NumPy, Pandas, Scikit-Learn, SciPy
* **Visualization:** Matplotlib
* **Environment & Version Control:** Jupyter Notebook, Git, SSH

---



