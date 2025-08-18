# 🚀 XGBoost

<div align="center">

![Method](https://img.shields.io/badge/Method-Gradient_Boosting-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-green?style=for-the-badge)
![Difficulty](https://img.shields.io/badge/Difficulty-Advanced-red?style=for-the-badge)

*A Comprehensive Guide to XGBoost (Extreme Gradient Boosting) for High-Performance Machine Learning*

</div>

---

## 📚 Table of Contents

- [Introduction to XGBoost](#introduction-to-xgboost)
- [Mathematical Foundation](#mathematical-foundation)
- [Implementation Guide](#implementation-guide)
- [Key Parameters](#key-parameters)
- [Feature Importance](#feature-importance)
- [Practical Applications](#practical-applications)
- [Advanced Techniques](#advanced-techniques)
- [Performance Optimization](#performance-optimization)
- [Comparison with Other Methods](#comparison-with-other-methods)
- [Common Pitfalls](#common-pitfalls)
- [FAQ](#faq)

---

## 🎯 Introduction to XGBoost

XGBoost (eXtreme Gradient Boosting) is a powerful, efficient, and versatile implementation of gradient boosted decision trees. It has become a leading choice for structured/tabular data in machine learning competitions and real-world applications due to its exceptional performance and speed.

### Key Concepts:

- **Gradient Boosting**: Builds an ensemble of weak prediction models (typically decision trees) sequentially, with each new model correcting errors of previous ones
- **Regularization**: Includes advanced regularization techniques to prevent overfitting
- **Parallelization**: Implements efficient parallel processing for faster computation
- **Tree Pruning**: Uses a novel tree pruning algorithm that reduces complexity
- **Out-of-Core Computing**: Handles datasets that don't fit in memory
- **Built-in Cross-Validation**: Provides mechanisms for monitoring model performance
- **Sparsity Awareness**: Efficiently handles missing values and sparse data

### Why XGBoost is Important:

1. **Performance**: Consistently outperforms many other algorithms on structured data
2. **Efficiency**: Optimized implementation for faster training and inference
3. **Flexibility**: Works well for regression, classification, and ranking problems
4. **Robustness**: Handles missing values and outliers effectively
5. **Interpretability**: Provides feature importance and visualization tools
6. **Community Support**: Widely used with extensive documentation and examples
7. **Production Ready**: Designed for real-world deployment scenarios

### Brief History:

- **2014**: Tianqi Chen began developing XGBoost as a research project at the University of Washington
- **2015**: XGBoost gained prominence by winning numerous Kaggle competitions
- **2016**: The first formal paper on XGBoost was published
- **2019**: Distributed version with DASK integration was introduced
- **Present**: XGBoost continues to evolve with new features and optimizations, becoming a standard tool in machine learning workflows

---

## 🧮 Mathematical Foundation

### Gradient Boosting Framework

XGBoost builds on the gradient boosting framework, which creates an ensemble of weak prediction models (typically decision trees) to minimize a loss function. The model is built in an additive manner:

$$\hat{y}_i = \sum_{k=1}^{K} f_k(x_i), \quad f_k \in \mathcal{F}$$

Where:
- $\hat{y}_i$ is the prediction for the $i$-th instance
- $f_k$ represents the $k$-th tree in the ensemble
- $\mathcal{F}$ is the space of regression trees
- $K$ is the number of trees

### Objective Function

XGBoost's objective function consists of a loss term and a regularization term:

$$\mathcal{L} = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)$$

Where:
- $l(y_i, \hat{y}_i)$ is the loss function measuring the difference between the prediction $\hat{y}_i$ and the target $y_i$
- $\Omega(f_k)$ is the regularization term penalizing the complexity of the tree $f_k$

### Regularization

The regularization term is defined as:

$$\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$

Where:
- $T$ is the number of leaves in the tree
- $w_j$ is the score (weight) assigned to the $j$-th leaf
- $\gamma$ is a parameter penalizing the number of leaves
- $\lambda$ is a parameter controlling L2 regularization on leaf weights

### Training Process

XGBoost uses a second-order approximation of the loss function for faster optimization:

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n [l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \Omega(f_t)$$

Where:
- $g_i = \partial_{\hat{y}_i^{(t-1)}} l(y_i, \hat{y}_i^{(t-1)})$ is the first derivative
- $h_i = \partial_{\hat{y}_i^{(t-1)}}^2 l(y_i, \hat{y}_i^{(t-1)})$ is the second derivative
- $\hat{y}_i^{(t-1)}$ is the prediction at iteration $t-1$

### Split Finding Algorithm

To find the best split in a tree, XGBoost uses the following gain formula:

$$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$

Where:
- $G_L$ and $G_R$ are the sums of first derivatives for the left and right nodes
- $H_L$ and $H_R$ are the sums of second derivatives for the left and right nodes
- $\lambda$ and $\gamma$ are regularization parameters

### Handling Missing Values

XGBoost has a built-in method for handling missing values by learning the best direction (left or right) for instances with missing values at each split.

### Key Algorithmic Innovations

1. **Sparsity-Aware Split Finding**: Efficiently handles sparse data and missing values
2. **Weighted Quantile Sketch**: Approximates optimal split points for weighted datasets
3. **Block Structure for Parallel Learning**: Enables parallelization and out-of-core computation
4. **Cache-Aware Access**: Optimizes memory usage for better hardware utilization

---

## 💻 Implementation Guide

### Basic Usage with Python

```python
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DMatrix (XGBoost's optimized data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters
params = {
    'objective': 'binary:logistic',  # Binary classification
    'eval_metric': 'logloss',        # Evaluation metric
    'max_depth': 3,                  # Maximum depth of a tree
    'learning_rate': 0.1,            # Learning rate
    'subsample': 0.8,                # Subsample ratio of the training instances
    'colsample_bytree': 0.8,         # Subsample ratio of columns when constructing each tree
    'reg_alpha': 0,                  # L1 regularization
    'reg_lambda': 1,                 # L2 regularization
    'random_state': 42               # Random seed
}

# Train model
num_rounds = 100
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_rounds,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=10,
    verbose_eval=10
)

# Make predictions
preds = model.predict(dtest)
best_preds = np.asarray([1 if i > 0.5 else 0 for i in preds])

# Evaluate
accuracy = accuracy_score(y_test, best_preds)
print(f"Accuracy: {accuracy:.4f}")

# Save model
model.save_model('xgboost_model.json')

# Load model
loaded_model = xgb.Booster()
loaded_model.load_model('xgboost_model.json')
```

### Using the Scikit-Learn API

```python
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load regression dataset
boston = load_boston()
X, y = boston.data, boston.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model using scikit-learn API
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=1,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric='rmse',
    early_stopping_rounds=10,
    verbose=True
)

# Make predictions
preds = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"RMSE: {rmse:.4f}")

# Feature importance
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]

print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. feature {boston.feature_names[indices[f]]} ({importance[indices[f]]})")
```

### Cross-Validation with XGBoost

```python
import xgboost as xgb
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

# Load multiclass dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create DMatrix
ddata = xgb.DMatrix(X, label=y)

# Set parameters
params = {
    'objective': 'multi:softmax',    # Multiclass classification
    'num_class': 3,                  # Number of classes
    'eval_metric': 'mlogloss',       # Evaluation metric
    'max_depth': 4,                  # Maximum depth of a tree
    'eta': 0.1,                      # Learning rate
    'subsample': 0.8,                # Subsample ratio of the training instances
    'colsample_bytree': 0.8,         # Subsample ratio of columns
    'random_state': 42               # Random seed
}

# Perform cross-validation
cv_results = xgb.cv(
    params,
    ddata,
    num_boost_round=100,
    nfold=5,
    stratified=True,              # Stratified sampling
    metrics=['merror', 'mlogloss'],
    early_stopping_rounds=10,
    seed=42,
    verbose_eval=10
)

# Print results
print(f"Best iteration: {len(cv_results)}")
print(f"Best error: {cv_results['test-merror-mean'].min()}")
print(f"Best logloss: {cv_results['test-mlogloss-mean'].min()}")

# Train final model with optimal number of rounds
best_rounds = len(cv_results)
final_model = xgb.train(params, ddata, num_boost_round=best_rounds)

# Save model
final_model.save_model('xgboost_iris_model.json')
```

### Grid Search for Hyperparameter Tuning

```python
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# Create and run GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Print best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", np.sqrt(-grid_search.best_score_))

# Evaluate on test set
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
print(f"Test RMSE: {rmse:.4f}")

# Analyze results
results = pd.DataFrame(grid_search.cv_results_)
results = results.sort_values(by='rank_test_score')

# Plot parameter effects
plt.figure(figsize=(15, 10))

# Plot max_depth vs learning_rate
pivot_table = results.pivot_table(
    index='param_max_depth', 
    columns='param_learning_rate', 
    values='mean_test_score'
)

plt.subplot(2, 2, 1)
sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.4f')
plt.title('max_depth vs learning_rate')

# Plot n_estimators effect
plt.subplot(2, 2, 2)
grouped = results.groupby('param_n_estimators')['mean_test_score'].mean()
plt.plot(grouped.index, grouped.values, 'o-')
plt.title('Effect of n_estimators')
plt.xlabel('n_estimators')
plt.ylabel('Mean Test Score')
plt.grid(True)

# Plot subsample effect
plt.subplot(2, 2, 3)
grouped = results.groupby('param_subsample')['mean_test_score'].mean()
plt.plot(grouped.index, grouped.values, 'o-')
plt.title('Effect of subsample')
plt.xlabel('subsample')
plt.ylabel('Mean Test Score')
plt.grid(True)

# Plot colsample_bytree effect
plt.subplot(2, 2, 4)
grouped = results.groupby('param_colsample_bytree')['mean_test_score'].mean()
plt.plot(grouped.index, grouped.values, 'o-')
plt.title('Effect of colsample_bytree')
plt.xlabel('colsample_bytree')
plt.ylabel('Mean Test Score')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## 🔑 Key Parameters

XGBoost has many parameters that control various aspects of the training process. Here's a guide to the most important ones:

### General Parameters

- **booster** [default=gbtree]
  - Which booster to use: 'gbtree', 'gblinear', or 'dart'
  - *Example*: `'booster': 'gbtree'`

- **verbosity** [default=1]
  - Controls the level of XGBoost's verbosity
  - *Example*: `'verbosity': 0` (silent), `'verbosity': 1` (warnings), `'verbosity': 2` (info), `'verbosity': 3` (debug)

- **nthread** [default=maximum available]
  - Number of parallel threads used
  - *Example*: `'nthread': 4`

### Booster Parameters

#### Tree Booster

- **eta** [default=0.3, alias: learning_rate]
  - Step size shrinkage used to prevent overfitting
  - Range: [0, 1]
  - *Example*: `'eta': 0.1`

- **gamma** [default=0, alias: min_split_loss]
  - Minimum loss reduction required for a split
  - Range: [0, ∞]
  - *Example*: `'gamma': 1`

- **max_depth** [default=6]
  - Maximum depth of a tree
  - Range: [1, ∞]
  - *Example*: `'max_depth': 3`

- **min_child_weight** [default=1]
  - Minimum sum of instance weight needed in a child
  - Range: [0, ∞]
  - *Example*: `'min_child_weight': 5`

- **subsample** [default=1]
  - Fraction of samples used for fitting trees
  - Range: (0, 1]
  - *Example*: `'subsample': 0.8`

- **colsample_bytree** [default=1]
  - Fraction of features used for fitting trees
  - Range: (0, 1]
  - *Example*: `'colsample_bytree': 0.8`

- **lambda** [default=1, alias: reg_lambda]
  - L2 regularization term on weights
  - Range: [0, ∞]
  - *Example*: `'lambda': 1.5`

- **alpha** [default=0, alias: reg_alpha]
  - L1 regularization term on weights
  - Range: [0, ∞]
  - *Example*: `'alpha': 0.5`

### Learning Task Parameters

- **objective** [default=reg:squarederror]
  - Defines the loss function to be optimized
  - For regression: 'reg:squarederror', 'reg:logistic', 'reg:pseudohubererror'
  - For classification: 'binary:logistic', 'multi:softmax', 'multi:softprob'
  - For ranking: 'rank:pairwise', 'rank:ndcg', 'rank:map'
  - *Example*: `'objective': 'binary:logistic'`

- **eval_metric** [default according to objective]
  - Evaluation metrics for validation data
  - For regression: 'rmse', 'mae', 'rmsle'
  - For classification: 'error', 'auc', 'logloss'
  - For ranking: 'map', 'ndcg'
  - *Example*: `'eval_metric': 'auc'`

- **seed** [default=0]
  - Random number seed
  - *Example*: `'seed': 42`

### Parameter Tuning Strategy

Here's a step-by-step approach to tuning XGBoost parameters:

1. **Start with reasonable defaults**:
   ```python
   params = {
       'objective': 'binary:logistic',  # Change based on your problem
       'max_depth': 6,
       'learning_rate': 0.1,
       'subsample': 0.8,
       'colsample_bytree': 0.8,
       'reg_alpha': 0,
       'reg_lambda': 1
   }
   ```

2. **Tune tree-based parameters**:
   - `max_depth`: Controls model complexity, try values between 3-10
   - `min_child_weight`: Balance between bias-variance, try 1-10
   - `gamma`: Controls pruning, try 0-1 in 0.1 increments

3. **Tune regularization parameters**:
   - `reg_alpha`: L1 regularization, try 0, 0.001, 0.01, 0.1, 1
   - `reg_lambda`: L2 regularization, try 0.1, 1, 5, 10, 50, 100

4. **Tune subsampling parameters**:
   - `subsample`: Try 0.6, 0.7, 0.8, 0.9, 1.0
   - `colsample_bytree`: Try 0.6, 0.7, 0.8, 0.9, 1.0

5. **Tune learning rate and number of estimators**:
   - Decrease `learning_rate` (try 0.01, 0.05, 0.1)
   - Increase `n_estimators` accordingly

### Parameter Visualization with Python

```python
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           n_redundant=5, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Function to evaluate a parameter
def evaluate_parameter(param_name, param_values, fixed_params=None):
    if fixed_params is None:
        fixed_params = {}
    
    results = []
    for value in param_values:
        # Set parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42
        }
        # Update with fixed parameters
        params.update(fixed_params)
        # Set the parameter being evaluated
        params[param_name] = value
        
        # Train the model
        evals_result = {}
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=10,
            verbose_eval=False,
            evals_result=evals_result
        )
        
        # Get the best score
        best_score = min(evals_result['test']['logloss'])
        results.append((value, best_score, model.best_iteration))
    
    return results

# Evaluate max_depth
max_depth_values = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
max_depth_results = evaluate_parameter('max_depth', max_depth_values)

# Evaluate learning_rate
learning_rate_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
learning_rate_results = evaluate_parameter('learning_rate', learning_rate_values)

# Evaluate subsample
subsample_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
subsample_results = evaluate_parameter('subsample', subsample_values)

# Evaluate colsample_bytree
colsample_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
colsample_results = evaluate_parameter('colsample_bytree', colsample_values)

# Plot the results
plt.figure(figsize=(20, 10))

# Plot max_depth
plt.subplot(2, 2, 1)
plt.plot([r[0] for r in max_depth_results], [r[1] for r in max_depth_results], 'o-')
plt.xlabel('max_depth')
plt.ylabel('Logloss')
plt.title('Effect of max_depth')
plt.grid(True)

# Plot learning_rate
plt.subplot(2, 2, 2)
plt.plot([r[0] for r in learning_rate_results], [r[1] for r in learning_rate_results], 'o-')
plt.xlabel('learning_rate')
plt.ylabel('Logloss')
plt.title('Effect of learning_rate')
plt.xscale('log')
plt.grid(True)

# Plot subsample
plt.subplot(2, 2, 3)
plt.plot([r[0] for r in subsample_results], [r[1] for r in subsample_results], 'o-')
plt.xlabel('subsample')
plt.ylabel('Logloss')
plt.title('Effect of subsample')
plt.grid(True)

# Plot colsample_bytree
plt.subplot(2, 2, 4)
plt.plot([r[0] for r in colsample_results], [r[1] for r in colsample_results], 'o-')
plt.xlabel('colsample_bytree')
plt.ylabel('Logloss')
plt.title('Effect of colsample_bytree')
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot the number of trees needed for each value
plt.figure(figsize=(20, 10))

# Plot max_depth vs n_trees
plt.subplot(2, 2, 1)
plt.plot([r[0] for r in max_depth_results], [r[2] for r in max_depth_results], 'o-')
plt.xlabel('max_depth')
plt.ylabel('Optimal Number of Trees')
plt.title('max_depth vs. Number of Trees')
plt.grid(True)

# Plot learning_rate vs n_trees
plt.subplot(2, 2, 2)
plt.plot([r[0] for r in learning_rate_results], [r[2] for r in learning_rate_results], 'o-')
plt.xlabel('learning_rate')
plt.ylabel('Optimal Number of Trees')
plt.title('learning_rate vs. Number of Trees')
plt.xscale('log')
plt.grid(True)

# Plot relationship between learning_rate and n_trees
learning_rate_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
n_estimators_values = [5000, 1000, 500, 200, 100, 50]

# Train models with different learning_rate and n_estimators combinations
lr_trees_results = []
for lr, n_trees in zip(learning_rate_values, n_estimators_values):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 3,
        'learning_rate': lr,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'random_state': 42
    }
    
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_trees,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=50,
        verbose_eval=False,
        evals_result=evals_result
    )
    
    best_score = min(evals_result['test']['logloss'])
    lr_trees_results.append((lr, n_trees, best_score, model.best_iteration))

# Plot learning_rate and n_trees combinations
plt.subplot(2, 2, 3)
plt.scatter([r[0] for r in lr_trees_results], [r[2] for r in lr_trees_results], 
            s=100, c=[r[3] for r in lr_trees_results], cmap='viridis')
plt.colorbar(label='Actual Trees Used')
plt.xlabel('Learning Rate')
plt.ylabel('Logloss')
plt.title('Learning Rate vs. Performance')
plt.xscale('log')
plt.grid(True)

for i, (lr, n_trees, score, actual_trees) in enumerate(lr_trees_results):
    plt.annotate(f"{actual_trees}/{n_trees}", (lr, score), 
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.show()
```

---

## 🔍 Feature Importance

XGBoost provides several methods to evaluate feature importance, helping you understand which features contribute most to the model's predictions.

### Types of Feature Importance in XGBoost

1. **Weight (default)**: Number of times a feature is used to split the data
2. **Gain**: Total gain of splits which use the feature
3. **Cover**: Number of samples affected by splits using the feature
4. **Total Gain**: Total gain across all splits
5. **Total Cover**: Total coverage across all splits

### Extracting Feature Importance

```python
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Get feature importance
importance_weight = model.feature_importances_

# Alternative methods using the booster
booster = model.get_booster()
importance_gain = booster.get_score(importance_type='gain')
importance_cover = booster.get_score(importance_type='cover')
importance_total_gain = booster.get_score(importance_type='total_gain')
importance_total_cover = booster.get_score(importance_type='total_cover')

# Convert to DataFrames
def get_importance_df(importance_dict, feature_names):
    if isinstance(importance_dict, dict):
        df = pd.DataFrame({
            'Feature': [fn.replace('f', '') for fn in importance_dict.keys()],
            'Importance': list(importance_dict.values())
        })
        df['Feature'] = df['Feature'].astype(int)
        df = df.sort_values('Feature')
        df['Feature'] = [feature_names[int(i)] for i in df['Feature']]
    else:
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_dict
        })
    
    # Sort by importance
    df = df.sort_values('Importance', ascending=False)
    return df

# Create DataFrames
weight_df = get_importance_df(importance_weight, feature_names)
gain_df = get_importance_df(importance_gain, feature_names)
cover_df = get_importance_df(importance_cover, feature_names)

# Plotting
plt.figure(figsize=(15, 12))

# Plot Weight importance
plt.subplot(3, 1, 1)
plt.barh(weight_df['Feature'][:10], weight_df['Importance'][:10])
plt.title('Feature Importance (Weight)')
plt.xlabel('Importance')
plt.gca().invert_yaxis()  # Highest importance at the top

# Plot Gain importance
plt.subplot(3, 1, 2)
plt.barh(gain_df['Feature'][:10], gain_df['Importance'][:10])
plt.title('Feature Importance (Gain)')
plt.xlabel('Importance')
plt.gca().invert_yaxis()

# Plot Cover importance
plt.subplot(3, 1, 3)
plt.barh(cover_df['Feature'][:10], cover_df['Importance'][:10])
plt.title('Feature Importance (Cover)')
plt.xlabel('Importance')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
```

### Visualizing Feature Importance with SHAP Values

```python
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load data
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot summary
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('Feature Importance (SHAP Values)')
plt.tight_layout()
plt.show()

# Plot detailed SHAP values
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Values')
plt.tight_layout()
plt.show()

# Plot SHAP dependence plots for top features
top_features = np.argsort(np.abs(shap_values).mean(0))[-3:]  # Top 3 features
for feature in top_features:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature, shap_values, X_test, show=False)
    plt.title(f'SHAP Dependence Plot for {X_test.columns[feature]}')
    plt.tight_layout()
    plt.show()

# Plot SHAP interaction values
shap_interaction_values = explainer.shap_interaction_values(X_test.iloc[:100])  # Limit to 100 samples for speed

# Plot interaction for a single prediction
plt.figure(figsize=(12, 12))
shap.plots.force(explainer.expected_value, shap_values[0], X_test.iloc[0], show=False)
plt.title('SHAP Force Plot for a Single Prediction')
plt.tight_layout()
plt.show()

# Plot waterfall for a single prediction
plt.figure(figsize=(10, 8))
shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, 
                                     data=X_test.iloc[0], feature_names=X_test.columns))
plt.title('SHAP Waterfall Plot for a Single Prediction')
plt.tight_layout()
plt.show()
```

### Feature Importance Stability Analysis

```python
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold

# Load data
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

# Set up K-Fold cross-validation
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# Store feature importance from each fold
feature_importance_weight = []
feature_importance_gain = []

# Train models and get feature importance for each fold
for train_idx, val_idx in k_fold.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Get feature importance
    importance_weight = model.feature_importances_
    feature_importance_weight.append(importance_weight)
    
    # Get gain importance
    booster = model.get_booster()
    importance_gain = booster.get_score(importance_type='gain')
    
    # Convert gain importance to array matching feature order
    gain_array = np.zeros(len(X.columns))
    for feature, importance in importance_gain.items():
        feature_idx = int(feature.replace('f', ''))
        gain_array[feature_idx] = importance
    
    feature_importance_gain.append(gain_array)

# Convert to numpy arrays
feature_importance_weight = np.array(feature_importance_weight)
feature_importance_gain = np.array(feature_importance_gain)

# Calculate mean and standard deviation
mean_importance_weight = np.mean(feature_importance_weight, axis=0)
std_importance_weight = np.std(feature_importance_weight, axis=0)

mean_importance_gain = np.mean(feature_importance_gain, axis=0)
std_importance_gain = np.std(feature_importance_gain, axis=0)

# Create DataFrames
stability_weight_df = pd.DataFrame({
    'Feature': X.columns,
    'Mean_Importance': mean_importance_weight,
    'Std_Importance': std_importance_weight,
    'CV': std_importance_weight / mean_importance_weight  # Coefficient of variation
})

stability_gain_df = pd.DataFrame({
    'Feature': X.columns,
    'Mean_Importance': mean_importance_gain,
    'Std_Importance': std_importance_gain,
    'CV': std_importance_gain / mean_importance_gain  # Coefficient of variation
})

# Sort by mean importance
stability_weight_df = stability_weight_df.sort_values('Mean_Importance', ascending=False)
stability_gain_df = stability_gain_df.sort_values('Mean_Importance', ascending=False)

# Plotting
plt.figure(figsize=(15, 10))

# Plot Weight importance stability
plt.subplot(2, 1, 1)
plt.errorbar(
    range(len(stability_weight_df)), 
    stability_weight_df['Mean_Importance'], 
    yerr=stability_weight_df['Std_Importance'],
    fmt='o',
    capsize=5
)
plt.xticks(range(len(stability_weight_df)), stability_weight_df['Feature'], rotation=45)
plt.title('Feature Importance Stability (Weight)')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.grid(True, linestyle='--', alpha=0.7)

# Plot Gain importance stability
plt.subplot(2, 1, 2)
plt.errorbar(
    range(len(stability_gain_df)), 
    stability_gain_df['Mean_Importance'], 
    yerr=stability_gain_df['Std_Importance'],
    fmt='o',
    capsize=5
)
plt.xticks(range(len(stability_gain_df)), stability_gain_df['Feature'], rotation=45)
plt.title('Feature Importance Stability (Gain)')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Plot coefficient of variation (lower is more stable)
plt.figure(figsize=(12, 6))
plt.bar(stability_weight_df['Feature'], stability_weight_df['CV'])
plt.title('Feature Importance Stability (Coefficient of Variation)')
plt.xlabel('Feature')
plt.ylabel('Coefficient of Variation (lower is more stable)')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7, axis='y')
plt.tight_layout()
plt.show()
```

---

## 🔬 Practical Applications

### Classification: Credit Risk Modeling

```python
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE

# Load dataset (use credit default dataset or any similar dataset)
# For this example, let's create synthetic data
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=10000, n_features=20, n_informative=10, n_redundant=5, 
    n_classes=2, weights=[0.9, 0.1], random_state=42  # Imbalanced
)

# Convert to DataFrame for better readability
feature_names = [f'feature_{i}' for i in range(1, 21)]
X = pd.DataFrame(X, columns=feature_names)

# Add some domain-specific features
X['credit_score'] = 300 + 500 * np.clip(X['feature_1'] + 0.1 * np.random.randn(len(X)), 0, 1)
X['income'] = 20000 + 80000 * np.clip(X['feature_2'] + 0.1 * np.random.randn(len(X)), 0, 1)
X['age'] = 18 + 62 * np.clip(X['feature_3'] + 0.1 * np.random.randn(len(X)), 0, 1)
X['employment_length'] = 0 + 40 * np.clip(X['feature_4'] + 0.1 * np.random.randn(len(X)), 0, 1)

# Exploratory Data Analysis
print("Class distribution:")
print(pd.Series(y).value_counts(normalize=True))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Original training class distribution:")
print(pd.Series(y_train).value_counts(normalize=True))

print("Resampled training class distribution:")
print(pd.Series(y_train_resampled).value_counts(normalize=True))

# Define evaluation metrics specific to credit risk
def evaluate_credit_model(model, X_val, y_val, threshold=0.5):
    """Evaluate a credit risk model with appropriate metrics."""
    # Predict probabilities
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    # Classification metrics
    accuracy = accuracy_score(y_val, y_pred)
    confusion = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    
    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve and AUC
    precision, recall, _ = precision_recall_curve(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)
    
    # Calculate additional business metrics
    tn, fp, fn, tp = confusion.ravel()
    
    # Default Rate (DR)
    default_rate = (fp + tp) / (tn + fp + fn + tp)
    
    # Approval Rate (AR)
    approval_rate = (tn + fp) / (tn + fp + fn + tp)
    
    # Bad Rate Among Accepts (BRAA)
    braa = fp / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': confusion,
        'classification_report': report,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'default_rate': default_rate,
        'approval_rate': approval_rate,
        'braa': braa
    }

# Train an XGBoost model with class weights
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=9,  # Compensate for class imbalance (90% / 10% = 9)
    random_state=42
)

# Train on resampled data
model.fit(
    X_train_resampled, y_train_resampled,
    eval_set=[(X_train_resampled, y_train_resampled), (X_test, y_test)],
    early_stopping_rounds=10,
    verbose=True
)

# Train another model on original data with scale_pos_weight
model_weighted = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=9,  # Compensate for class imbalance
    random_state=42
)

model_weighted.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    early_stopping_rounds=10,
    verbose=True
)

# Evaluate both models
results_resampled = evaluate_credit_model(model, X_test, y_test)
results_weighted = evaluate_credit_model(model_weighted, X_test, y_test)

print("\nModel trained on resampled data:")
print(f"AUC: {results_resampled['roc_auc']:.4f}")
print(f"PR-AUC: {results_resampled['pr_auc']:.4f}")
print(f"Approval Rate: {results_resampled['approval_rate']:.4f}")
print(f"Bad Rate Among Accepts: {results_resampled['braa']:.4f}")
print("\nConfusion Matrix:")
print(results_resampled['confusion_matrix'])
print("\nClassification Report:")
print(results_resampled['classification_report'])

print("\nModel trained with class weights:")
print(f"AUC: {results_weighted['roc_auc']:.4f}")
print(f"PR-AUC: {results_weighted['pr_auc']:.4f}")
print(f"Approval Rate: {results_weighted['approval_rate']:.4f}")
print(f"Bad Rate Among Accepts: {results_weighted['braa']:.4f}")
print("\nConfusion Matrix:")
print(results_weighted['confusion_matrix'])
print("\nClassification Report:")
print(results_weighted['classification_report'])

# Plot ROC curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(results_resampled['fpr'], results_resampled['tpr'], label=f'Resampled (AUC = {results_resampled["roc_auc"]:.4f})')
plt.plot(results_weighted['fpr'], results_weighted['tpr'], label=f'Weighted (AUC = {results_weighted["roc_auc"]:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
plt.plot(results_resampled['recall'], results_resampled['precision'], label=f'Resampled (PR-AUC = {results_resampled["pr_auc"]:.4f})')
plt.plot(results_weighted['recall'], results_weighted['precision'], label=f'Weighted (PR-AUC = {results_weighted["pr_auc"]:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Feature importance
plt.figure(figsize=(12, 8))
xgb.plot_importance(model, max_num_features=15, importance_type='gain')
plt.title('Feature Importance (Gain)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Threshold optimization
def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """Find the optimal threshold for a specific metric."""
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = []
    
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        elif metric == 'balanced_accuracy':
            score = (recall_score(y_true, y_pred) + 
                     precision_score(y_true, y_pred)) / 2
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    # Find the best threshold
    best_score_idx = np.argmax(scores)
    best_threshold = thresholds[best_score_idx]
    best_score = scores[best_score_idx]
    
    return best_threshold, best_score, thresholds, scores

# Get predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Find optimal thresholds for different metrics
f1_threshold, f1_score, f1_thresholds, f1_scores = find_optimal_threshold(y_test, y_prob, 'f1')
precision_threshold, precision_score, precision_thresholds, precision_scores = find_optimal_threshold(y_test, y_prob, 'precision')
recall_threshold, recall_score, recall_thresholds, recall_scores = find_optimal_threshold(y_test, y_prob, 'recall')
balanced_threshold, balanced_score, balanced_thresholds, balanced_scores = find_optimal_threshold(y_test, y_prob, 'balanced_accuracy')

# Plot threshold optimization
plt.figure(figsize=(14, 8))
plt.plot(f1_thresholds, f1_scores, label=f'F1 (Max: {f1_score:.4f} at {f1_threshold:.2f})')
plt.plot(precision_thresholds, precision_scores, label=f'Precision (Max: {precision_score:.4f} at {precision_threshold:.2f})')
plt.plot(recall_thresholds, recall_scores, label=f'Recall (Max: {recall_score:.4f} at {recall_threshold:.2f})')
plt.plot(balanced_thresholds, balanced_scores, label=f'Balanced (Max: {balanced_score:.4f} at {balanced_threshold:.2f})')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Threshold Optimization')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Create a profit curve (assumes costs and benefits)
def profit_curve(y_true, y_prob, cost_fp=1, cost_fn=5, benefit_tp=10, benefit_tn=0):
    """
    Create a profit curve based on costs and benefits.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities
    cost_fp : float
        Cost of a false positive (default loan that was predicted as non-default)
    cost_fn : float
        Cost of a false negative (non-default loan that was denied)
    benefit_tp : float
        Benefit of a true positive (correctly identified default)
    benefit_tn : float
        Benefit of a true negative (correctly identified non-default)
    
    Returns:
    --------
    thresholds : array
        Threshold values
    profits : array
        Profit at each threshold
    best_threshold : float
        Threshold that maximizes profit
    max_profit : float
        Maximum profit
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    profits = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        profit = (benefit_tp * tp + benefit_tn * tn - cost_fp * fp - cost_fn * fn)
        profits.append(profit)
    
    # Find the best threshold
    best_idx = np.argmax(profits)
    best_threshold = thresholds[best_idx]
    max_profit = profits[best_idx]
    
    return thresholds, profits, best_threshold, max_profit

# Create profit curve
thresholds, profits, best_threshold, max_profit = profit_curve(y_test, y_prob)

# Plot profit curve
plt.figure(figsize=(12, 6))
plt.plot(thresholds, profits)
plt.axvline(x=best_threshold, color='r', linestyle='--', 
           label=f'Best Threshold: {best_threshold:.2f}')
plt.axhline(y=max_profit, color='g', linestyle='--',
           label=f'Max Profit: {max_profit:.2f}')
plt.xlabel('Threshold')
plt.ylabel('Profit')
plt.title('Profit Curve')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Get feature importance
importance = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
importance_df = importance_df.sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
plt.xlabel('Importance')
plt.title('Top 15 Features')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Plot partial dependence plots for top features
from sklearn.inspection import partial_dependence, plot_partial_dependence

# Get top 5 features
top_features = importance_df['Feature'][:5].values

# Create partial dependence plots
fig, ax = plt.subplots(figsize=(12, 10))
plot_partial_dependence(model, X_train, features=top_features, ax=ax)
plt.tight_layout()
plt.show()
```

### Regression: House Price Prediction

```python
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import shap

# Load dataset (use housing dataset or any similar dataset)
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Exploratory Data Analysis
print("Dataset shape:", X.shape)
print("Feature names:", X.columns.tolist())
print("\nFeature statistics:")
print(X.describe())

# Check for missing values
print("\nMissing values:")
print(X.isnull().sum())

# Visualize target distribution
plt.figure(figsize=(10, 6))
plt.hist(y, bins=50)
plt.title('House Price Distribution')
plt.xlabel('Price (in $100,000s)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
corr = pd.concat([X, pd.Series(y, name='PRICE')], axis=1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (not always necessary for XGBoost, but can be helpful)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for better feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Define evaluation metrics for regression
def evaluate_regression_model(model, X_val, y_val):
    """Evaluate a regression model with appropriate metrics."""
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Calculate additional metrics
    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100  # Mean Absolute Percentage Error
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'predictions': y_pred
    }

# Train a baseline XGBoost model
baseline_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

baseline_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
    eval_metric='rmse',
    early_stopping_rounds=10,
    verbose=True
)

# Hyperparameter tuning with K-Fold Cross-Validation
def xgb_cv_tuning(X, y, param_grid, cv=5):
    """Perform cross-validation for XGBoost with parameter grid."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    best_score = float('inf')
    best_params = None
    all_results = []
    
    # For each parameter combination
    for params in param_grid:
        cv_scores = []
        
        # Perform cross-validation
        for train_idx, val_idx in kf.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_cv_scaled = scaler.fit_transform(X_train_cv)
            X_val_cv_scaled = scaler.transform(X_val_cv)
            
            # Convert back to DataFrame
            X_train_cv_scaled = pd.DataFrame(X_train_cv_scaled, columns=X_train_cv.columns)
            X_val_cv_scaled = pd.DataFrame(X_val_cv_scaled, columns=X_val_cv.columns)
            
            # Train model
            model = xgb.XGBRegressor(**params, random_state=42)
            model.fit(X_train_cv_scaled, y_train_cv, 
                     early_stopping_rounds=10, 
                     eval_set=[(X_val_cv_scaled, y_val_cv)],
                     verbose=False)
            
            # Evaluate
            y_pred = model.predict(X_val_cv_scaled)
                        rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred))
            cv_scores.append(rmse)
        
        # Calculate mean score for this parameter combination
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        # Store results
        all_results.append({
            'params': params,
            'mean_rmse': mean_score,
            'std_rmse': std_score
        })
        
        # Update best parameters if needed
        if mean_score < best_score:
            best_score = mean_score
            best_params = params
    
    return best_params, best_score, all_results

# Define parameter grid for tuning
param_grid = [
    {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0
    },
    {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0
    },
    {
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'gamma': 0.1
    },
    {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'gamma': 0.2
    }
]

# Convert y_train to Series if it's not already
if not isinstance(y_train, pd.Series):
    y_train = pd.Series(y_train)

# Perform cross-validation tuning
best_params, best_score, all_results = xgb_cv_tuning(X_train, y_train, param_grid)

print("Best parameters:", best_params)
print("Best RMSE:", best_score)

# Train model with best parameters
best_model = xgb.XGBRegressor(**best_params, random_state=42)
best_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
    eval_metric='rmse',
    early_stopping_rounds=10,
    verbose=True
)

# Evaluate baseline and best models
baseline_results = evaluate_regression_model(baseline_model, X_test_scaled, y_test)
best_results = evaluate_regression_model(best_model, X_test_scaled, y_test)

print("\nBaseline Model Results:")
print(f"RMSE: {baseline_results['rmse']:.4f}")
print(f"MAE: {baseline_results['mae']:.4f}")
print(f"R²: {baseline_results['r2']:.4f}")
print(f"MAPE: {baseline_results['mape']:.2f}%")

print("\nTuned Model Results:")
print(f"RMSE: {best_results['rmse']:.4f}")
print(f"MAE: {best_results['mae']:.4f}")
print(f"R²: {best_results['r2']:.4f}")
print(f"MAPE: {best_results['mape']:.2f}%")

# Plot actual vs. predicted
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, baseline_results['predictions'], alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Baseline Model: Actual vs. Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
plt.scatter(y_test, best_results['predictions'], alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Tuned Model: Actual vs. Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Plot residuals
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
residuals_baseline = y_test - baseline_results['predictions']
plt.scatter(baseline_results['predictions'], residuals_baseline, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Baseline Model: Residuals')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
residuals_best = y_test - best_results['predictions']
plt.scatter(best_results['predictions'], residuals_best, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Tuned Model: Residuals')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Feature importance analysis
plt.figure(figsize=(12, 8))
xgb.plot_importance(best_model, max_num_features=10, importance_type='gain')
plt.title('Feature Importance (Gain)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# SHAP analysis for more detailed feature importance
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_scaled)

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", show=False)
plt.title('SHAP Feature Importance')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_scaled, show=False)
plt.title('SHAP Feature Impact')
plt.tight_layout()
plt.show()

# Create a function to calculate and plot learning curves
def plot_learning_curves(model, X_train, y_train, X_test, y_test, metric='rmse'):
    """Plot learning curves for an XGBoost model."""
    results = {'train': [], 'test': []}
    
    for i in range(10, model.n_estimators + 1, 10):
        # Create a model with i trees
        temp_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=i,
            max_depth=model.max_depth,
            learning_rate=model.learning_rate,
            subsample=model.subsample,
            colsample_bytree=model.colsample_bytree,
            random_state=42
        )
        
        # Train the model
        temp_model.fit(X_train, y_train, verbose=False)
        
        # Make predictions
        y_train_pred = temp_model.predict(X_train)
        y_test_pred = temp_model.predict(X_test)
        
        # Calculate metric
        if metric == 'rmse':
            train_metric = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_metric = np.sqrt(mean_squared_error(y_test, y_test_pred))
        elif metric == 'mae':
            train_metric = mean_absolute_error(y_train, y_train_pred)
            test_metric = mean_absolute_error(y_test, y_test_pred)
        elif metric == 'r2':
            train_metric = r2_score(y_train, y_train_pred)
            test_metric = r2_score(y_test, y_test_pred)
        
        # Store results
        results['train'].append(train_metric)
        results['test'].append(test_metric)
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(10, model.n_estimators + 1, 10), results['train'], 'o-', label='Training')
    plt.plot(range(10, model.n_estimators + 1, 10), results['test'], 'o-', label='Testing')
    plt.xlabel('Number of Trees')
    plt.ylabel(metric.upper())
    plt.title(f'Learning Curves ({metric.upper()})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    return results

# Plot learning curves for the best model
learning_curve_results = plot_learning_curves(
    best_model, X_train_scaled, y_train, X_test_scaled, y_test
)

# Get optimal number of trees
optimal_n_trees = 10 * (np.argmin(learning_curve_results['test']) + 1)
print(f"Optimal number of trees: {optimal_n_trees}")

# Feature importance stability
n_iterations = 5
importance_iterations = []

for i in range(n_iterations):
    # Random subsample of the data
    indices = np.random.choice(len(X_train_scaled), size=int(0.8 * len(X_train_scaled)), replace=False)
    X_subsample = X_train_scaled.iloc[indices]
    y_subsample = y_train.iloc[indices]
    
    # Train model
    subsample_model = xgb.XGBRegressor(**best_params, random_state=i)
    subsample_model.fit(X_subsample, y_subsample, verbose=False)
    
    # Get feature importance
    importance = subsample_model.feature_importances_
    importance_iterations.append(importance)

# Convert to array
importance_iterations = np.array(importance_iterations)

# Calculate mean and standard deviation
mean_importance = np.mean(importance_iterations, axis=0)
std_importance = np.std(importance_iterations, axis=0)

# Create DataFrame
importance_stability = pd.DataFrame({
    'Feature': X_train.columns,
    'Mean_Importance': mean_importance,
    'Std_Importance': std_importance
})

# Sort by importance
importance_stability = importance_stability.sort_values('Mean_Importance', ascending=False)

# Plot importance stability
plt.figure(figsize=(12, 8))
plt.errorbar(
    range(len(importance_stability)),
    importance_stability['Mean_Importance'],
    yerr=importance_stability['Std_Importance'],
    fmt='o',
    capsize=5
)
plt.xticks(range(len(importance_stability)), importance_stability['Feature'], rotation=90)
plt.title('Feature Importance Stability')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Partial dependence plots for top features
from sklearn.inspection import plot_partial_dependence

# Get top 4 features
top_features = importance_stability['Feature'][:4].tolist()

# Create partial dependence plots
fig, ax = plt.subplots(figsize=(12, 10))
plot_partial_dependence(best_model, X_train_scaled, features=top_features, ax=ax)
plt.tight_layout()
plt.show()

# Export model
best_model.save_model('xgboost_housing_model.json')

# Sample inference code
loaded_model = xgb.XGBoost()
loaded_model.load_model('xgboost_housing_model.json')

# Make predictions on new data
new_data = X_test_scaled.iloc[:5]  # Just for demonstration
predictions = loaded_model.predict(new_data)
print("Sample predictions:", predictions)
```

### Time Series Forecasting

```python
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import shap

# Create synthetic time series data
np.random.seed(42)
dates = pd.date_range(start='2018-01-01', periods=1000, freq='D')
y = pd.Series(np.sin(np.linspace(0, 100, 1000)) + 0.1 * np.random.randn(1000) + 
              0.05 * np.linspace(0, 100, 1000), index=dates)

# Add some seasonality
y += 0.5 * np.sin(np.linspace(0, 40 * np.pi, 1000))  # Weekly seasonality
y += 1.0 * np.sin(np.linspace(0, 4 * np.pi, 1000))   # Yearly seasonality

# Create a dataframe
df = pd.DataFrame({'value': y})

# Extract datetime features
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['dayofweek'] = df.index.dayofweek
df['quarter'] = df.index.quarter
df['dayofyear'] = df.index.dayofyear
df['is_weekend'] = df.index.dayofweek >= 5

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['value'])
plt.title('Synthetic Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Create lag features
for lag in range(1, 8):  # 1 to 7 day lags
    df[f'lag_{lag}'] = df['value'].shift(lag)

# Create rolling statistics
for window in [7, 14, 30]:
    df[f'rolling_mean_{window}'] = df['value'].rolling(window=window).mean()
    df[f'rolling_std_{window}'] = df['value'].rolling(window=window).std()
    df[f'rolling_min_{window}'] = df['value'].rolling(window=window).min()
    df[f'rolling_max_{window}'] = df['value'].rolling(window=window).max()

# Remove NaN values
df = df.dropna()

# Create temporal train-test split
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# Prepare features and target
X_train = train_data.drop('value', axis=1)
y_train = train_data['value']
X_test = test_data.drop('value', axis=1)
y_test = test_data['value']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

# Train XGBoost model
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
    eval_metric='rmse',
    early_stopping_rounds=10,
    verbose=True
)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, y_test, label='Actual')
plt.plot(test_data.index, y_pred, label='Predicted')
plt.title('Time Series Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Plot residuals
plt.figure(figsize=(12, 6))
residuals = y_test - y_pred
plt.plot(test_data.index, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Forecast Residuals')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Analyze residual distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(residuals, bins=30)
plt.title('Residual Distribution')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
import scipy.stats as stats
stats.probplot(residuals, plot=plt)
plt.title('Residual Q-Q Plot')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Feature importance
plt.figure(figsize=(12, 8))
xgb.plot_importance(model, max_num_features=15, importance_type='gain')
plt.title('Feature Importance (Gain)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# SHAP analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", show=False)
plt.title('SHAP Feature Importance')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_scaled, show=False)
plt.title('SHAP Feature Impact')
plt.tight_layout()
plt.show()

# Multi-step forecasting
def forecast_future(model, last_known_values, future_steps=30, include_predictions=True):
    """
    Generate multi-step forecasts using the trained model.
    
    Parameters:
    -----------
    model : XGBoost model
        Trained XGBoost model
    last_known_values : DataFrame
        Last known values from the dataset
    future_steps : int
        Number of steps to forecast
    include_predictions : bool
        Whether to include predictions in the next step features
        
    Returns:
    --------
    pd.Series
        Forecasted values with datetime index
    """
    # Get the last date
    last_date = last_known_values.index[-1]
    
    # Create a dataframe for future dates
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)
    future_df = pd.DataFrame(index=future_dates)
    
    # Extract datetime features for future dates
    future_df['year'] = future_df.index.year
    future_df['month'] = future_df.index.month
    future_df['day'] = future_df.index.day
    future_df['dayofweek'] = future_df.index.dayofweek
    future_df['quarter'] = future_df.index.quarter
    future_df['dayofyear'] = future_df.index.dayofyear
    future_df['is_weekend'] = future_df.index.dayofweek >= 5
    
    # Initialize with known values
    known_values = last_known_values.copy()
    
    # Make predictions one step at a time
    predictions = []
    
    for i in range(future_steps):
        # Current date to predict
        current_date = future_dates[i]
        
        # Create lag features using known values
        for lag in range(1, 8):
            if i - lag + 1 >= 0:  # Use predictions
                if include_predictions:
                    future_df.loc[current_date, f'lag_{lag}'] = predictions[i - lag]
                else:
                    # If not using predictions, use the last known value
                    future_df.loc[current_date, f'lag_{lag}'] = known_values['value'].iloc[-lag]
            else:  # Use known values
                future_df.loc[current_date, f'lag_{lag}'] = known_values['value'].iloc[-(lag - i)]
        
        # Create rolling statistics (this is more complex and would need more careful implementation)
        # For simplicity, we'll use the last known rolling statistics
        for window in [7, 14, 30]:
            future_df.loc[current_date, f'rolling_mean_{window}'] = known_values[f'rolling_mean_{window}'].iloc[-1]
            future_df.loc[current_date, f'rolling_std_{window}'] = known_values[f'rolling_std_{window}'].iloc[-1]
            future_df.loc[current_date, f'rolling_min_{window}'] = known_values[f'rolling_min_{window}'].iloc[-1]
            future_df.loc[current_date, f'rolling_max_{window}'] = known_values[f'rolling_max_{window}'].iloc[-1]
        
        # Scale the features
        current_features = future_df.loc[current_date:current_date]
        current_features_scaled = scaler.transform(current_features)
        
        # Make prediction
        pred = model.predict(current_features_scaled)[0]
        predictions.append(pred)
        
        # If including predictions, update known values
        if include_predictions:
            new_row = known_values.iloc[-1:].copy()
            new_row.index = [current_date]
            new_row['value'] = pred
            known_values = pd.concat([known_values, new_row])
    
    # Return predictions as a series
    return pd.Series(predictions, index=future_dates)

# Get the last N rows of the training data
last_known = df.iloc[train_size-30:train_size]

# Generate forecasts
forecast_recursive = forecast_future(model, last_known, future_steps=len(test_data), include_predictions=True)
forecast_direct = forecast_future(model, last_known, future_steps=len(test_data), include_predictions=False)

# Plot forecasts
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, y_test, label='Actual')
plt.plot(test_data.index, y_pred, label='One-Step Forecast')
plt.plot(forecast_recursive.index, forecast_recursive, label='Recursive Forecast')
plt.plot(forecast_direct.index, forecast_direct, label='Direct Forecast')
plt.title('Multi-Step Time Series Forecasting')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Calculate RMSE for each method
rmse_one_step = np.sqrt(mean_squared_error(y_test, y_pred))
rmse_recursive = np.sqrt(mean_squared_error(y_test, forecast_recursive))
rmse_direct = np.sqrt(mean_squared_error(y_test, forecast_direct))

print(f"One-Step Forecast RMSE: {rmse_one_step:.4f}")
print(f"Recursive Forecast RMSE: {rmse_recursive:.4f}")
print(f"Direct Forecast RMSE: {rmse_direct:.4f}")

# Error analysis by forecast horizon
def error_by_horizon(actual, predicted, max_horizon=30):
    """Calculate error metrics by forecast horizon."""
    horizons = min(max_horizon, len(actual))
    rmse_by_horizon = []
    
    for h in range(1, horizons + 1):
        # Calculate RMSE for this horizon
        rmse = np.sqrt(mean_squared_error(actual[:h], predicted[:h]))
        rmse_by_horizon.append(rmse)
    
    return rmse_by_horizon

# Calculate error by horizon for each method
horizons = min(30, len(y_test))
rmse_by_horizon_one_step = error_by_horizon(y_test[:horizons], y_pred[:horizons])
rmse_by_horizon_recursive = error_by_horizon(y_test[:horizons], forecast_recursive[:horizons])
rmse_by_horizon_direct = error_by_horizon(y_test[:horizons], forecast_direct[:horizons])

# Plot error by horizon
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(rmse_by_horizon_one_step) + 1), rmse_by_horizon_one_step, 'o-', label='One-Step')
plt.plot(range(1, len(rmse_by_horizon_recursive) + 1), rmse_by_horizon_recursive, 's-', label='Recursive')
plt.plot(range(1, len(rmse_by_horizon_direct) + 1), rmse_by_horizon_direct, '^-', label='Direct')
plt.title('Forecast Error by Horizon')
plt.xlabel('Forecast Horizon (days)')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```

---

## 🔄 Advanced Techniques

### Handling Imbalanced Data

```python
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Create an imbalanced dataset
X, y = make_classification(
    n_samples=10000, n_features=20, n_informative=10, n_redundant=5,
    n_classes=2, weights=[0.95, 0.05], random_state=42  # 95% class 0, 5% class 1
)

# Convert to DataFrame for better handling
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(1, 21)])
y = pd.Series(y, name='target')

# Display class distribution
print("Class distribution:")
print(y.value_counts(normalize=True))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define a function to train and evaluate XGBoost with different approaches
def train_evaluate_xgboost(X_train, y_train, X_test, y_test, method='standard', **kwargs):
    """
    Train and evaluate XGBoost with different approaches for imbalanced data.
    
    Parameters:
    -----------
    X_train, y_train : training data
    X_test, y_test : testing data
    method : str
        One of 'standard', 'scale_pos_weight', 'smote', 'undersampling', 'combined'
    **kwargs : additional parameters for the specific method
    
    Returns:
    --------
    dict with model and evaluation metrics
    """
    if method == 'standard':
        # Standard XGBoost
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        X_train_resampled, y_train_resampled = X_train, y_train
        
    elif method == 'scale_pos_weight':
        # XGBoost with scale_pos_weight
        scale_pos_weight = kwargs.get('scale_pos_weight', 
                                     y_train.value_counts()[0] / y_train.value_counts()[1])
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        X_train_resampled, y_train_resampled = X_train, y_train
        
    elif method == 'smote':
        # SMOTE oversampling
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
    elif method == 'undersampling':
        # Random undersampling
        undersampler = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
    elif method == 'combined':
        # Combined SMOTE and Tomek links
        combined = SMOTETomek(random_state=42)
        X_train_resampled, y_train_resampled = combined.fit_resample(X_train, y_train)
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Train the model
    model.fit(
        X_train_resampled, y_train_resampled,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        early_stopping_rounds=10,
        verbose=False
    )
    
    # Make predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    # Find optimal threshold using F1 score
    from sklearn.metrics import f1_score
    thresholds = np.linspace(0.01, 0.99, 99)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)
    
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    
    # Confusion matrix at optimal threshold
    y_pred = (y_prob >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    # True positive rate and true negative rate
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)  # Sensitivity, recall
    tnr = tn / (tn + fp)  # Specificity
    
    # Print class distribution after resampling
    print(f"{method} - Training class distribution after resampling:")
    print(pd.Series(y_train_resampled).value_counts(normalize=True))
    
    return {
        'method': method,
        'model': model,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'tpr': tpr,
        'tnr': tnr,
        'best_threshold': best_threshold,
        'probabilities': y_prob
    }

# Train and evaluate with different approaches
results = {}

# 1. Standard XGBoost
results['standard'] = train_evaluate_xgboost(X_train, y_train, X_test, y_test, method='standard')

# 2. XGBoost with scale_pos_weight
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
results['scale_pos_weight'] = train_evaluate_xgboost(
    X_train, y_train, X_test, y_test, 
    method='scale_pos_weight',
    scale_pos_weight=scale_pos_weight
)

# 3. SMOTE oversampling
results['smote'] = train_evaluate_xgboost(X_train, y_train, X_test, y_test, method='smote')

# 4. Random undersampling
results['undersampling'] = train_evaluate_xgboost(X_train, y_train, X_test, y_test, method='undersampling')

# 5. Combined approach
results['combined'] = train_evaluate_xgboost(X_train, y_train, X_test, y_test, method='combined')

# Summarize results
print("\nResults Summary:")
print(f"{'Method':<20} {'ROC AUC':<10} {'PR AUC':<10} {'TPR':<10} {'TNR':<10}")
print("-" * 60)
for method, result in results.items():
    print(f"{method:<20} {result['roc_auc']:.4f}    {result['pr_auc']:.4f}    {result['tpr']:.4f}    {result['tnr']:.4f}")

# Plot ROC curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for method, result in results.items():
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
    plt.plot(fpr, tpr, label=f"{method} (AUC = {result['roc_auc']:.4f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Plot Precision-Recall curves
plt.subplot(1, 2, 2)
for method, result in results.items():
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_test, result['probabilities'])
    plt.plot(recall, precision, label=f"{method} (AUC = {result['pr_auc']:.4f})")

# Add baseline
plt.plot([0, 1], [y_test.mean(), y_test.mean()], 'k--', label='Baseline')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Plot confusion matrices
plt.figure(figsize=(15, 8))
for i, (method, result) in enumerate(results.items()):
    plt.subplot(2, 3, i+1)
    cm = result['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{method} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# Feature importance comparison
plt.figure(figsize=(15, 10))
for i, (method, result) in enumerate(results.items()):
    if i < 6:  # Limit to 6 subplots
        plt.subplot(2, 3, i+1)
        xgb.plot_importance(result['model'], max_num_features=10, importance_type='gain', ax=plt.gca())
        plt.title(f'{method} Feature Importance')

plt.tight_layout()
plt.show()

# Cost-sensitive learning with custom objective function
def custom_obj(y_pred, dtrain):
    """Custom objective function with asymmetric costs."""
    y_true = dtrain.get_label()
    
    # Sigmoid transformation of raw predictions
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    
    # Cost of false positive (predicting 1 when actual is 0)
    cost_fp = 1.0
    # Cost of false negative (predicting 0 when actual is 1)
    cost_fn = 10.0  # Higher cost for missing positive cases
    
    # Calculate gradients
    grad = cost_fp * y_pred * (1 - y_true) - cost_fn * (1 - y_pred) * y_true
    
    # Calculate hessian
    hess = (cost_fp * (1 - y_true) + cost_fn * y_true) * y_pred * (1 - y_pred)
    
    return grad, hess

# Train XGBoost with custom objective
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'max_depth': 4,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

model_custom = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    obj=custom_obj,
    evals=[(dtest, 'test')],
    early_stopping_rounds=10,
    verbose_eval=False
)

# Make predictions
y_prob_custom = model_custom.predict(dtest)

# Calculate metrics
roc_auc_custom = roc_auc_score(y_test, y_prob_custom)
precision, recall, _ = precision_recall_curve(y_test, y_prob_custom)
pr_auc_custom = auc(recall, precision)

# Find optimal threshold
thresholds = np.linspace(0.01, 0.99, 99)
f1_scores = []

for threshold in thresholds:
    y_pred = (y_prob_custom >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

best_threshold_idx = np.argmax(f1_scores)
best_threshold_custom = thresholds[best_threshold_idx]

# Confusion matrix at optimal threshold
y_pred_custom = (y_prob_custom >= best_threshold_custom).astype(int)
cm_custom = confusion_matrix(y_test, y_pred_custom)

# True positive rate and true negative rate
tn, fp, fn, tp = cm_custom.ravel()
tpr_custom = tp / (tp + fn)
tnr_custom = tn / (tn + fp)

# Add custom approach to results
results['custom_objective'] = {
    'method': 'custom_objective',
    'roc_auc': roc_auc_custom,
    'pr_auc': pr_auc_custom,
    'confusion_matrix': cm_custom,
    'tpr': tpr_custom,
    'tnr': tnr_custom,
    'best_threshold': best_threshold_custom,
    'probabilities': y_prob_custom
}

# Update results summary
print("\nUpdated Results Summary (including custom objective):")
print(f"{'Method':<20} {'ROC AUC':<10} {'PR AUC':<10} {'TPR':<10} {'TNR':<10}")
print("-" * 60)
for method, result in results.items():
    print(f"{method:<20} {result['roc_auc']:.4f}    {result['pr_auc']:.4f}    {result['tpr']:.4f}    {result['tnr']:.4f}")

# Update ROC and PR curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for method, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
    plt.plot(fpr, tpr, label=f"{method} (AUC = {result['roc_auc']:.4f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
for method, result in results.items():
    precision, recall, _ = precision_recall_curve(y_test, result['probabilities'])
    plt.plot(recall, precision, label=f"{method} (AUC = {result['pr_auc']:.4f})")

plt.plot([0, 1], [y_test.mean(), y_test.mean()], 'k--', label='Baseline')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
```

### XGBoost for Multi-Label Classification

```python
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier

# Create multi-label dataset
X, y = make_multilabel_classification(
    n_samples=1000, n_features=20, n_classes=5, n_labels=2,
    random_state=42
)

# Convert to DataFrame for better handling
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(1, 21)])
y = pd.DataFrame(y, columns=[f'label_{i}' for i in range(1, 6)])

# Display dataset info
print("Multi-label dataset shape:", X.shape, y.shape)
print("Label distribution per class:")
for col in y.columns:
    print(f"{col}: {y[col].mean():.2f} (positive rate)")

print("\nLabel combinations:")
label_combinations = y.sum(axis=1).value_counts().sort_index()
print(label_combinations)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Method 1: Binary Relevance (One XGBoost model per label)
def train_binary_relevance(X_train, y_train, X_test, y_test):
    """Train one XGBoost model per label (Binary Relevance approach)."""
    models = {}
    predictions = {}
    
    for col in y_train.columns:
        # Train model for this label
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        model.fit(
            X_train, y_train[col],
            eval_set=[(X_test, y_test[col])],
            eval_metric='logloss',
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Store model
        models[col] = model
        
        # Make predictions
        predictions[col] = model.predict(X_test)
    
    # Combine predictions
    y_pred = pd.DataFrame(predictions)
    
    return models, y_pred

# Method 2: Using OneVsRestClassifier
def train_one_vs_rest(X_train, y_train, X_test, y_test):
    """Train XGBoost with OneVsRestClassifier."""
    # Create base model
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Create multi-label model
    model = OneVsRestClassifier(base_model)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, pd.DataFrame(y_pred, columns=y_test.columns)

# Method 3: Using XGBoost's built-in multi:softprob
def train_multi_softprob(X_train, y_train, X_test, y_test):
    """
    Train XGBoost with multi:softprob.
    Note: This approach works best when labels are mutually exclusive,
    but we're adapting it for multi-label by setting custom thresholds.
    """
    # Convert labels to array
    y_train_array = y_train.values
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=None)
    dtest = xgb.DMatrix(X_test, label=None)
    
    # Add multi-label target as additional features
    for i, col in enumerate(y_train.columns):
        dtrain.set_float_info(f'label_{i}', y_train[col].values)
    
    # Set parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 4,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    
    # Train separate models for each label
    models = {}
    predictions = {}
    
    for i, col in enumerate(y_train.columns):
        # Train model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # Store model
        models[col] = model
        
        # Make predictions
        predictions[col] = (model.predict(dtest) > 0.5).astype(int)
    
    # Combine predictions
    y_pred = pd.DataFrame(predictions)
    
    return models, y_pred

# Method 4: Custom XGBoost Multi-Label Loss
def custom_multi_label_obj(y_pred, dtrain):
    """Custom objective function for multi-label classification."""
    # Get multi-label targets from DMatrix
    y_true = np.column_stack([
        dtrain.get_float_info(f'label_{i}')
        for i in range(5)  # Number of labels
    ])
    
    # Reshape predictions to match number of labels
    y_pred = y_pred.reshape(-1, 5)
    
    # Apply sigmoid
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    
    # Calculate binary cross-entropy for each label
    grad = y_pred - y_true
    hess = y_pred * (1 - y_pred)
    
    return grad.flatten(), hess.flatten()

def train_custom_multi_label(X_train, y_train, X_test, y_test):
    """Train XGBoost with custom multi-label objective."""
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=None)
    dtest = xgb.DMatrix(X_test, label=None)
    
    # Add multi-label target as additional features
    for i, col in enumerate(y_train.columns):
        dtrain.set_float_info(f'label_{i}', y_train[col].values)
        dtest.set_float_info(f'label_{i}', y_test[col].values)
    
    # Set parameters
    params = {
        'max_depth': 4,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    
    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        obj=custom_multi_label_obj,
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    # Make predictions
    y_pred_proba = model.predict(dtest).reshape(-1, 5)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    return model, pd.DataFrame(y_pred, columns=y_test.columns)

# Train models using different approaches
print("\nTraining Binary Relevance approach...")
br_models, br_preds = train_binary_relevance(X_train_scaled, y_train, X_test_scaled, y_test)

print("Training OneVsRest approach...")
ovr_model, ovr_preds = train_one_vs_rest(X_train_scaled, y_train, X_test_scaled, y_test)

print("Training multi:softprob approach...")
ms_models, ms_preds = train_multi_softprob(X_train_scaled, y_train, X_test_scaled, y_test)

print("Training custom multi-label approach...")
custom_model, custom_preds = train_custom_multi_label(X_train_scaled, y_train, X_test_scaled, y_test)

# Evaluate models
def evaluate_multi_label(y_true, y_pred, method_name):
    """Evaluate multi-label predictions with various metrics."""
    # Hamming loss (lower is better)
    h_loss = hamming_loss(y_true, y_pred)
    
    # Subset accuracy (exact match, higher is better)
    subset_acc = accuracy_score(y_true, y_pred)
    
    # Sample-wise F1 score (higher is better)
    sample_f1 = f1_score(y_true, y_pred, average='samples')
    
    # Micro-averaged F1 score (higher is better)
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    
    # Macro-averaged F1 score (higher is better)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # Per-label F1 scores
    label_f1 = {}
    for i, col in enumerate(y_true.columns):
        label_f1[col] = f1_score(y_true[col], y_pred[col])
    
    print(f"\n{method_name} Results:")
    print(f"Hamming Loss: {h_loss:.4f} (lower is better)")
    print(f"Subset Accuracy: {subset_acc:.4f}")
    print(f"Sample-wise F1: {sample_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Per-label F1 scores:")
    for label, score in label_f1.items():
        print(f"  {label}: {score:.4f}")
    
    return {
        'method': method_name,
        'hamming_loss': h_loss,
        'subset_accuracy': subset_acc,
        'sample_f1': sample_f1,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'label_f1': label_f1
    }

# Evaluate all methods
results = []
results.append(evaluate_multi_label(y_test, br_preds, "Binary Relevance"))
results.append(evaluate_multi_label(y_test, ovr_preds, "OneVsRest"))
results.append(evaluate_multi_label(y_test, ms_preds, "Multi:Softprob"))
results.append(evaluate_multi_label(y_test, custom_preds, "Custom Multi-Label"))

# Plot comparison of methods
plt.figure(figsize=(15, 10))

# Plot Hamming Loss (lower is better)
plt.subplot(2, 2, 1)
plt.bar([r['method'] for r in results], [r['hamming_loss'] for r in results])
plt.title('Hamming Loss (lower is better)')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

# Plot Subset Accuracy
plt.subplot(2, 2, 2)
plt.bar([r['method'] for r in results], [r['subset_accuracy'] for r in results])
plt.title('Subset Accuracy')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

# Plot F1 Scores
plt.subplot(2, 2, 3)
x = np.arange(len(results))
width = 0.25
plt.bar(x - width, [r['sample_f1'] for r in results], width, label='Sample F1')
plt.bar(x, [r['micro_f1'] for r in results], width, label='Micro F1')
plt.bar(x + width, [r['macro_f1'] for r in results], width, label='Macro F1')
plt.xticks(x, [r['method'] for r in results], rotation=45)
plt.title('F1 Scores')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

# Plot Per-label F1 Scores
plt.subplot(2, 2, 4)
x = np.arange(len(y_test.columns))
width = 0.2
for i, r in enumerate(results):
    plt.bar(x + (i - 1.5) * width, [r['label_f1'][label] for label in y_test.columns], 
            width, label=r['method'])
plt.xticks(x, y_test.columns)
plt.title('Per-label F1 Scores')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Make room for the legend
plt.show()

# Feature importance analysis for Binary Relevance models
plt.figure(figsize=(15, 12))
for i, (label, model) in enumerate(br_models.items()):
    if i < 6:  # Limit to 6 subplots
        plt.subplot(2, 3, i+1)
        xgb.plot_importance(model, max_num_features=10, importance_type='gain', ax=plt.gca())
        plt.title(f'Feature Importance for {label}')

plt.tight_layout()
plt.show()

# Analyze label correlations
plt.figure(figsize=(10, 8))
corr = y.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Label Correlations')
plt.tight_layout()
plt.show()

# Analyze prediction errors
def analyze_prediction_errors(y_true, y_pred, method_name):
    """Analyze prediction errors for multi-label classification."""
    # Calculate errors per sample
    errors_per_sample = np.sum(y_true.values != y_pred.values, axis=1)
    
    # Calculate errors per label
    errors_per_label = {}
    for col in y_true.columns:
        errors_per_label[col] = np.sum(y_true[col].values != y_pred[col].values)
    
    # Error types per label
    fn_per_label = {}  # False negatives
    fp_per_label = {}  # False positives
    
    for col in y_true.columns:
        true_vals = y_true[col].values
        pred_vals = y_pred[col].values
        
        fn_per_label[col] = np.sum((true_vals == 1) & (pred_vals == 0))
        fp_per_label[col] = np.sum((true_vals == 0) & (pred_vals == 1))
    
    return {
        'method': method_name,
        'errors_per_sample': errors_per_sample,
        'errors_per_label': errors_per_label,
        'fn_per_label': fn_per_label,
        'fp_per_label': fp_per_label
    }

# Analyze errors for all methods
error_analysis = []
error_analysis.append(analyze_prediction_errors(y_test, br_preds, "Binary Relevance"))
error_analysis.append(analyze_prediction_errors(y_test, ovr_preds, "OneVsRest"))
error_analysis.append(analyze_prediction_errors(y_test, ms_preds, "Multi:Softprob"))
error_analysis.append(analyze_prediction_errors(y_test, custom_preds, "Custom Multi-Label"))

# Plot error analysis
plt.figure(figsize=(15, 10))

# Plot errors per sample distribution
plt.subplot(2, 2, 1)
for analysis in error_analysis:
    plt.hist(analysis['errors_per_sample'], alpha=0.5, label=analysis['method'], bins=range(7))
plt.title('Distribution of Errors per Sample')
plt.xlabel('Number of Errors')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Plot errors per label
plt.subplot(2, 2, 2)
x = np.arange(len(y_test.columns))
width = 0.2
for i, analysis in enumerate(error_analysis):
    plt.bar(x + (i - 1.5) * width, [analysis['errors_per_label'][label] for label in y_test.columns], 
            width, label=analysis['method'])
plt.xticks(x, y_test.columns)
plt.title('Total Errors per Label')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

# Plot false negatives per label
plt.subplot(2, 2, 3)
for i, analysis in enumerate(error_analysis):
    plt.bar(x + (i - 1.5) * width, [analysis['fn_per_label'][label] for label in y_test.columns], 
            width, label=analysis['method'])
plt.xticks(x, y_test.columns)
plt.title('False Negatives per Label')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

# Plot false positives per label
plt.subplot(2, 2, 4)
for i, analysis in enumerate(error_analysis):
    plt.bar(x + (i - 1.5) * width, [analysis['fp_per_label'][label] for label in y_test.columns], 
            width, label=analysis['method'])
plt.xticks(x, y_test.columns)
plt.title('False Positives per Label')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

plt.tight_layout()
plt.show()
```

### XGBoost for Ranking

```python
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random

# Generate synthetic ranking data
def generate_ranking_data(n_samples=1000, n_features=20, n_queries=100):
    """Generate synthetic data for learning to rank."""
    # Generate features and target
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1, random_state=42)
    
    # Normalize target to [0, 1] range and convert to relevance scores (0-4)
    y = (y - y.min()) / (y.max() - y.min())
    y = np.floor(y * 5).astype(int)  # 0-4 relevance scores
    
    # Assign query groups
    query_ids = np.random.randint(0, n_queries, size=n_samples)
        # Sort within each query group to simulate different ranking positions
    query_groups = {}
    for i, qid in enumerate(query_ids):
        if qid not in query_groups:
            query_groups[qid] = []
        query_groups[qid].append((X[i], y[i]))
    
    # Collect sorted data
    X_sorted = []
    y_sorted = []
    group_sizes = []
    
    for qid in sorted(query_groups.keys()):
        items = query_groups[qid]
        group_sizes.append(len(items))
        
        # Add some randomness to relevance to make it more realistic
        for item in items:
            X_sorted.append(item[0])
            y_sorted.append(item[1])
    
    return np.array(X_sorted), np.array(y_sorted), np.array(group_sizes)

# Generate data
X, y, groups = generate_ranking_data(n_samples=5000, n_features=20, n_queries=200)

# Convert to DataFrame
X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(1, 21)])

# Display dataset info
print(f"Dataset shape: {X_df.shape}")
print(f"Number of queries: {len(groups)}")
print(f"Relevance score distribution:")
for relevance in range(5):
    print(f"  Relevance {relevance}: {np.sum(y == relevance)} items ({np.mean(y == relevance):.1%})")

# Split data for training and testing
# We need to split by query groups to maintain the ranking structure
query_indices = np.cumsum(groups)
query_boundaries = list(zip([0] + query_indices[:-1].tolist(), query_indices.tolist()))
random.seed(42)
random.shuffle(query_boundaries)

# Use 80% for training, 20% for testing
train_size = int(len(query_boundaries) * 0.8)
train_bounds = query_boundaries[:train_size]
test_bounds = query_boundaries[train_size:]

# Collect indices for train and test sets
train_indices = []
for start, end in train_bounds:
    train_indices.extend(range(start, end))

test_indices = []
for start, end in test_bounds:
    test_indices.extend(range(start, end))

# Split the data
X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]

# Calculate group sizes after split
train_groups = []
current_group = 0
for start, end in train_bounds:
    train_groups.append(end - start)

test_groups = []
for start, end in test_bounds:
    test_groups.append(end - start)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# Set group info for ranking
dtrain.set_group(train_groups)
dtest.set_group(test_groups)

# Train XGBoost ranking model
params = {
    'objective': 'rank:pairwise',  # For ranking
    'eval_metric': ['ndcg@5', 'ndcg@10'],  # Common ranking metrics
    'eta': 0.1,
    'gamma': 1.0,
    'min_child_weight': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# Train model
model = xgb.train(
    params,
    dtrain,
    num_boost_round=200,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=20,
    verbose_eval=10
)

# Make predictions
preds_train = model.predict(dtrain)
preds_test = model.predict(dtest)

# Define function to evaluate ranking performance
def evaluate_ranking(y_true, y_pred, groups, k_values=[1, 3, 5, 10]):
    """
    Evaluate ranking performance with NDCG, MAP, and Precision@k.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth relevance scores
    y_pred : array-like
        Predicted scores
    groups : array-like
        Group sizes for queries
    k_values : list
        Values of k to evaluate metrics at
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import ndcg_score
    
    results = {}
    
    # Calculate metrics for each query group
    ndcg_scores = {k: [] for k in k_values}
    precision_scores = {k: [] for k in k_values}
    
    start = 0
    for group_size in groups:
        end = start + group_size
        
        # Get relevance and predictions for this group
        group_y_true = y_true[start:end]
        group_y_pred = y_pred[start:end]
        
        # Sort by predictions (descending)
        sorted_indices = np.argsort(group_y_pred)[::-1]
        sorted_y_true = group_y_true[sorted_indices]
        
        # Calculate NDCG for each k
        for k in k_values:
            if k <= group_size:
                # Use sklearn's ndcg_score which expects 2D arrays
                y_true_reshaped = np.asarray([group_y_true])
                y_pred_reshaped = np.asarray([group_y_pred])
                ndcg = ndcg_score(y_true_reshaped, y_pred_reshaped, k=k)
                ndcg_scores[k].append(ndcg)
                
                # Calculate Precision@k (relevant items in top k / k)
                # Consider items with relevance >= 3 as relevant (scale 0-4)
                precision = sum(sorted_y_true[:k] >= 3) / k
                precision_scores[k].append(precision)
        
        start = end
    
    # Average metrics across all queries
    for k in k_values:
        results[f'ndcg@{k}'] = np.mean(ndcg_scores[k]) if ndcg_scores[k] else 0
        results[f'precision@{k}'] = np.mean(precision_scores[k]) if precision_scores[k] else 0
    
    return results

# Evaluate model performance
train_metrics = evaluate_ranking(y_train, preds_train, train_groups)
test_metrics = evaluate_ranking(y_test, preds_test, test_groups)

# Print results
print("\nTraining performance:")
for metric, value in train_metrics.items():
    print(f"  {metric}: {value:.4f}")

print("\nTest performance:")
for metric, value in test_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Visualize results
plt.figure(figsize=(14, 6))

# Plot NDCG scores
plt.subplot(1, 2, 1)
k_values = [1, 3, 5, 10]
train_ndcg = [train_metrics[f'ndcg@{k}'] for k in k_values]
test_ndcg = [test_metrics[f'ndcg@{k}'] for k in k_values]

x = np.arange(len(k_values))
width = 0.35
plt.bar(x - width/2, train_ndcg, width, label='Train')
plt.bar(x + width/2, test_ndcg, width, label='Test')
plt.xlabel('k')
plt.ylabel('NDCG')
plt.title('NDCG@k')
plt.xticks(x, [f'k={k}' for k in k_values])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

# Plot Precision scores
plt.subplot(1, 2, 2)
train_precision = [train_metrics[f'precision@{k}'] for k in k_values]
test_precision = [test_metrics[f'precision@{k}'] for k in k_values]

plt.bar(x - width/2, train_precision, width, label='Train')
plt.bar(x + width/2, test_precision, width, label='Test')
plt.xlabel('k')
plt.ylabel('Precision')
plt.title('Precision@k')
plt.xticks(x, [f'k={k}' for k in k_values])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

plt.tight_layout()
plt.show()

# Feature importance analysis
plt.figure(figsize=(12, 6))
xgb.plot_importance(model, max_num_features=15)
plt.title('Feature Importance for Ranking Model')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Analyze the effect of different ranking objectives
ranking_objectives = [
    'rank:pairwise',    # Pairwise ranking (optimize AUC)
    'rank:ndcg',        # Optimize NDCG
    'rank:map'          # Optimize MAP
]

results = {}

for objective in ranking_objectives:
    print(f"\nTraining with objective: {objective}")
    
    # Update parameters
    params['objective'] = objective
    
    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=20,
        verbose_eval=False
    )
    
    # Make predictions
    preds_test = model.predict(dtest)
    
    # Evaluate
    metrics = evaluate_ranking(y_test, preds_test, test_groups)
    results[objective] = metrics
    
    print(f"Performance with {objective}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

# Visualize comparison of different objectives
plt.figure(figsize=(15, 6))

# Plot NDCG comparison
plt.subplot(1, 2, 1)
x = np.arange(len(k_values))
width = 0.25

for i, (objective, metrics) in enumerate(results.items()):
    ndcg_values = [metrics[f'ndcg@{k}'] for k in k_values]
    plt.bar(x + (i-1)*width, ndcg_values, width, label=objective)

plt.xlabel('k')
plt.ylabel('NDCG')
plt.title('NDCG@k for Different Objectives')
plt.xticks(x, [f'k={k}' for k in k_values])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

# Plot Precision comparison
plt.subplot(1, 2, 2)
for i, (objective, metrics) in enumerate(results.items()):
    precision_values = [metrics[f'precision@{k}'] for k in k_values]
    plt.bar(x + (i-1)*width, precision_values, width, label=objective)

plt.xlabel('k')
plt.ylabel('Precision')
plt.title('Precision@k for Different Objectives')
plt.xticks(x, [f'k={k}' for k in k_values])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

plt.tight_layout()
plt.show()

# Analyze feature importance across different objectives
plt.figure(figsize=(15, 10))

for i, objective in enumerate(ranking_objectives):
    params['objective'] = objective
    model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
    
    plt.subplot(1, 3, i+1)
    xgb.plot_importance(model, max_num_features=10, ax=plt.gca())
    plt.title(f'Feature Importance - {objective}')

plt.tight_layout()
plt.show()

# Re-rank results with different lambdas for NDCG optimization
lambda_weights = [0.5, 1.0, 2.0, 5.0]
results_lambda = {}

# Train with different lambda weights
for lambda_weight in lambda_weights:
    params['objective'] = 'rank:ndcg'
    params['lambdarank_pair_method'] = 1  # Using ndcg-based lambda
    params['lambdarank_norm'] = True
    params['lambdarank_num_pair_per_sample'] = 8  # Num of pairs per document
    params['lambdarank_unbiased'] = False
    params['lambdarank_bias_norm'] = 1.0
    params['lambdarank_query_weight_scale'] = lambda_weight  # Lambda weight
    
    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=20,
        verbose_eval=False
    )
    
    # Make predictions
    preds_test = model.predict(dtest)
    
    # Evaluate
    metrics = evaluate_ranking(y_test, preds_test, test_groups)
    results_lambda[f'lambda={lambda_weight}'] = metrics

# Visualize lambda weight effect
plt.figure(figsize=(14, 6))

# Plot NDCG comparison
plt.subplot(1, 2, 1)
x = np.arange(len(k_values))
width = 0.2

for i, (lambda_name, metrics) in enumerate(results_lambda.items()):
    ndcg_values = [metrics[f'ndcg@{k}'] for k in k_values]
    plt.bar(x + (i-1.5)*width, ndcg_values, width, label=lambda_name)

plt.xlabel('k')
plt.ylabel('NDCG')
plt.title('NDCG@k for Different Lambda Weights')
plt.xticks(x, [f'k={k}' for k in k_values])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

# Plot Precision comparison
plt.subplot(1, 2, 2)
for i, (lambda_name, metrics) in enumerate(results_lambda.items()):
    precision_values = [metrics[f'precision@{k}'] for k in k_values]
    plt.bar(x + (i-1.5)*width, precision_values, width, label=lambda_name)

plt.xlabel('k')
plt.ylabel('Precision')
plt.title('Precision@k for Different Lambda Weights')
plt.xticks(x, [f'k={k}' for k in k_values])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

plt.tight_layout()
plt.show()
```

---

## 🔧 Performance Optimization

### Parallel Processing

XGBoost supports various forms of parallelization to speed up training:

```python
import xgboost as xgb
import numpy as np
import pandas as pd
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a large synthetic dataset
X, y = make_classification(
    n_samples=100000, n_features=50, n_informative=25, n_redundant=15,
    n_classes=2, random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to measure training time with different thread counts
def benchmark_threads(X_train, y_train, thread_counts):
    """Benchmark XGBoost training with different thread counts."""
    results = []
    
    for n_threads in thread_counts:
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Set parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'nthread': n_threads  # Set number of threads
        }
        
        # Measure training time
        start_time = time.time()
        model = xgb.train(params, dtrain, num_boost_round=100)
        training_time = time.time() - start_time
        
        results.append({
            'n_threads': n_threads,
            'training_time': training_time
        })
        
        print(f"Training with {n_threads} threads took {training_time:.2f} seconds")
    
    return results

# Benchmark with different thread counts
thread_counts = [1, 2, 4, 8, 16, -1]  # -1 means use all available threads
thread_results = benchmark_threads(X_train, y_train, thread_counts)

# Plot results
plt.figure(figsize=(10, 6))
threads = [result['n_threads'] for result in thread_results]
times = [result['training_time'] for result in thread_results]

# Replace -1 with actual thread count for display
threads = ['All' if t == -1 else str(t) for t in threads]

plt.bar(threads, times)
plt.xlabel('Number of Threads')
plt.ylabel('Training Time (seconds)')
plt.title('XGBoost Training Time vs. Number of Threads')
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

# Add time values on top of bars
for i, time_val in enumerate(times):
    plt.text(i, time_val + 0.1, f"{time_val:.2f}s", ha='center')

plt.tight_layout()
plt.show()

# Calculate speedup
base_time = thread_results[0]['training_time']  # Time with 1 thread
speedups = [base_time / result['training_time'] for result in thread_results]

plt.figure(figsize=(10, 6))
plt.bar(threads, speedups)
plt.xlabel('Number of Threads')
plt.ylabel('Speedup (relative to 1 thread)')
plt.title('XGBoost Training Speedup vs. Number of Threads')
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

# Add speedup values on top of bars
for i, speedup in enumerate(speedups):
    plt.text(i, speedup + 0.1, f"{speedup:.2f}x", ha='center')

plt.tight_layout()
plt.show()
```

### GPU Acceleration

XGBoost also supports GPU acceleration for faster training:

```python
import xgboost as xgb
import numpy as np
import pandas as pd
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a large synthetic dataset
X, y = make_classification(
    n_samples=100000, n_features=50, n_informative=25, n_redundant=15,
    n_classes=2, random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define parameters for CPU training
params_cpu = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist'  # Fastest CPU algorithm
}

# Define parameters for GPU training
params_gpu = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'gpu_hist'  # Use GPU
}

# Train on CPU and measure time
try:
    print("Training on CPU...")
    start_time = time.time()
    model_cpu = xgb.train(params_cpu, dtrain, num_boost_round=100)
    cpu_time = time.time() - start_time
    print(f"CPU training time: {cpu_time:.2f} seconds")
    
    # Train on GPU and measure time
    print("\nTraining on GPU...")
    try:
        start_time = time.time()
        model_gpu = xgb.train(params_gpu, dtrain, num_boost_round=100)
        gpu_time = time.time() - start_time
        print(f"GPU training time: {gpu_time:.2f} seconds")
        print(f"GPU speedup: {cpu_time / gpu_time:.2f}x")
        
        # Compare results
        cpu_pred = model_cpu.predict(dtest)
        gpu_pred = model_gpu.predict(dtest)
        
        # Check if predictions are similar
        prediction_diff = np.abs(cpu_pred - gpu_pred).mean()
        print(f"\nMean absolute difference in predictions: {prediction_diff:.6f}")
        
        # Plot training times
        plt.figure(figsize=(8, 6))
        plt.bar(['CPU', 'GPU'], [cpu_time, gpu_time])
        plt.ylabel('Training Time (seconds)')
        plt.title('XGBoost Training Time: CPU vs GPU')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add time values on top of bars
        plt.text(0, cpu_time + 0.1, f"{cpu_time:.2f}s", ha='center')
        plt.text(1, gpu_time + 0.1, f"{gpu_time:.2f}s", ha='center')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"GPU training failed: {e}")
        print("This could be because no GPU is available or XGBoost was not built with GPU support.")
        print("Continuing with CPU benchmarks only.")
        gpu_time = None
except Exception as e:
    print(f"Error during training: {e}")
```

### Memory Optimization

XGBoost offers several techniques to reduce memory usage for large datasets:

```python
import xgboost as xgb
import numpy as np
import pandas as pd
import time
import os
import psutil
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Function to get memory usage
def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB

# Generate a large synthetic dataset
X, y = make_classification(
    n_samples=500000, n_features=50, n_informative=25, n_redundant=15,
    n_classes=2, random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Benchmark different memory optimization techniques
print("Initial memory usage: {:.2f} MB".format(get_memory_usage()))

# 1. Standard training (baseline)
print("\n1. Standard training (baseline)")
memory_before = get_memory_usage()
start_time = time.time()

dtrain = xgb.DMatrix(X_train, label=y_train)
params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'eta': 0.1
}
model = xgb.train(params, dtrain, num_boost_round=100)

end_time = time.time()
memory_after = get_memory_usage()
print("  Training time: {:.2f} seconds".format(end_time - start_time))
print("  Memory increase: {:.2f} MB".format(memory_after - memory_before))

# 2. External memory mode
print("\n2. External memory mode")
# Create a temporary file
temp_file = 'temp_xgb_data.libsvm'

# Function to save data in LibSVM format
def save_to_libsvm(X, y, filename):
    with open(filename, 'w') as f:
        for i in range(len(y)):
            f.write(f"{int(y[i])}")
            for j in range(X.shape[1]):
                f.write(f" {j+1}:{X[i, j]}")
            f.write("\n")

# Save data to disk
save_to_libsvm(X_train, y_train, temp_file)

# Train using external memory
memory_before = get_memory_usage()
start_time = time.time()

dtrain = xgb.DMatrix(f"{temp_file}?format=libsvm")
model = xgb.train(params, dtrain, num_boost_round=100)

end_time = time.time()
memory_after = get_memory_usage()
print("  Training time: {:.2f} seconds".format(end_time - start_time))
print("  Memory increase: {:.2f} MB".format(memory_after - memory_before))

# Clean up temp file
os.remove(temp_file)

# 3. Subsample and grow
print("\n3. Subsample and grow approach")
memory_before = get_memory_usage()
start_time = time.time()

# Divide data into chunks
chunk_size = 100000
n_chunks = (len(X_train) + chunk_size - 1) // chunk_size
models = []

for i in range(n_chunks):
    print(f"  Training on chunk {i+1}/{n_chunks}")
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(X_train))
    
    # Create DMatrix for this chunk
    X_chunk = X_train[start_idx:end_idx]
    y_chunk = y_train[start_idx:end_idx]
    dtrain_chunk = xgb.DMatrix(X_chunk, label=y_chunk)
    
    # Train model on this chunk
    model_chunk = xgb.train(params, dtrain_chunk, num_boost_round=100 // n_chunks)
    models.append(model_chunk)

end_time = time.time()
memory_after = get_memory_usage()
print("  Training time: {:.2f} seconds".format(end_time - start_time))
print("  Memory increase: {:.2f} MB".format(memory_after - memory_before))

# 4. Sparse matrices for datasets with many zeros
from scipy import sparse

print("\n4. Using sparse matrices")
# Create a sparse version of the data (artificially setting 70% of values to 0)
mask = np.random.random(X_train.shape) > 0.7
X_train_sparse = X_train.copy()
X_train_sparse[mask] = 0
X_train_sparse = sparse.csr_matrix(X_train_sparse)

memory_before = get_memory_usage()
start_time = time.time()

dtrain = xgb.DMatrix(X_train_sparse, label=y_train)
model = xgb.train(params, dtrain, num_boost_round=100)

end_time = time.time()
memory_after = get_memory_usage()
print("  Training time: {:.2f} seconds".format(end_time - start_time))
print("  Memory increase: {:.2f} MB".format(memory_after - memory_before))

# Summary
print("\nMemory Optimization Techniques Summary:")
print("1. Standard approach: Easiest to implement but uses the most memory")
print("2. External memory: Reduces memory usage but slower due to disk I/O")
print("3. Subsample and grow: Trains on data chunks, good for very large datasets")
print("4. Sparse matrices: Efficient for datasets with many zeros")
print("\nAdditional Tips:")
print("- Use single precision (32-bit) instead of double precision (64-bit)")
print("- Use histogram-based training (tree_method='hist' or 'gpu_hist')")
print("- Reduce max_bin parameter to use less memory")
print("- Sample data before training if full precision isn't required")
```

---

## 🔄 Comparison with Other Methods

### XGBoost vs. Random Forest

```python
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import seaborn as sns

# Load a real dataset
breast_cancer = load_breast_cancer()
X_real, y_real = breast_cancer.data, breast_cancer.target

# Generate a synthetic dataset
X_synth, y_synth = make_classification(
    n_samples=1000, n_features=20, n_informative=10, n_redundant=5,
    n_classes=2, random_state=42
)

# Function to compare XGBoost and Random Forest
def compare_models(X, y, dataset_name):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=42
    )
    
    # Train and evaluate models
    results = {}
    
    for name, model in [('XGBoost', xgb_model), ('Random Forest', rf_model)]:
        # Measure training time
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # Store results
        results[name] = {
            'training_time': training_time,
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'model': model
        }
    
    # Calculate learning curves
    train_sizes = np.linspace(0.1, 1.0, 5)
    
    for name, result in results.items():
        train_sizes_abs, train_scores, test_scores = learning_curve(
            result['model'], X, y, train_sizes=train_sizes, cv=5, scoring='accuracy'
        )
        
        result['learning_curve'] = {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'test_scores_mean': np.mean(test_scores, axis=1),
            'test_scores_std': np.std(test_scores, axis=1)
        }
    
    # Print results
    print(f"\nResults for {dataset_name} dataset:")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Training Time: {result['training_time']:.4f} seconds")
        print(f"  Test Accuracy: {result['accuracy']:.4f}")
        print(f"  AUC: {result['auc']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall: {result['recall']:.4f}")
        print(f"  F1 Score: {result['f1']:.4f}")
        print(f"  CV Accuracy: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 12))
    
    # Plot training time
    plt.subplot(2, 2, 1)
    plt.bar(['XGBoost', 'Random Forest'], [results['XGBoost']['training_time'], results['Random Forest']['training_time']])
    plt.title('Training Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot metrics
    plt.subplot(2, 2, 2)
    metrics = ['accuracy', 'auc', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics))
    width = 0.35
    
    xgb_values = [results['XGBoost'][metric] for metric in metrics]
    rf_values = [results['Random Forest'][metric] for metric in metrics]
    
    plt.bar(x - width/2, xgb_values, width, label='XGBoost')
    plt.bar(x + width/2, rf_values, width, label='Random Forest')
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Performance Metrics')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot learning curves
    plt.subplot(2, 2, 3)
    for name, result in results.items():
        lc = result['learning_curve']
        plt.plot(lc['train_sizes'], lc['test_scores_mean'], 'o-', label=f'{name} CV')
        plt.fill_between(lc['train_sizes'], 
                        lc['test_scores_mean'] - lc['test_scores_std'],
                        lc['test_scores_mean'] + lc['test_scores_std'],
                        alpha=0.1)
    
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot feature importance
    plt.subplot(2, 2, 4)
    
    # Get feature importance
    xgb_importance = results['XGBoost']['model'].feature_importances_
    rf_importance = results['Random Forest']['model'].feature_importances_
    
    # Sort features by average importance
    avg_importance = (xgb_importance + rf_importance) / 2
    sorted_idx = np.argsort(avg_importance)[-10:]  # Top 10 features
    
    # Create DataFrame for easier plotting
    feature_names = [f'Feature {i}' for i in sorted_idx]
    importance_df = pd.DataFrame({
        'Feature': np.repeat(feature_names, 2),
        'Importance': np.concatenate([xgb_importance[sorted_idx], rf_importance[sorted_idx]]),
        'Model': ['XGBoost'] * len(sorted_idx) + ['Random Forest'] * len(sorted_idx)
    })
    
    # Plot with seaborn
    sns.barplot(x='Importance', y='Feature', hue='Model', data=importance_df)
    plt.title('Feature Importance Comparison')
    plt.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    plt.tight_layout()
    plt.suptitle(f'XGBoost vs. Random Forest on {dataset_name} Dataset', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    return results

# Compare on both datasets
real_results = compare_models(X_real, y_real, "Breast Cancer")
synth_results = compare_models(X_synth, y_synth, "Synthetic")

# Parameter sensitivity analysis
def parameter_sensitivity(X, y, dataset_name):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Parameters to test
    n_estimators_list = [10, 50, 100, 200, 500]
    max_depth_list = [3, 5, 7, 10, None]  # None means unlimited depth
    
    # Store results
    xgb_results = np.zeros((len(n_estimators_list), len(max_depth_list)))
    rf_results = np.zeros((len(n_estimators_list), len(max_depth_list)))
    xgb_times = np.zeros((len(n_estimators_list), len(max_depth_list)))
    rf_times = np.zeros((len(n_estimators_list), len(max_depth_list)))
    
    # Test each parameter combination
    for i, n_estimators in enumerate(n_estimators_list):
        for j, max_depth in enumerate(max_depth_list):
            # XGBoost
            xgb_model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth if max_depth is not None else 10,  # XGBoost needs a numeric value
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            # Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            
            # Train and evaluate XGBoost
            start_time = time.time()
            xgb_model.fit(X_train, y_train)
            xgb_time = time.time() - start_time
            xgb_score = xgb_model.score(X_test, y_test)
            
            # Train and evaluate Random Forest
            start_time = time.time()
            rf_model.fit(X_train, y_train)
            rf_time = time.time() - start_time
            rf_score = rf_model.score(X_test, y_test)
            
            # Store results
            xgb_results[i, j] = xgb_score
            rf_results[i, j] = rf_score
            xgb_times[i, j] = xgb_time
            rf_times[i, j] = rf_time
    
    # Create heatmaps
    plt.figure(figsize=(15, 10))
    
    # XGBoost accuracy
    plt.subplot(2, 2, 1)
    sns.heatmap(xgb_results, annot=True, fmt='.4f', cmap='viridis',
               xticklabels=max_depth_list, yticklabels=n_estimators_list)
    plt.xlabel('max_depth')
    plt.ylabel('n_estimators')
    plt.title('XGBoost Accuracy')
    
    # Random Forest accuracy
    plt.subplot(2, 2, 2)
    sns.heatmap(rf_results, annot=True, fmt='.4f', cmap='viridis',
               xticklabels=max_depth_list, yticklabels=n_estimators_list)
    plt.xlabel('max_depth')
    plt.ylabel('n_estimators')
    plt.title('Random Forest Accuracy')
    
    # XGBoost training time
    plt.subplot(2, 2, 3)
    sns.heatmap(xgb_times, annot=True, fmt='.2f', cmap='plasma',
               xticklabels=max_depth_list, yticklabels=n_estimators_list)
    plt.xlabel('max_depth')
    plt.ylabel('n_estimators')
    plt.title('XGBoost Training Time (seconds)')
    
    # Random Forest training time
    plt.subplot(2, 2, 4)
    sns.heatmap(rf_times, annot=True, fmt='.2f', cmap='plasma',
               xticklabels=max_depth_list, yticklabels=n_estimators_list)
    plt.xlabel('max_depth')
    plt.ylabel('n_estimators')
    plt.title('Random Forest Training Time (seconds)')
    
    plt.tight_layout()
    plt.suptitle(f'Parameter Sensitivity on {dataset_name} Dataset', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    return {
        'xgb_results': xgb_results,
        'rf_results': rf_results,
        'xgb_times': xgb_times,
        'rf_times': rf_times
    }

# Run parameter sensitivity analysis on the real dataset
sensitivity_results = parameter_sensitivity(X_real, y_real, "Breast Cancer")

# Model complexity vs. performance comparison
def complexity_performance_tradeoff(X, y, dataset_name):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Parameters related to model complexity
    n_estimators_list = [1, 5, 10, 50, 100, 200, 500]
    
    # Store results
    xgb_train_scores = []
    xgb_test_scores = []
    xgb_times = []
    rf_train_scores = []
    rf_test_scores = []
    rf_times = []
    
    # Test each parameter value
    for n_estimators in n_estimators_list:
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            random_state=42
        )
        
        # Train and evaluate XGBoost
        start_time = time.time()
        xgb_model.fit(X_train, y_train)
        xgb_time = time.time() - start_time
        xgb_train_score = xgb_model.score(X_train, y_train)
        xgb_test_score = xgb_model.score(X_test, y_test)
        
        # Train and evaluate Random Forest
        start_time = time.time()
        rf_model.fit(X_train, y_train)
        rf_time = time.time() - start_time
        rf_train_score = rf_model.score(X_train, y_train)
        rf_test_score = rf_model.score(X_test, y_test)
        
        # Store results
        xgb_train_scores.append(xgb_train_score)
        xgb_test_scores.append(xgb_test_score)
        xgb_times.append(xgb_time)
        rf_train_scores.append(rf_train_score)
        rf_test_scores.append(rf_test_score)
        rf_times.append(rf_time)
    
    # Create plots
    plt.figure(figsize=(15, 8))
    
    # Train vs. Test Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(n_estimators_list, xgb_train_scores, 'o-', label='XGBoost Train')
    plt.plot(n_estimators_list, xgb_test_scores, 's-', label='XGBoost Test')
    plt.plot(n_estimators_list, rf_train_scores, 'o--', label='Random Forest Train')
    plt.plot(n_estimators_list, rf_test_scores, 's--', label='Random Forest Test')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title('Train vs. Test Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Training Time
    plt.subplot(1, 2, 2)
    plt.plot(n_estimators_list, xgb_times, 'o-', label='XGBoost')
    plt.plot(n_estimators_list, rf_times, 's-', label='Random Forest')
    plt.xlabel('Number of Trees')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs. Model Complexity')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.suptitle(f'Model Complexity Analysis on {dataset_name} Dataset', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    return {
        'n_estimators': n_estimators_list,
        'xgb_train_scores': xgb_train_scores,
        'xgb_test_scores': xgb_test_scores,
        'xgb_times': xgb_times,
        'rf_train_scores': rf_train_scores,
        'rf_test_scores': rf_test_scores,
        'rf_times': rf_times
    }

# Run complexity vs. performance analysis
complexity_results = complexity_performance_tradeoff(X_real, y_real, "Breast Cancer")
```

### XGBoost vs. Gradient Boosting

```python
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
import seaborn as sns

# Load datasets
# Classification dataset
breast_cancer = load_breast_cancer()
X_class, y_class = breast_cancer.data, breast_cancer.target

# Regression dataset
california = fetch_california_housing()
X_reg, y_reg = california.data, california.target

# Function to compare XGBoost and Gradient Boosting
def compare_gbm_models(X, y, task='classification', dataset_name=''):
    """Compare XGBoost and Gradient Boosting Machine (sklearn)."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    if task == 'classification':
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        gbm_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        scoring = 'accuracy'
        
    else:  # regression
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        gbm_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        scoring = 'neg_mean_squared_error'
    
    # Train and evaluate models
    results = {}
    
    for name, model in [('XGBoost', xgb_model), ('GBM', gbm_model)]:
        # Measure training time
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if task == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            
            metrics = {
                'accuracy': accuracy,
                'auc': auc
            }
            
        else:  # regression
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # Store results
        results[name] = {
            'training_time': training_time,
            'metrics': metrics,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'model': model
        }
    
    # Training curves (learning rate)
    xgb_training_curve = []
    gbm_training_curve = []
    
    # Create a new model for training curve
    if task == 'classification':
        xgb_curve_model = xgb.XGBClassifier(
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        gbm_curve_model = GradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    else:
        xgb_curve_model = xgb.XGBRegressor(
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        gbm_curve_model = GradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    
    # Set evaluation sets for XGBoost
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    # Train XGBoost with evaluation
    xgb_curve_model.fit(
        X_train, y_train,
        eval_set=eval_set,
        eval_metric='logloss' if task == 'classification' else 'rmse',
        verbose=False
    )
    
    # Get XGBoost training history
    results_dict = xgb_curve_model.evals_result()
    xgb_train_curve = results_dict['validation_0']['logloss' if task == 'classification' else 'rmse']
    xgb_test_curve = results_dict['validation_1']['logloss' if task == 'classification' else 'rmse']
    
    # Train GBM with staged predictions
    gbm_curve_model.fit(X_train, y_train)
    
    # Get GBM training history using staged predictions
    if task == 'classification':
        gbm_staged_train = list(gbm_curve_model.staged_predict_proba(X_train))
        gbm_staged_test = list(gbm_curve_model.staged_predict_proba(X_test))
        
        # Convert to loss
        gbm_train_curve = [log_loss(y_train, pred[:, 1]) for pred in gbm_staged_train]
        gbm_test_curve = [log_loss(y_test, pred[:, 1]) for pred in gbm_staged_test]
    else:
        gbm_staged_train = list(gbm_curve_model.staged_predict(X_train))
        gbm_staged_test = list(gbm_curve_model.staged_predict(X_test))
        
        # Convert to RMSE
        gbm_train_curve = [np.sqrt(mean_squared_error(y_train, pred)) for pred in gbm_staged_train]
        gbm_test_curve = [np.sqrt(mean_squared_error(y_test, pred)) for pred in gbm_staged_test]
    
    # Store training curves
    results['XGBoost']['training_curve'] = {
        'train': xgb_train_curve,
        'test': xgb_test_curve
    }
    
    results['GBM']['training_curve'] = {
        'train': gbm_train_curve,
        'test': gbm_test_curve
    }
    
    # Print results
    print(f"\nResults for {dataset_name} ({task}):")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Training Time: {result['training_time']:.4f} seconds")
        
        for metric_name, metric_value in result['metrics'].items():
            print(f"  {metric_name.upper()}: {metric_value:.4f}")
        
        if task == 'classification':
            print(f"  CV Accuracy: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
        else:
            print(f"  CV Neg MSE: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 12))
    
    # Plot training time
    plt.subplot(2, 2, 1)
    plt.bar(['XGBoost', 'GBM'], [results['XGBoost']['training_time'], results['GBM']['training_time']])
    plt.title('Training Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot metrics
    plt.subplot(2, 2, 2)
    metrics_names = list(results['XGBoost']['metrics'].keys())
    x = np.arange(len(metrics_names))
    width = 0.35
    
    xgb_values = [results['XGBoost']['metrics'][m] for m in metrics_names]
    gbm_values = [results['GBM']['metrics'][m] for m in metrics_names]
    
    plt.bar(x - width/2, xgb_values, width, label='XGBoost')
    plt.bar(x + width/2, gbm_values, width, label='GBM')
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Performance Metrics')
    plt.xticks(x, metrics_names)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot training curves
    plt.subplot(2, 2, 3)
    iterations = range(1, len(results['XGBoost']['training_curve']['train']) + 1)
    
    plt.plot(iterations, results['XGBoost']['training_curve']['train'], 'b-', label='XGBoost Train')
    plt.plot(iterations, results['XGBoost']['training_curve']['test'], 'b--', label='XGBoost Test')
    
    # Adjust for potentially different lengths
    gbm_iterations = range(1, len(results['GBM']['training_curve']['train']) + 1)
    plt.plot(gbm_iterations, results['GBM']['training_curve']['train'], 'r-', label='GBM Train')
    plt.plot(gbm_iterations, results['GBM']['training_curve']['test'], 'r--', label='GBM Test')
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss' if task == 'classification' else 'RMSE')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot feature importance
    plt.subplot(2, 2, 4)
    
    # Get feature importance
    xgb_importance = results['XGBoost']['model'].feature_importances_
    gbm_importance = results['GBM']['model'].feature_importances_
    
    # Sort features by average importance
    avg_importance = (xgb_importance + gbm_importance) / 2
    sorted_idx = np.argsort(avg_importance)[-10:]  # Top 10 features
    
    # Get feature names
    if dataset_name == 'Breast Cancer':
        feature_names = breast_cancer.feature_names
    elif dataset_name == 'California Housing':
        feature_names = california.feature_names
    else:
        feature_names = [f'Feature {i}' for i in range(len(xgb_importance))]
    
    # Create DataFrame for easier plotting
    selected_features = [feature_names[i] for i in sorted_idx]
    importance_df = pd.DataFrame({
        'Feature': np.repeat(selected_features, 2),
        'Importance': np.concatenate([xgb_importance[sorted_idx], gbm_importance[sorted_idx]]),
        'Model': ['XGBoost'] * len(sorted_idx) + ['GBM'] * len(sorted_idx)
    })
    
    # Plot with seaborn
    sns.barplot(x='Importance', y='Feature', hue='Model', data=importance_df)
    plt.title('Feature Importance Comparison')
    plt.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    plt.tight_layout()
    plt.suptitle(f'XGBoost vs. Gradient Boosting on {dataset_name} Dataset', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    return results

# Compare on both datasets
class_results = compare_gbm_models(X_class, y_class, 'classification', 'Breast Cancer')
reg_results = compare_gbm_models(X_reg, y_reg, 'regression', 'California Housing')

# Learning rate comparison
def learning_rate_comparison(X, y, task='classification', dataset_name=''):
    """Compare XGBoost and GBM with different learning rates."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Learning rates to test
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
    
    # Store results
    xgb_scores = []
    gbm_scores = []
    xgb_times = []
    gbm_times = []
    
    for lr in learning_rates:
        # Initialize models
        if task == 'classification':
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=lr,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            gbm_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=lr,
                subsample=0.8,
                random_state=42
            )
            
            metric_func = accuracy_score
            
        else:  # regression
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=lr,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            gbm_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=lr,
                subsample=0.8,
                random_state=42
            )
            
            metric_func = r2_score
        
        # Train and evaluate XGBoost
        start_time = time.time()
        xgb_model.fit(X_train, y_train)
        xgb_time = time.time() - start_time
        
        xgb_pred = xgb_model.predict(X_test)
        xgb_score = metric_func(y_test, xgb_pred)
        
        # Train and evaluate GBM
        start_time = time.time()
        gbm_model.fit(X_train, y_train)
                gbm_time = time.time() - start_time
        
        gbm_pred = gbm_model.predict(X_test)
        gbm_score = metric_func(y_test, gbm_pred)
        
        # Store results
        xgb_scores.append(xgb_score)
        gbm_scores.append(gbm_score)
        xgb_times.append(xgb_time)
        gbm_times.append(gbm_time)
    
    # Plot results
    plt.figure(figsize=(15, 6))
    
    # Plot scores
    plt.subplot(1, 2, 1)
    plt.plot(learning_rates, xgb_scores, 'o-', label='XGBoost')
    plt.plot(learning_rates, gbm_scores, 's-', label='GBM')
    plt.xlabel('Learning Rate')
    plt.ylabel('Score')
    plt.title(f'Performance vs. Learning Rate ({dataset_name})')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot training times
    plt.subplot(1, 2, 2)
    plt.plot(learning_rates, xgb_times, 'o-', label='XGBoost')
    plt.plot(learning_rates, gbm_times, 's-', label='GBM')
    plt.xlabel('Learning Rate')
    plt.ylabel('Training Time (seconds)')
    plt.title(f'Training Time vs. Learning Rate ({dataset_name})')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'learning_rates': learning_rates,
        'xgb_scores': xgb_scores,
        'gbm_scores': gbm_scores,
        'xgb_times': xgb_times,
        'gbm_times': gbm_times
    }

# Run learning rate comparison
lr_class_results = learning_rate_comparison(X_class, y_class, 'classification', 'Breast Cancer')
lr_reg_results = learning_rate_comparison(X_reg, y_reg, 'regression', 'California Housing')
```

### XGBoost vs. LightGBM

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import xgboost as xgb
import lightgbm as lgb
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Load datasets
diabetes = load_diabetes()
X_diabetes, y_diabetes = diabetes.data, diabetes.target

california = fetch_california_housing()
X_california, y_california = california.data, california.target

# Function to compare XGBoost and LightGBM
def compare_xgb_lightgbm(X, y, dataset_name):
    """Compare XGBoost and LightGBM performance and characteristics."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Train and evaluate models
    results = {}
    
    for name, model in [('XGBoost', xgb_model), ('LightGBM', lgb_model)]:
        # Measure training time
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Measure prediction time
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        cv_std = np.sqrt(cv_scores.std())
        
        # Store model and metrics
        results[name] = {
            'model': model,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'cv_rmse': cv_rmse,
            'cv_std': cv_std
        }
    
    # Print results
    print(f"\nResults for {dataset_name} dataset:")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Training Time: {result['training_time']:.4f} seconds")
        print(f"  Prediction Time: {result['prediction_time']:.4f} seconds")
        print(f"  RMSE: {result['rmse']:.4f}")
        print(f"  R²: {result['r2']:.4f}")
        print(f"  CV RMSE: {result['cv_rmse']:.4f} ± {result['cv_std']:.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot training and prediction times
    plt.subplot(2, 2, 1)
    time_data = pd.DataFrame({
        'Model': ['XGBoost', 'LightGBM', 'XGBoost', 'LightGBM'],
        'Time (seconds)': [
            results['XGBoost']['training_time'], 
            results['LightGBM']['training_time'],
            results['XGBoost']['prediction_time'], 
            results['LightGBM']['prediction_time']
        ],
        'Type': ['Training', 'Training', 'Prediction', 'Prediction']
    })
    sns.barplot(x='Model', y='Time (seconds)', hue='Type', data=time_data)
    plt.title('Training and Prediction Times')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot RMSE and R²
    plt.subplot(2, 2, 2)
    metrics_data = pd.DataFrame({
        'Model': ['XGBoost', 'LightGBM', 'XGBoost', 'LightGBM'],
        'Value': [
            results['XGBoost']['rmse'], 
            results['LightGBM']['rmse'],
            results['XGBoost']['r2'], 
            results['LightGBM']['r2']
        ],
        'Metric': ['RMSE', 'RMSE', 'R²', 'R²']
    })
    sns.barplot(x='Model', y='Value', hue='Metric', data=metrics_data)
    plt.title('Performance Metrics')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot feature importance
    plt.subplot(2, 2, 3)
    xgb_importance = results['XGBoost']['model'].feature_importances_
    lgb_importance = results['LightGBM']['model'].feature_importances_
    
    # Get feature names
    if dataset_name == 'Diabetes':
        feature_names = diabetes.feature_names
    elif dataset_name == 'California Housing':
        feature_names = california.feature_names
    else:
        feature_names = [f'Feature {i}' for i in range(len(xgb_importance))]
    
    # Sort by average importance
    avg_importance = (xgb_importance + lgb_importance) / 2
    sorted_idx = np.argsort(avg_importance)[-10:]  # Top 10 features
    
    # Create DataFrame for easier plotting
    selected_features = [feature_names[i] for i in sorted_idx]
    importance_df = pd.DataFrame({
        'Feature': np.repeat(selected_features, 2),
        'Importance': np.concatenate([xgb_importance[sorted_idx], lgb_importance[sorted_idx]]),
        'Model': ['XGBoost'] * len(sorted_idx) + ['LightGBM'] * len(sorted_idx)
    })
    
    # Plot with seaborn
    sns.barplot(x='Importance', y='Feature', hue='Model', data=importance_df)
    plt.title('Feature Importance Comparison')
    plt.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    # Compare memory usage
    plt.subplot(2, 2, 4)
    # Function to get model size
    def get_model_size(model):
        import pickle
        import sys
        return sys.getsizeof(pickle.dumps(model)) / (1024 * 1024)  # Size in MB
    
    xgb_size = get_model_size(results['XGBoost']['model'])
    lgb_size = get_model_size(results['LightGBM']['model'])
    
    plt.bar(['XGBoost', 'LightGBM'], [xgb_size, lgb_size])
    plt.ylabel('Model Size (MB)')
    plt.title('Model Size Comparison')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add size values on top of bars
    plt.text(0, xgb_size + 0.1, f"{xgb_size:.2f} MB", ha='center')
    plt.text(1, lgb_size + 0.1, f"{lgb_size:.2f} MB", ha='center')
    
    plt.tight_layout()
    plt.suptitle(f'XGBoost vs. LightGBM on {dataset_name} Dataset', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    return results

# Compare models on both datasets
diabetes_results = compare_xgb_lightgbm(X_diabetes, y_diabetes, 'Diabetes')
california_results = compare_xgb_lightgbm(X_california, y_california, 'California Housing')

# Compare training speed with different dataset sizes
def compare_scaling_performance(X, y, dataset_name):
    """Compare how XGBoost and LightGBM scale with dataset size."""
    # Create different dataset sizes
    sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
    xgb_times = []
    lgb_times = []
    
    # Full dataset size
    full_size = len(X)
    
    for size in sizes:
        n_samples = int(full_size * size)
        
        # Sample data
        if size < 1.0:
            indices = np.random.choice(full_size, n_samples, replace=False)
            X_subset = X[indices]
            y_subset = y[indices]
        else:
            X_subset = X
            y_subset = y
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)
        
        # Initialize models
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # Train XGBoost and measure time
        start_time = time.time()
        xgb_model.fit(X_train, y_train)
        xgb_time = time.time() - start_time
        
        # Train LightGBM and measure time
        start_time = time.time()
        lgb_model.fit(X_train, y_train)
        lgb_time = time.time() - start_time
        
        # Store times
        xgb_times.append(xgb_time)
        lgb_times.append(lgb_time)
        
        print(f"Dataset size: {n_samples} samples")
        print(f"  XGBoost training time: {xgb_time:.4f} seconds")
        print(f"  LightGBM training time: {lgb_time:.4f} seconds")
        print(f"  Speedup: {xgb_time / lgb_time:.2f}x")
    
    # Plot scaling behavior
    plt.figure(figsize=(12, 6))
    
    # Convert sizes to actual sample counts
    sample_counts = [int(full_size * s) for s in sizes]
    
    plt.subplot(1, 2, 1)
    plt.plot(sample_counts, xgb_times, 'o-', label='XGBoost')
    plt.plot(sample_counts, lgb_times, 's-', label='LightGBM')
    plt.xlabel('Dataset Size (samples)')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs. Dataset Size')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    speedups = [xgb_time / lgb_time for xgb_time, lgb_time in zip(xgb_times, lgb_times)]
    plt.plot(sample_counts, speedups, 'o-')
    plt.xlabel('Dataset Size (samples)')
    plt.ylabel('Speedup (XGBoost time / LightGBM time)')
    plt.title('LightGBM Speedup vs. Dataset Size')
    plt.axhline(y=1, color='r', linestyle='--', label='Equal Performance')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.suptitle(f'Scaling Performance on {dataset_name} Dataset', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    return {
        'sizes': sample_counts,
        'xgb_times': xgb_times,
        'lgb_times': lgb_times
    }

# Compare scaling performance
california_scaling = compare_scaling_performance(X_california, y_california, 'California Housing')

# Compare tree growth strategies
def compare_tree_growth(X, y, dataset_name):
    """Compare leaf-wise (LightGBM) vs. level-wise (XGBoost) tree growth."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Different max_depth settings to compare
    max_depths = [3, 6, 10, 15, 20]
    
    # Store results
    xgb_scores = []
    lgb_scores = []
    xgb_times = []
    lgb_times = []
    xgb_leaves = []
    lgb_leaves = []
    
    for depth in max_depths:
        # XGBoost model
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=depth,
            learning_rate=0.1,
            random_state=42
        )
        
        # LightGBM model
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=depth,
            learning_rate=0.1,
            random_state=42
        )
        
        # Train and evaluate XGBoost
        start_time = time.time()
        xgb_model.fit(X_train, y_train)
        xgb_time = time.time() - start_time
        
        xgb_pred = xgb_model.predict(X_test)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        
        # Train and evaluate LightGBM
        start_time = time.time()
        lgb_model.fit(X_train, y_train)
        lgb_time = time.time() - start_time
        
        lgb_pred = lgb_model.predict(X_test)
        lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))
        
        # Count number of leaves (approximate for XGBoost)
        xgb_n_leaves = 2**depth  # Maximum possible for a balanced tree
        lgb_n_leaves = sum(tree.num_leaves for tree in lgb_model.booster_.dump_model()['tree_info'])
        
        # Store results
        xgb_scores.append(xgb_rmse)
        lgb_scores.append(lgb_rmse)
        xgb_times.append(xgb_time)
        lgb_times.append(lgb_time)
        xgb_leaves.append(xgb_n_leaves)
        lgb_leaves.append(lgb_n_leaves)
        
        print(f"Max depth: {depth}")
        print(f"  XGBoost RMSE: {xgb_rmse:.4f}, Time: {xgb_time:.4f}s, Est. Leaves: {xgb_n_leaves}")
        print(f"  LightGBM RMSE: {lgb_rmse:.4f}, Time: {lgb_time:.4f}s, Leaves: {lgb_n_leaves}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot RMSE vs. max_depth
    plt.subplot(2, 2, 1)
    plt.plot(max_depths, xgb_scores, 'o-', label='XGBoost')
    plt.plot(max_depths, lgb_scores, 's-', label='LightGBM')
    plt.xlabel('Max Depth')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. Max Depth')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot training time vs. max_depth
    plt.subplot(2, 2, 2)
    plt.plot(max_depths, xgb_times, 'o-', label='XGBoost')
    plt.plot(max_depths, lgb_times, 's-', label='LightGBM')
    plt.xlabel('Max Depth')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs. Max Depth')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot number of leaves vs. max_depth
    plt.subplot(2, 2, 3)
    plt.plot(max_depths, xgb_leaves, 'o-', label='XGBoost (estimated)')
    plt.plot(max_depths, lgb_leaves, 's-', label='LightGBM')
    plt.xlabel('Max Depth')
    plt.ylabel('Number of Leaves')
    plt.title('Tree Structure Comparison')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot efficiency (RMSE improvement per leaf)
    plt.subplot(2, 2, 4)
    xgb_efficiency = [(max(xgb_scores) - score) / leaves for score, leaves in zip(xgb_scores, xgb_leaves)]
    lgb_efficiency = [(max(lgb_scores) - score) / leaves for score, leaves in zip(lgb_scores, lgb_leaves)]
    
    plt.plot(max_depths, xgb_efficiency, 'o-', label='XGBoost')
    plt.plot(max_depths, lgb_efficiency, 's-', label='LightGBM')
    plt.xlabel('Max Depth')
    plt.ylabel('Efficiency (RMSE improvement per leaf)')
    plt.title('Tree Growth Efficiency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.suptitle(f'Tree Growth Strategy Comparison on {dataset_name} Dataset', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    return {
        'max_depths': max_depths,
        'xgb_scores': xgb_scores,
        'lgb_scores': lgb_scores,
        'xgb_times': xgb_times,
        'lgb_times': lgb_times,
        'xgb_leaves': xgb_leaves,
        'lgb_leaves': lgb_leaves
    }

# Compare tree growth strategies
tree_growth_results = compare_tree_growth(X_california, y_california, 'California Housing')
```

---

## ⚠️ Common Pitfalls

### Overfitting

```python
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic data with some noise
X, y = make_regression(n_samples=1000, n_features=20, n_informative=10, 
                     noise=50, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to demonstrate overfitting
def demonstrate_overfitting():
    """Show how XGBoost can overfit and how to prevent it."""
    # Models with different complexity settings
    models = {
        'Overfit (deep trees)': xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=15,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0,
            reg_lambda=0,
            random_state=42
        ),
        'Balanced model': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        ),
        'Underfit (shallow trees)': xgb.XGBRegressor(
            n_estimators=10,
            max_depth=2,
            learning_rate=0.1,
            subsample=0.5,
            colsample_bytree=0.5,
            reg_alpha=10,
            reg_lambda=10,
            random_state=42
        )
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        # Train with eval set to track progress
        eval_set = [(X_train, y_train), (X_test, y_test)]
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric='rmse',
            verbose=False
        )
        
        # Get evaluation results
        evals_result = model.evals_result()
        train_rmse = evals_result['validation_0']['rmse']
        test_rmse = evals_result['validation_1']['rmse']
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate final metrics
        train_final_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_final_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        # Store results
        results[name] = {
            'train_rmse_history': train_rmse,
            'test_rmse_history': test_rmse,
            'train_final_rmse': train_final_rmse,
            'test_final_rmse': test_final_rmse,
            'gap': train_final_rmse - test_final_rmse,
            'model': model
        }
        
        print(f"{name}:")
        print(f"  Training RMSE: {train_final_rmse:.2f}")
        print(f"  Test RMSE: {test_final_rmse:.2f}")
        print(f"  Gap: {abs(train_final_rmse - test_final_rmse):.2f}")
    
    # Visualize training curves
    plt.figure(figsize=(15, 10))
    
    # Plot training curves
    plt.subplot(2, 2, 1)
    for name, result in results.items():
        plt.plot(result['train_rmse_history'], '--', label=f'{name} (Train)')
        plt.plot(result['test_rmse_history'], '-', label=f'{name} (Test)')
    
    plt.xlabel('Boosting Iterations')
    plt.ylabel('RMSE')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot final RMSE comparison
    plt.subplot(2, 2, 2)
    model_names = list(results.keys())
    train_rmse = [results[name]['train_final_rmse'] for name in model_names]
    test_rmse = [results[name]['test_final_rmse'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, train_rmse, width, label='Train RMSE')
    plt.bar(x + width/2, test_rmse, width, label='Test RMSE')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.ylabel('RMSE')
    plt.title('Train vs. Test RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot predictions vs. actual
    plt.subplot(2, 2, 3)
    overfit_preds = results['Overfit (deep trees)']['model'].predict(X_test)
    balanced_preds = results['Balanced model']['model'].predict(X_test)
    underfit_preds = results['Underfit (shallow trees)']['model'].predict(X_test)
    
    plt.scatter(y_test, overfit_preds, alpha=0.5, label='Overfit')
    plt.scatter(y_test, balanced_preds, alpha=0.5, label='Balanced')
    plt.scatter(y_test, underfit_preds, alpha=0.5, label='Underfit')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predictions vs. Actual')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot regularization effect
    plt.subplot(2, 2, 4)
    
    # Train models with different regularization parameters
    reg_lambdas = [0, 0.01, 0.1, 1, 10, 100]
    train_scores = []
    test_scores = []
    
    for reg_lambda in reg_lambdas:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=10,  # Intentionally deep trees to show regularization effect
            learning_rate=0.1,
            reg_lambda=reg_lambda,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        train_scores.append(train_rmse)
        test_scores.append(test_rmse)
    
    plt.plot(reg_lambdas, train_scores, 'o-', label='Train RMSE')
    plt.plot(reg_lambdas, test_scores, 's-', label='Test RMSE')
    plt.xscale('log')
    plt.xlabel('L2 Regularization (lambda)')
    plt.ylabel('RMSE')
    plt.title('Effect of Regularization')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Demonstrate overfitting
overfitting_results = demonstrate_overfitting()

# Strategies to prevent overfitting
def prevent_overfitting():
    """Demonstrate various strategies to prevent overfitting."""
    # Base overfit model
    base_model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=15,
        learning_rate=0.1,
        random_state=42
    )
    
    # Strategy 1: Early stopping
    early_stopping_model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=15,
        learning_rate=0.1,
        random_state=42
    )
    
    # Strategy 2: Tree constraints
    tree_constraints_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,  # Shallower trees
        min_child_weight=5,  # Require more observations per node
        gamma=1,  # Minimum loss reduction for split
        random_state=42
    )
    
    # Strategy 3: Stochastic sampling
    stochastic_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=10,
        subsample=0.7,  # Sample 70% of training instances
        colsample_bytree=0.7,  # Sample 70% of features
        random_state=42
    )
    
    # Strategy 4: Regularization
    regularization_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=10,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        random_state=42
    )
    
    # Strategy 5: Combined approach
    combined_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )
    
    # Train and evaluate models
    models = {
        'Overfit (base)': base_model,
        'Early Stopping': early_stopping_model,
        'Tree Constraints': tree_constraints_model,
        'Stochastic Sampling': stochastic_model,
        'Regularization': regularization_model,
        'Combined Approach': combined_model
    }
    
    results = {}
    
    for name, model in models.items():
        # Create train/validation split for early stopping
        if name == 'Early Stopping':
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            eval_set = [(X_tr, y_tr), (X_val, y_val)]
            
            model.fit(
                X_tr, y_tr,
                eval_set=eval_set,
                eval_metric='rmse',
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Use best model for final evaluation
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
        else:
            # Train other models normally
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        # Store results
        results[name] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'gap': abs(train_rmse - test_rmse),
            'model': model
        }
        
        print(f"{name}:")
        print(f"  Training RMSE: {train_rmse:.2f}")
        print(f"  Test RMSE: {test_rmse:.2f}")
        print(f"  Gap: {abs(train_rmse - test_rmse):.2f}")
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot train vs test RMSE
    plt.subplot(2, 2, 1)
    model_names = list(results.keys())
    train_rmse = [results[name]['train_rmse'] for name in model_names]
    test_rmse = [results[name]['test_rmse'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, train_rmse, width, label='Train RMSE')
    plt.bar(x + width/2, test_rmse, width, label='Test RMSE')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.ylabel('RMSE')
    plt.title('Train vs. Test RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot RMSE gap (measure of overfitting)
    plt.subplot(2, 2, 2)
    gaps = [results[name]['gap'] for name in model_names]
    
    plt.bar(model_names, gaps)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('|Train RMSE - Test RMSE|')
    plt.title('Overfitting Gap')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Early stopping visualization
    if 'Early Stopping' in results:
        plt.subplot(2, 2, 3)
        
        # Get the results dictionary from the early stopping model
        evals_result = results['Early Stopping']['model'].evals_result()
        
        x_axis = range(len(evals_result['validation_0']['rmse']))
        plt.plot(x_axis, evals_result['validation_0']['rmse'], label='Train')
        plt.plot(x_axis, evals_result['validation_1']['rmse'], label='Validation')
        
        best_iteration = results['Early Stopping']['model'].best_iteration
        best_score = evals_result['validation_1']['rmse'][best_iteration]
        
        plt.axvline(x=best_iteration, color='r', linestyle='--', 
                   label=f'Best iteration: {best_iteration}')
        
        plt.xlabel('Boosting Iterations')
        plt.ylabel('RMSE')
        plt.title('Early Stopping Process')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # Feature importance from combined model
    plt.subplot(2, 2, 4)
    combined_model = results['Combined Approach']['model']
    xgb.plot_importance(combined_model, max_num_features=10, importance_type='gain', ax=plt.gca())
    plt.title('Feature Importance (Combined Approach)')
    
    plt.tight_layout()
    plt.show()
    
    return results

# Demonstrate overfitting prevention strategies
prevention_results = prevent_overfitting()
```

### Incorrect Parameter Tuning

```python
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import time

# Load dataset
boston = load_boston()
X, y = boston.data, boston.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to demonstrate common parameter tuning mistakes
def demonstrate_parameter_tuning_mistakes():
    """Show common mistakes in XGBoost parameter tuning and how to fix them."""
    
    # Mistake 1: Ignoring the relationship between n_estimators and learning_rate
    def mistake_1():
        """Ignoring n_estimators and learning_rate relationship."""
        print("\nMistake 1: Ignoring n_estimators and learning_rate relationship")
        
        # Test different combinations
        learning_rates = [0.3, 0.1, 0.01]
        n_estimators_list = [10, 100, 1000]
        
        results = []
        
        for lr in learning_rates:
            for n_est in n_estimators_list:
                model = xgb.XGBRegressor(
                    n_estimators=n_est,
                    learning_rate=lr,
                    max_depth=5,
                    random_state=42
                )
                
                # Train model
                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                # Evaluate
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                
                results.append({
                    'learning_rate': lr,
                    'n_estimators': n_est,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_time': train_time
                })
                
                print(f"  learning_rate={lr}, n_estimators={n_est}:")
                print(f"    Train RMSE: {train_rmse:.4f}")
                print(f"    Test RMSE: {test_rmse:.4f}")
                print(f"    Training Time: {train_time:.4f} seconds")
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        # Visualize
        plt.figure(figsize=(15, 5))
        
        # Plot test RMSE
        plt.subplot(1, 3, 1)
        for lr in learning_rates:
            subset = results_df[results_df['learning_rate'] == lr]
            plt.plot(subset['n_estimators'], subset['test_rmse'], 'o-', label=f'LR={lr}')
        
        plt.xlabel('Number of Trees')
        plt.ylabel('Test RMSE')
        plt.title('Test RMSE vs. Number of Trees')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot training time
        plt.subplot(1, 3, 2)
        for lr in learning_rates:
            subset = results_df[results_df['learning_rate'] == lr]
            plt.plot(subset['n_estimators'], subset['train_time'], 'o-', label=f'LR={lr}')
        
        plt.xlabel('Number of Trees')
        plt.ylabel('Training Time (seconds)')
        plt.title('Training Time vs. Number of Trees')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Calculate learning rate * n_estimators as a measure of "total learning"
        results_df['total_learning'] = results_df['learning_rate'] * results_df['n_estimators']
        
        # Plot test RMSE vs. total learning
        plt.subplot(1, 3, 3)
        plt.scatter(results_df['total_learning'], results_df['test_rmse'], c=results_df['learning_rate'],
                   cmap='viridis', s=50)
        
        plt.xlabel('Learning Rate * Number of Trees')
        plt.ylabel('Test RMSE')
        plt.title('Test RMSE vs. Total Learning')
        plt.colorbar(label='Learning Rate')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        print("\nCorrect approach:")
        print("  1. Lower learning rate requires more trees")
        print("  2. Start with a moderate learning rate (0.1) and enough trees (100)")
        print("  3. Use early stopping to find the optimal number of trees")
        print("  4. For final model, consider lower learning rate with more trees")
        
        return results_df
    
    # Mistake 2: Tuning max_depth without considering other complexity parameters
    def mistake_2():
        """Tuning max_depth in isolation."""
        print("\nMistake 2: Tuning max_depth without considering other complexity parameters")
        
        # Test different max_depth values
        max_depths = [3, 5, 10, 15, None]  # None means unlimited
        
        results = []
        
        for depth in max_depths:
            # Basic model with just max_depth variation
            basic_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=depth,
                learning_rate=0.1,
                random_state=42
            )
            
            # Better model with additional complexity parameters
            better_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=depth,
                learning_rate=0.1,
                min_child_weight=3,
                gamma=0.1,  # Minimum loss reduction for split
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            # Train basic model
            basic_model.fit(X_train, y_train)
            
            # Train better model
            better_model.fit(X_train, y_train)
            
            # Evaluate basic model
            basic_train_pred = basic_model.predict(X_train)
            basic_test_pred = basic_model.predict(X_test)
            
            basic_train_rmse = np.sqrt(mean_squared_error(y_train, basic_train_pred))
            basic_test_rmse = np.sqrt(mean_squared_error(y_test, basic_test_pred))
            
            # Evaluate better model
            better_train_pred = better_model.predict(X_train)
            better_test_pred = better_model.predict(X_test)
            
            better_train_rmse = np.sqrt(mean_squared_error(y_train, better_train_pred))
            better_test_rmse = np.sqrt(mean_squared_error(y_test, better_test_pred))
            
            # Store results
            results.append({
                'max_depth': 'Unlimited' if depth is None else depth,
                'approach': 'Basic',
                'train_rmse': basic_train_rmse,
                'test_rmse': basic_test_rmse,
                'gap': abs(basic_train_rmse - basic_test_rmse)
            })
            
            results.append({
                'max_depth': 'Unlimited' if depth is None else depth,
                'approach': 'Better',
                'train_rmse': better_train_rmse,
                'test_rmse': better_test_rmse,
                'gap': abs(better_train_rmse - better_test_rmse)
            })
            
            print(f"  max_depth={depth}:")
            print(f"    Basic - Train RMSE: {basic_train_rmse:.4f}, Test RMSE: {basic_test_rmse:.4f}")
            print(f"    Better - Train RMSE: {better_train_rmse:.4f}, Test RMSE: {better_test_rmse:.4f}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Visualize
        plt.figure(figsize=(15, 5))
        
        # Plot test RMSE
        plt.subplot(1, 3, 1)
        basic_results = results_df[results_df['approach'] == 'Basic']
        better_results = results_df[results_df['approach'] == 'Better']
        
        plt.plot(basic_results['max_depth'], basic_results['test_rmse'], 'o-', label='Basic')
        plt.plot(better_results['max_depth'], better_results['test_rmse'], 's-', label='Better')
        
        plt.xlabel('Max Depth')
        plt.ylabel('Test RMSE')
        plt.title('Test RMSE vs. Max Depth')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot overfitting gap
        plt.subplot(1, 3, 2)
        plt.plot(basic_results['max_depth'], basic_results['gap'], 'o-', label='Basic')
        plt.plot(better_results['max_depth'], better_results['gap'], 's-', label='Better')
        
        plt.xlabel('Max Depth')
        plt.ylabel('|Train RMSE - Test RMSE|')
        plt.title('Overfitting Gap vs. Max Depth')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot train vs test RMSE for unlimited depth
        plt.subplot(1, 3, 3)
        
        unlimited_basic = basic_results[basic_results['max_depth'] == 'Unlimited']
        unlimited_better = better_results[better_results['max_depth'] == 'Unlimited']
        
        approaches = ['Basic', 'Better']
        train_rmse = [unlimited_basic['train_rmse'].values[0], unlimited_better['train_rmse'].values[0]]
        test_rmse = [unlimited_basic['test_rmse'].values[0], unlimited_better['test_rmse'].values[0]]
        
        x = np.arange(len(approaches))
        width = 0.35
        
        plt.bar(x - width/2, train_rmse, width, label='Train RMSE')
        plt.bar(x + width/2, test_rmse, width, label='Test RMSE')
        
        plt.xlabel('Approach')
        plt.ylabel('RMSE')
        plt.title('Train vs. Test RMSE (Unlimited Depth)')
        plt.xticks(x, approaches)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        print("\nCorrect approach:")
        print("  1. Consider max_depth along with other complexity parameters")
        print("  2. Use min_child_weight to control leaf node size")
        print("  3. Use gamma to control split thresholds")
        print("  4. Use subsample and colsample_bytree for robustness")
        print("  5. Tune these parameters together, not in isolation")
        
        return results_df
    
    # Mistake 3: Using default evaluation metrics
    def mistake_3():
        """Using default evaluation metrics."""
        print("\nMistake 3: Using default evaluation metrics")
        
        # Create different evaluation metrics
        eval_metrics = ['rmse', 'mae']
        
        results = []
        
        for metric in eval_metrics:
            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            
            # Set parameters
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': metric,
                'max_depth': 5,
                'eta': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
            
            # Train with early stopping
            evals_result = {}
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dtrain, 'train'), (dtest, 'test')],
                early_stopping_rounds=20,
                verbose_eval=False,
                evals_result=evals_result
            )
            
            # Evaluate on test set
            test_pred = model.predict(dtest)
            rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            # Mean absolute error
            mae = np.mean(np.abs(y_test - test_pred))
            
            # Calculate mean squared error on a small subset with unusual values
            # (simulating a business case where some instances are more important)
            high_value_indices = np.argsort(y_test)[-10:]  # Top 10 highest value properties
            high_value_pred = test_pred[high_value_indices]
            high_value_true = y_test[high_value_indices]
            
            high_value_rmse = np.sqrt(mean_squared_error(high_value_true, high_value_pred))
            
            results.append({
                'eval_metric': metric,
                'best_iteration': model.best_iteration,
                'test_rmse': rmse,
                'test_mae': mae,
                'high_value_rmse': high_value_rmse,
                'train_history': evals_result['train'][metric],
                'test_history': evals_result['test'][metric]
            })
            
            print(f"  Evaluation metric: {metric}")
            print(f"    Best iteration: {model.best_iteration}")
            print(f"    Test RMSE: {rmse:.4f}")
            print(f"    Test MAE: {mae:.4f}")
            print(f"    High-value RMSE: {high_value_rmse:.4f}")
        
        # Visualize
        plt.figure(figsize=(15, 10))
        
        # Plot training curves for different metrics
        plt.subplot(2, 2, 1)
        for result in results:
            metric = result['eval_metric']
            iterations = range(1, len(result['train_history']) + 1)
            plt.plot(iterations, result['train_history'], '--', label=f'{metric} (train)')
            plt.plot(iterations, result['test_history'], '-', label=f'{metric} (test)')
            plt.axvline(x=result['best_iteration'], linestyle='--', 
                       color='red' if metric == 'rmse' else 'green',
                       label=f"{metric} best iter: {result['best_iteration']}")
        
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Metric Value')
        plt.title('Training Curves with Different Evaluation Metrics')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot final metrics
        plt.subplot(2, 2, 2)
        metrics = ['test_rmse', 'test_mae', 'high_value_rmse']
        x = np.arange(len(metrics))
        width = 0.35
        
        rmse_values = [results[0][metric] for metric in metrics]
        mae_values = [results[1][metric] for metric in metrics]
        
        plt.bar(x - width/2, rmse_values, width, label='RMSE as eval_metric')
        plt.bar(x + width/2, mae_values, width, label='MAE as eval_metric')
        
        plt.xlabel('Performance Metric')
        plt.ylabel('Value (lower is better)')
        plt.title('Performance Metrics by Evaluation Metric')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Custom evaluation function demonstration
        plt.subplot(2, 2, 3)
        
        # Define custom evaluation function that focuses on high-value instances
        def high_value_rmse(predt, dtrain):
            labels = dtrain.get_label()
            high_indices = np.argsort(labels)[-int(len(labels)*0.1):]  # Top 10%
            high_predt = predt[high_indices]
            high_labels = labels[high_indices]
            return 'high_value_rmse', np.sqrt(np.mean((high_predt - high_labels)**2))
        
        # Train model with custom metric
        params['eval_metric'] = 'rmse'  # Default metric
        
        evals_result = {}
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=20,
            verbose_eval=False,
            evals_result=evals_result,
            feval=high_value_rmse
        )
        
        # Plot default and custom metrics
        iterations = range(1, len(evals_result['test']['rmse']) + 1)
        plt.plot(iterations, evals_result['test']['rmse'], '-', label='RMSE')
        plt.plot(iterations, evals_result['test']['high_value_rmse'], '-', label='High-Value RMSE')
        
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Metric Value')
        plt.title('Default vs. Custom Evaluation Metric')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        print("\nCorrect approach:")
        print("  1. Choose evaluation metrics that align with business objectives")
        print("  2. Consider multiple metrics for a comprehensive evaluation")
        print("  3. Use custom evaluation functions for specific requirements")
        print("  4. Understand that different metrics may lead to different optimal parameters")
        
        return results
    
    # Run all demonstrations
    results = {
        'mistake_1': mistake_1(),
        'mistake_2': mistake_2(),
        'mistake_3': mistake_3()
    }
    
    # Summary
    print("\nSummary of Parameter Tuning Best Practices:")
    print("1. Understand parameter relationships (e.g., learning_rate and n_estimators)")
    print("2. Tune groups of related parameters together, not in isolation")
    print("3. Choose appropriate evaluation metrics for your specific problem")
    print("4. Use early stopping to find the optimal number of trees")
    print("5. Start with a coarse grid, then refine around promising values")
    print("6. Use cross-validation for more reliable parameter selection")
    print("7. Consider computational efficiency in your parameter choices")
    print("8. Balance model complexity with generalization ability")
    
    return results

# Demonstrate parameter tuning mistakes
tuning_mistakes = demonstrate_parameter_tuning_mistakes()
```

### Memory Issues with Large Datasets

```python
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import time
import os
import tempfile
import gc
import psutil

# Function to get memory usage
def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB

# Function to create large synthetic dataset
def create_large_dataset(n_samples=1000000, n_features=50, chunk_size=None):
    """Create a large synthetic dataset, optionally in chunks."""
    print(f"Creating dataset with {n_samples} samples and {n_features} features...")
    
    if chunk_size is None:
        # Create dataset all at once
        X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                            n_informative=10, random_state=42)
        return X, y
    else:
        # Create dataset in chunks
        chunks_X = []
        chunks_y = []
        remaining = n_samples
        
        while remaining > 0:
            current_chunk = min(chunk_size, remaining)
            X_chunk, y_chunk = make_regression(n_samples=current_chunk, n_features=n_features,
                                           n_informative=10, random_state=42)
            chunks_X.append(X_chunk)
            chunks_y.append(y_chunk)
            remaining -= current_chunk
            print(f"  Created chunk of {current_chunk} samples, {remaining} remaining...")
        
        return np.vstack(chunks_X), np.concatenate(chunks_y)

# Function to demonstrate memory issues and solutions
def demonstrate_memory_solutions(n_samples=500000, n_features=50):
    """Demonstrate XGBoost memory issues and solutions for large datasets."""
    
    print(f"Initial memory usage: {get_memory_usage():.2f} MB")
    
    # 1. Standard approach (baseline, may cause memory issues)
    print("\n1. Standard approach (baseline)")
    try:
        memory_before = get_memory_usage()
        start_time = time.time()
        
        # Create large dataset
        X, y = create_large_dataset(n_samples, n_features)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # Set parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'eta': 0.1
        }
        
        # Train model
        model = xgb.train(params, dtrain, num_boost_round=10)
        
        end_time = time.time()
        memory_after = get_memory_usage()
        
        print(f"  Training time: {end_time - start_time:.2f} seconds")
        print(f"  Memory usage: {memory_after:.2f} MB (increase: {memory_after - memory_before:.2f} MB)")
        
        # Clean up
        del X, y, dtrain, model
        gc.collect()
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  Try using a smaller dataset or one of the memory-efficient approaches below.")
    
    # Reset memory
    gc.collect()
    print(f"Memory after cleanup: {get_memory_usage():.2f} MB")
    
    # 2. Using external memory mode
    print("\n2. External memory mode")
    try:
        memory_before = get_memory_usage()
        start_time = time.time()
        
        # Create temporary libsvm file
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, 'temp_data.libsvm')
        
        # Create dataset in chunks and save to disk
        chunk_size = 100000
        X, y = create_large_dataset(n_samples, n_features, chunk_size)
        
        # Save to libsvm format
        print(f"  Saving data to {temp_file}")
        with open(temp_file, 'w') as f:
            for i in range(len(y)):
                f.write(f"{y[i]}")
                for j in range(X.shape[1]):
                    f.write(f" {j+1}:{X[i, j]}")
                f.write("\n")
                
                # Print progress
                if i % 100000 == 0 and i > 0:
                    print(f"  Wrote {i} samples...")
        
        # Clean up memory
        del X, y
        gc.collect()
        
        # Create DMatrix with external memory mode
        dtrain = xgb.DMatrix(f"{temp_file}?format=libsvm#cache_prefix={temp_dir}/cache")
        
        # Train model
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'eta': 0.1
        }
        
        model = xgb.train(params, dtrain, num_boost_round=10)
        
        end_time = time.time()
        memory_after = get_memory_usage()
        
        print(f"  Training time: {end_time - start_time:.2f} seconds")
        print(f"  Memory usage: {memory_after:.2f} MB (increase: {memory_after - memory_before:.2f} MB)")
        
        # Clean up
        os.remove(temp_file)
        os.rmdir(temp_dir)
        del dtrain, model
        gc.collect()
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Reset memory
    gc.collect()
    print(f"Memory after cleanup: {get_memory_usage():.2f} MB")
    
    # 3. Using sparse matrices
    print("\n3. Using sparse matrices")
    try:
        from scipy import sparse
        
                memory_before = get_memory_usage()
        start_time = time.time()
        
        # Create large dataset with sparsity
        X, y = create_large_dataset(n_samples, n_features)
        
        # Artificially make the dataset sparse by setting most values to zero
        # (In a real scenario, your data might already be sparse)
        sparsity = 0.8  # 80% of values will be zero
        mask = np.random.random(X.shape) < sparsity
        X[mask] = 0
        
        # Convert to sparse matrix
        X_sparse = sparse.csr_matrix(X)
        
        # Check sparsity and memory reduction
        dense_size = X.nbytes / (1024 * 1024)
        sparse_size = X_sparse.data.nbytes / (1024 * 1024) + X_sparse.indptr.nbytes / (1024 * 1024) + X_sparse.indices.nbytes / (1024 * 1024)
        
        print(f"  Dense matrix size: {dense_size:.2f} MB")
        print(f"  Sparse matrix size: {sparse_size:.2f} MB")
        print(f"  Memory reduction: {dense_size / sparse_size:.2f}x")
        
        # Create DMatrix from sparse matrix
        dtrain = xgb.DMatrix(X_sparse, label=y)
        
        # Train model
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'eta': 0.1
        }
        
        model = xgb.train(params, dtrain, num_boost_round=10)
        
        end_time = time.time()
        memory_after = get_memory_usage()
        
        print(f"  Training time: {end_time - start_time:.2f} seconds")
        print(f"  Memory usage: {memory_after:.2f} MB (increase: {memory_after - memory_before:.2f} MB)")
        
        # Clean up
        del X, y, X_sparse, dtrain, model
        gc.collect()
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Reset memory
    gc.collect()
    print(f"Memory after cleanup: {get_memory_usage():.2f} MB")
    
    # 4. Using histogram-based algorithm
    print("\n4. Using histogram-based algorithm")
    try:
        memory_before = get_memory_usage()
        start_time = time.time()
        
        # Create large dataset
        X, y = create_large_dataset(n_samples, n_features)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # Set parameters with histogram-based algorithm
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'eta': 0.1,
            'tree_method': 'hist',  # Use histogram-based algorithm
            'max_bin': 256  # Reduce number of bins for further memory savings
        }
        
        # Train model
        model = xgb.train(params, dtrain, num_boost_round=10)
        
        end_time = time.time()
        memory_after = get_memory_usage()
        
        print(f"  Training time: {end_time - start_time:.2f} seconds")
        print(f"  Memory usage: {memory_after:.2f} MB (increase: {memory_after - memory_before:.2f} MB)")
        
        # Clean up
        del X, y, dtrain, model
        gc.collect()
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Reset memory
    gc.collect()
    print(f"Memory after cleanup: {get_memory_usage():.2f} MB")
    
    # 5. Comparison of all approaches
    print("\n5. Memory usage comparison summary")
    print("  1. Standard approach: Highest memory usage, fastest for smaller datasets")
    print("  2. External memory: Lowest memory usage, slowest due to disk I/O")
    print("  3. Sparse matrices: Efficient for sparse data, faster than external memory")
    print("  4. Histogram-based: Good balance of speed and memory for large datasets")
    print("\nRecommendations:")
    print("  - For datasets that fit in memory: Use standard approach or histogram-based algorithm")
    print("  - For sparse datasets: Use sparse matrices")
    print("  - For datasets too large for memory: Use external memory mode")
    print("  - Additional options:")
    print("    - Reduce precision (use float32 instead of float64)")
    print("    - Train on data subsets and use ensemble methods")
    print("    - Use distributed XGBoost with multiple machines")

# Run demonstration
demonstrate_memory_solutions()
```

---

## ❓ FAQ

### Q1: When should I use XGBoost over other algorithms?

**A:** XGBoost is particularly well-suited for the following scenarios:

1. **Tabular/Structured Data**: XGBoost consistently performs extremely well on structured data with numerical and categorical features. It's typically among the top performers for:
   - Classification problems (binary and multi-class)
   - Regression problems
   - Ranking tasks

2. **When Performance Matters**: When you need state-of-the-art predictive performance and you're working with structured data, XGBoost is often the algorithm of choice because:
   - It handles complex non-linear relationships effectively
   - It's robust against overfitting when properly tuned
   - It implicitly performs feature selection

3. **Moderate-sized Datasets**: XGBoost works particularly well for datasets with:
   - Thousands to millions of rows
   - Dozens to hundreds of features

4. **When Interpretability is Important**: While not as interpretable as a single decision tree, XGBoost provides:
   - Feature importance scores
   - Compatibility with SHAP values for detailed explanations
   - Tree visualization options

5. **Handling Missing Values**: XGBoost has a built-in capability to handle missing values effectively.

6. **Imbalanced Data**: With appropriate parameterization (`scale_pos_weight`), XGBoost can work well with imbalanced classes.

7. **Production Environments**: XGBoost's speed and memory efficiency make it suitable for production deployments, especially with:
   - Efficient inference
   - Support for multiple programming languages
   - Model serialization options

XGBoost might NOT be the best choice when:
- You're working with image, audio, or text data (deep learning typically performs better)
- You need a highly interpretable model (consider decision trees or linear models)
- You have an extremely large dataset that doesn't fit in memory (though XGBoost has some ways to handle this)
- Computational resources are extremely limited
- The relationship in your data is known to be simple and linear

### Q2: How do I handle categorical features in XGBoost?

**A:** XGBoost doesn't handle categorical features natively, but there are several effective approaches:

1. **One-Hot Encoding**: Convert categorical variables to binary features.
   ```python
   import pandas as pd
   from sklearn.preprocessing import OneHotEncoder
   
   # Using pandas
   df = pd.get_dummies(df, columns=['category_column'])
   
   # Using scikit-learn for more control
   encoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' avoids multicollinearity
   encoded_features = encoder.fit_transform(df[['category_column']])
   ```

2. **Label Encoding**: Convert categories to numeric values (preserves ordinal relationships).
   ```python
   from sklearn.preprocessing import LabelEncoder
   
   encoder = LabelEncoder()
   df['category_encoded'] = encoder.fit_transform(df['category_column'])
   ```

3. **Target Encoding**: Replace categories with their target mean (careful about leakage).
   ```python
   # Calculate mean target value for each category
   target_means = df.groupby('category_column')['target'].mean()
   
   # Map these values to the original dataframe
   df['category_target_encoded'] = df['category_column'].map(target_means)
   ```

4. **Count Encoding**: Replace categories with their frequency.
   ```python
   count_map = df['category_column'].value_counts().to_dict()
   df['category_count_encoded'] = df['category_column'].map(count_map)
   ```

5. **Catboost Encoding**: An advanced encoding method that avoids target leakage.
   ```python
   from category_encoders import CatBoostEncoder
   
   encoder = CatBoostEncoder()
   df_encoded = encoder.fit_transform(df[['category_column']], df['target'])
   ```

6. **Using Feature Importance to Select Encodings**:
   ```python
   # Try different encodings and evaluate feature importance
   # Example combining one-hot and target encoding
   
   # Create different encoded versions
   df_onehot = pd.get_dummies(df, columns=['category_column'])
   
   # For target encoding
   target_means = df.groupby('category_column')['target'].mean()
   df_target = df.copy()
   df_target['category_target_encoded'] = df['category_column'].map(target_means)
   
   # Train models and compare feature importance
   model1 = xgb.XGBClassifier()
   model1.fit(df_onehot.drop('target', axis=1), df_onehot['target'])
   
   model2 = xgb.XGBClassifier()
   model2.fit(df_target.drop(['target', 'category_column'], axis=1), 
             df_target['target'])
   
   # Compare feature importance to choose the best encoding
   ```

7. **Using Feature Interactions**: XGBoost's tree structure automatically finds interactions.
   ```python
   # No special code needed - XGBoost's tree structure naturally
   # discovers interactions between features, including encoded categorical ones
   ```

8. **High-Cardinality Categories**: For categories with many values.
   ```python
   # For high-cardinality features, group rare categories
   value_counts = df['high_cardinality_col'].value_counts()
   threshold = 10  # Minimum frequency
   
   # Create a mapping
   mapping = {val: val if count >= threshold else 'Other' 
             for val, count in value_counts.items()}
   
   # Apply mapping
   df['high_card_grouped'] = df['high_cardinality_col'].map(mapping)
   
   # Then apply your preferred encoding method
   ```

When choosing an encoding method, consider:
- The number of categories (one-hot is inefficient for high-cardinality)
- Whether categories have an ordinal relationship
- The risk of target leakage with target encoding
- Computational and memory constraints

### Q3: How do I prevent overfitting in XGBoost?

**A:** XGBoost provides multiple ways to prevent overfitting:

1. **Early Stopping**: Stop training when performance on validation set stops improving.
   ```python
   model = xgb.XGBClassifier(
       n_estimators=1000,  # Set a high number
       learning_rate=0.1
   )
   
   model.fit(
       X_train, y_train,
       eval_set=[(X_train, y_train), (X_val, y_val)],
       early_stopping_rounds=10,  # Stop if no improvement for 10 rounds
       verbose=False
   )
   
   # The model automatically uses the optimal number of trees
   # You can access it with: model.best_iteration
   ```

2. **Tree Constraints**: Limit the complexity of individual trees.
   ```python
   model = xgb.XGBClassifier(
       max_depth=3,  # Shallow trees (default is 6)
       min_child_weight=5,  # Require more observations per leaf
       gamma=1  # Minimum loss reduction for further partition
   )
   ```

3. **Stochastic Sampling**: Add randomness to make trees more independent.
   ```python
   model = xgb.XGBClassifier(
       subsample=0.8,  # Use 80% of instances for each tree
       colsample_bytree=0.8,  # Use 80% of features for each tree
       colsample_bylevel=0.8  # Use 80% of features for each level
   )
   ```

4. **Regularization**: Add penalties for model complexity.
   ```python
   model = xgb.XGBClassifier(
       reg_alpha=0.1,  # L1 regularization
       reg_lambda=1.0  # L2 regularization
   )
   ```

5. **Learning Rate**: Use a smaller learning rate with more trees.
   ```python
   model = xgb.XGBClassifier(
       learning_rate=0.01,  # Small learning rate
       n_estimators=500  # More trees to compensate
   )
   ```

6. **Pruning**: Remove splits that don't improve performance.
   ```python
   # Pruning is controlled by gamma parameter
   model = xgb.XGBClassifier(gamma=1.0)
   ```

7. **Cross-Validation**: Use cross-validation for more reliable hyperparameter tuning.
   ```python
   from sklearn.model_selection import GridSearchCV
   
   param_grid = {
       'max_depth': [3, 5, 7],
       'min_child_weight': [1, 3, 5],
       'gamma': [0, 0.1, 0.2]
   }
   
   grid_search = GridSearchCV(
       xgb.XGBClassifier(),
       param_grid,
       cv=5,
       scoring='accuracy'
   )
   
   grid_search.fit(X_train, y_train)
   ```

8. **Combined Approach**: Use multiple techniques together.
   ```python
   model = xgb.XGBClassifier(
       n_estimators=200,
       learning_rate=0.05,
       max_depth=4,
       min_child_weight=3,
       gamma=0.1,
       subsample=0.8,
       colsample_bytree=0.8,
       reg_alpha=0.1,
       reg_lambda=1.0
   )
   
   model.fit(
       X_train, y_train,
       eval_set=[(X_val, y_val)],
       early_stopping_rounds=20,
       verbose=False
   )
   ```

Approach to tuning for overfitting:
1. Start with a reasonable learning rate (0.1) and enough trees (100-200)
2. Use early stopping with a validation set
3. Tune tree constraints (max_depth, min_child_weight, gamma)
4. Tune stochastic parameters (subsample, colsample_bytree)
5. Tune regularization parameters (reg_alpha, reg_lambda)
6. Fine-tune learning rate and number of trees
7. Monitor the gap between training and validation performance

### Q4: What are the key differences between XGBoost, LightGBM, and CatBoost?

**A:** XGBoost, LightGBM, and CatBoost are all high-performance gradient boosting frameworks, but they differ in several important ways:

1. **Tree Growth Strategy**:
   - **XGBoost**: Grows trees level-wise (all nodes at one level before moving to the next)
   - **LightGBM**: Grows trees leaf-wise (choosing the leaf with max delta loss to split)
   - **CatBoost**: Uses a combination of balanced trees and oblivious trees (same feature used at each level)

2. **Handling Categorical Features**:
   - **XGBoost**: No built-in handling; requires pre-encoding
   - **LightGBM**: Native support with categorical splits
   - **CatBoost**: Advanced ordered target statistics for categorical features

3. **Performance and Speed**:
   - **XGBoost**: Generally fast and memory-efficient
   - **LightGBM**: Faster than XGBoost, especially for large datasets
   - **CatBoost**: Usually slower than LightGBM but can be faster for categorical data

4. **Memory Usage**:
   - **XGBoost**: Moderate memory usage
   - **LightGBM**: More memory efficient than XGBoost
   - **CatBoost**: Can use more memory due to one-hot encoding

5. **Default Parameters**:
   - **XGBoost**: Often requires more tuning
   - **LightGBM**: Good performance with defaults but benefits from tuning
   - **CatBoost**: Excellent performance with minimal tuning

6. **Handling Overfitting**:
   - **XGBoost**: Multiple regularization parameters
   - **LightGBM**: Similar regularization options to XGBoost
   - **CatBoost**: Built-in mechanisms like ordered boosting to prevent overfitting

7. **Special Features**:
   - **XGBoost**: Robust, mature library with many extensions
   - **LightGBM**: DART booster, better distributed training
   - **CatBoost**: Best-in-class handling of categorical features, ordered boosting

8. **Typical Use Cases**:
   - **XGBoost**: General-purpose, great baseline for structured data
   - **LightGBM**: Large datasets, speed-critical applications
   - **CatBoost**: Datasets with many categorical features, need for minimal tuning

9. **Implementation and Ecosystem**:
   - **XGBoost**: Multiple language bindings, highly optimized C++
   - **LightGBM**: C++ with Python, R bindings, distributed computing support
   - **CatBoost**: Strong integration with Python, well-documented

10. **Code Comparison**:
    ```python
    # XGBoost
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )
    xgb_model.fit(X_train, y_train)
    
    # LightGBM
    import lightgbm as lgb
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )
    lgb_model.fit(X_train, y_train)
    
    # CatBoost
    import catboost as cb
    cat_features = [0, 1, 2]  # Indices of categorical features
    cb_model = cb.CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=5
    )
    cb_model.fit(X_train, y_train, cat_features=cat_features)
    ```

When to choose each:
- **Choose XGBoost** when you want a robust, well-tested solution and are willing to handle categorical encoding.
- **Choose LightGBM** when speed is critical or for very large datasets.
- **Choose CatBoost** when you have many categorical features or want excellent performance with minimal tuning.

All three algorithms perform exceptionally well in practice, so it's often worth trying all of them for your specific problem.

### Q5: How can I interpret XGBoost models?

**A:** XGBoost models can be interpreted through several techniques:

1. **Feature Importance**: Understand which features are most influential.
   ```python
   import matplotlib.pyplot as plt
   
   # Get importance scores
   importance_type = 'gain'  # Options: 'gain', 'weight', 'cover', 'total_gain', 'total_cover'
   importance = model.feature_importances_  # From sklearn API
   
   # Alternative using booster
   booster = model.get_booster()
   importance = booster.get_score(importance_type=importance_type)
   
   # Sort and plot
   sorted_idx = np.argsort(importance)
   plt.barh(range(len(sorted_idx)), importance[sorted_idx])
   plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
   plt.title('Feature Importance')
   plt.show()
   ```

2. **SHAP Values**: Detailed feature contribution analysis using game theory.
   ```python
   import shap
   
   # Create explainer
   explainer = shap.TreeExplainer(model)
   
   # Calculate SHAP values
   shap_values = explainer.shap_values(X)
   
   # Summary plot
   shap.summary_plot(shap_values, X, feature_names=feature_names)
   
   # Detailed plot for a single prediction
   shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
   
   # Dependence plots for specific features
   shap.dependence_plot("feature_name", shap_values, X)
   ```

3. **Partial Dependence Plots**: Show how features affect predictions, on average.
   ```python
   from sklearn.inspection import partial_dependence, plot_partial_dependence
   
   # Calculate and plot partial dependence
   features = [0, 1]  # Feature indices to plot
   plot_partial_dependence(model, X, features, feature_names=feature_names)
   ```

4. **Tree Visualization**: Examine individual trees in the ensemble.
   ```python
   import xgboost as xgb
   
   # Plot a specific tree
   xgb.plot_tree(model, num_trees=0)  # Plot the first tree
   
   # To plot multiple trees
   for i in range(3):  # Plot first 3 trees
       plt.figure(figsize=(15, 15))
       xgb.plot_tree(model, num_trees=i)
       plt.title(f'Tree {i}')
       plt.show()
   ```

5. **Individual Prediction Explanation**:
   ```python
   # Using SHAP for individual predictions
   instance = X.iloc[0,:]  # First instance
   
   # Get SHAP values
   instance_shap = explainer.shap_values(instance)
   
   # Create waterfall plot
   shap.waterfall_plot(shap.Explanation(values=instance_shap, 
                                    base_values=explainer.expected_value, 
                                    data=instance, 
                                    feature_names=feature_names))
   ```

6. **Feature Interactions**: Understand how features work together.
   ```python
   # SHAP interaction values
   interaction_values = explainer.shap_interaction_values(X)
   
   # Visualize interactions for a specific prediction
   shap.summary_plot(interaction_values[0], X)
   
   # Quantify overall interaction strength
   interaction_strength = np.sum(np.abs(interaction_values), axis=0)
   for i in range(len(feature_names)):
       for j in range(i+1, len(feature_names)):
           print(f"Interaction {feature_names[i]} x {feature_names[j]}: {interaction_strength[i, j]}")
   ```

7. **Decision Path Analysis**: Trace how specific predictions are made.
   ```python
   # Get leaf indices for a specific data point
   booster = model.get_booster()
   leaf_index = booster.predict(xgb.DMatrix(X.iloc[0:1]), pred_leaf=True)
   
   # Use the leaf index to understand the path through the trees
   print(f"Leaf indices for each tree: {leaf_index}")
   ```

8. **Global Surrogate Models**: Train an interpretable model to mimic XGBoost.
   ```python
   from sklearn.tree import DecisionTreeClassifier, plot_tree
   
   # Get predictions from complex model
   y_pred = model.predict(X)
   
   # Train an interpretable surrogate model
   surrogate = DecisionTreeClassifier(max_depth=3)
   surrogate.fit(X, y_pred)
   
   # Visualize the surrogate model
   plt.figure(figsize=(15, 10))
   plot_tree(surrogate, feature_names=feature_names, filled=True)
   plt.show()
   ```

9. **Permutation Importance**: Measure importance by shuffling features.
   ```python
   from sklearn.inspection import permutation_importance
   
   result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
   
   # Sort features by importance
   sorted_idx = result.importances_mean.argsort()
   
   plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx])
   plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
   plt.title('Permutation Importance')
   plt.show()
   ```

Different interpretation methods provide different insights:
- Feature importance gives a global overview
- SHAP values provide detailed local and global explanations
- Partial dependence shows the average effect of features
- Tree visualization helps understand the model structure
- Interaction analysis reveals feature relationships

For the most comprehensive understanding, combine multiple interpretation techniques.

---

<div align="center">

## 🌟 Key Takeaways

**XGBoost:**
- Powerful implementation of gradient boosted trees for structured/tabular data
- Combines strong predictive performance with computational efficiency
- Offers comprehensive regularization options to prevent overfitting
- Provides flexible APIs for various programming languages and frameworks
- Includes built-in support for missing values and sparse data
- Consistently ranks among the top algorithms for machine learning competitions
- Can be applied to classification, regression, and ranking problems

**Remember:**
- Start with default parameters and systematically tune for your specific problem
- Use early stopping to determine the optimal number of trees
- Balance model complexity with regularization to prevent overfitting
- Leverage feature importance and SHAP values for model interpretation
- Consider memory optimization techniques for large datasets
- Choose appropriate evaluation metrics aligned with your business objectives
- Compare with other algorithms to ensure you're using the best tool for the job

---

### 📖 Happy Boosting! 🚀

*Created by [@danialasim](https://github.com/danialasim) | Last updated: August 10, 2025*

</div>