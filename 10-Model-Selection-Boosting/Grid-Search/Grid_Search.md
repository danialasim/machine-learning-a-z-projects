# 🔍 Grid Search

<div align="center">

![Method](https://img.shields.io/badge/Method-Hyperparameter_Optimization-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-green?style=for-the-badge)
![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-yellow?style=for-the-badge)

*A Comprehensive Guide to Grid Search for Systematic Hyperparameter Tuning*

</div>

---

## 📚 Table of Contents

- [Introduction to Grid Search](#introduction-to-grid-search)
- [Mathematical Foundation](#mathematical-foundation)
- [Implementation Guide](#implementation-guide)
- [Best Practices](#best-practices)
- [Advanced Grid Search Techniques](#advanced-grid-search-techniques)
- [Practical Applications](#practical-applications)
- [Comparison with Other Methods](#comparison-with-other-methods)
- [Common Pitfalls](#common-pitfalls)
- [FAQ](#faq)

---

## 🎯 Introduction to Grid Search

Grid Search is a systematic approach to hyperparameter tuning that works by creating a "grid" of all possible combinations of hyperparameter values and evaluating each combination to find the best performing model. It's an exhaustive search algorithm that helps identify optimal hyperparameter settings for machine learning models.

### Key Concepts:

- **Hyperparameters**: Configuration variables that govern the training process and model architecture
- **Exhaustive Search**: Evaluating every possible combination of hyperparameter values
- **Cross-Validation**: Using techniques like k-fold cross-validation to robustly evaluate each combination
- **Scoring Metrics**: Defining appropriate metrics to compare different hyperparameter combinations
- **Parameter Space**: The set of all possible hyperparameter configurations to explore
- **Computational Complexity**: Understanding the time and resource requirements of the search process

### Why Grid Search is Important:

1. **Systematic Approach**: Provides a methodical way to find optimal hyperparameters
2. **Reproducibility**: Ensures results can be reproduced given the same search space
3. **Model Performance**: Significantly improves model accuracy and effectiveness
4. **Automation**: Reduces the need for manual hyperparameter tuning
5. **Comprehensive Exploration**: Guarantees coverage of the entire hyperparameter space
6. **Baseline Establishment**: Provides a reliable baseline for comparison with other methods

### Brief History:

- **1990s**: Early forms of automated parameter tuning in machine learning
- **2000s**: Grid search becomes a standard technique in the ML community
- **2010**: Scikit-learn implements GridSearchCV, making the technique widely accessible
- **2010s**: With the rise of deep learning, more efficient alternatives started emerging
- **Present**: Still widely used as a reliable baseline approach for hyperparameter optimization

---

## 🧮 Mathematical Foundation

### Formulation

Let's define a machine learning model $M$ with hyperparameters $\lambda_1, \lambda_2, \ldots, \lambda_n$. Each hyperparameter $\lambda_i$ can take values from a set $\Lambda_i$. Grid search aims to find the optimal combination of hyperparameters $\lambda^* = (\lambda_1^*, \lambda_2^*, \ldots, \lambda_n^*)$ that maximizes (or minimizes) a performance metric $P$:

$$\lambda^* = \underset{\lambda \in \Lambda_1 \times \Lambda_2 \times \ldots \times \Lambda_n}{\arg\max} \: P(M_\lambda, D_{valid})$$

Where:
- $M_\lambda$ is the model trained with hyperparameters $\lambda$
- $D_{valid}$ is the validation dataset
- $\Lambda_1 \times \Lambda_2 \times \ldots \times \Lambda_n$ is the Cartesian product of all hyperparameter sets, representing the entire search space

### Cross-Validation in Grid Search

When using k-fold cross-validation, the performance metric is calculated as:

$$P(M_\lambda, D) = \frac{1}{k} \sum_{i=1}^{k} P(M_\lambda^{(i)}, D_{valid}^{(i)})$$

Where:
- $M_\lambda^{(i)}$ is the model trained on the $i$-th training fold with hyperparameters $\lambda$
- $D_{valid}^{(i)}$ is the $i$-th validation fold
- $k$ is the number of folds

### Computational Complexity

The computational complexity of grid search is:

$$O(||\Lambda_1|| \times ||\Lambda_2|| \times \ldots \times ||\Lambda_n|| \times C_{train} \times k)$$

Where:
- $||\Lambda_i||$ is the number of values to explore for hyperparameter $\lambda_i$
- $C_{train}$ is the cost of training the model once
- $k$ is the number of cross-validation folds

This exponential complexity with respect to the number of hyperparameters is known as the "curse of dimensionality" and represents the main limitation of grid search.

### Statistical Guarantees

Grid search provides the following statistical guarantees:

1. **Optimality within the grid**: The solution is guaranteed to be the best among all evaluated combinations.
2. **Convergence**: As the grid becomes infinitely fine, grid search converges to the global optimum (assuming continuity).
3. **Generalization**: When combined with proper cross-validation, it provides estimates of how the model will generalize to new data.

---

## 💻 Implementation Guide

### Implementation with Python

#### Basic Grid Search with scikit-learn

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# Load sample dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define the model
model = RandomForestClassifier(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,                  # 5-fold cross-validation
    scoring='accuracy',    # Metric to optimize
    n_jobs=-1,             # Use all available cores
    verbose=1              # Print progress
)

# Fit the grid search
grid_search.fit(X, y)

# Print best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Get the results as a DataFrame
results = pd.DataFrame(grid_search.cv_results_)
results = results.sort_values(by='rank_test_score')

# Display top 5 combinations
print("\nTop 5 Hyperparameter Combinations:")
columns_to_show = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
print(results[columns_to_show].head(5))

# Train the best model on the full dataset
best_model = grid_search.best_estimator_
best_model.fit(X, y)
```

#### Visualizing Grid Search Results

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Extract results
results = pd.DataFrame(grid_search.cv_results_)

# Create a pivot table for n_estimators vs max_depth
pivot_table = results.pivot_table(
    index='param_n_estimators', 
    columns='param_max_depth', 
    values='mean_test_score'
)

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.3f')
plt.title('Grid Search Results: n_estimators vs max_depth')
plt.ylabel('n_estimators')
plt.xlabel('max_depth')
plt.show()

# Plot the effect of each parameter individually
param_names = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']

plt.figure(figsize=(15, 10))
for i, param in enumerate(param_names):
    plt.subplot(2, 2, i+1)
    
    # Group by parameter and calculate mean score
    param_col = f'param_{param}'
    grouped = results.groupby(param_col)['mean_test_score'].mean()
    
    # For 'None' values in max_depth
    if param == 'max_depth':
        # Convert index to string to handle 'None' values
        index = grouped.index.map(lambda x: str(x))
    else:
        index = grouped.index
    
    plt.plot(index, grouped.values, 'o-')
    plt.title(f'Effect of {param} on Accuracy')
    plt.xlabel(param)
    plt.ylabel('Mean Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    
plt.tight_layout()
plt.show()
```

#### Grid Search with Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Create a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(random_state=42))
])

# Define parameter grid
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'classifier__kernel': ['rbf', 'linear', 'poly']
}

# Create grid search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

# Fit grid search
grid_search.fit(X, y)

# Print best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Compare train and test scores to check for overfitting
results = pd.DataFrame(grid_search.cv_results_)
results = results.sort_values(by='rank_test_score')

# Calculate differences between training and testing scores
results['train_test_diff'] = results['mean_train_score'] - results['mean_test_score']

# Display top 5 combinations with train-test differences
columns_to_show = ['params', 'mean_train_score', 'mean_test_score', 'train_test_diff', 'rank_test_score']
print("\nTop 5 Hyperparameter Combinations:")
print(results[columns_to_show].head(5))
```

### Implementation in R

```r
library(caret)
library(randomForest)
library(tidyverse)
library(ggplot2)

# Load data
data(iris)

# Set up training control
train_control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE
)

# Define parameter grid
param_grid <- expand.grid(
  mtry = c(1, 2, 3, 4),
  splitrule = c("gini", "extratrees"),
  min.node.size = c(1, 3, 5)
)

# Perform grid search
set.seed(42)
grid_search <- train(
  Species ~ .,
  data = iris,
  method = "ranger",
  tuneGrid = param_grid,
  trControl = train_control,
  importance = 'impurity'
)

# Print results
print(grid_search)
print("Best tuning parameters:")
print(grid_search$bestTune)

# Plot results
ggplot(grid_search) + theme_bw()

# Create custom visualization
results <- grid_search$results
ggplot(results, aes(x = factor(mtry), y = factor(min.node.size), fill = Accuracy)) +
  geom_tile() +
  facet_wrap(~ splitrule) +
  scale_fill_viridis_c() +
  labs(
    title = "Grid Search Results: Random Forest Parameters",
    x = "mtry",
    y = "min.node.size"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    strip.background = element_rect(fill = "lightblue"),
    legend.position = "right"
  )

# Train final model with best parameters
final_model <- randomForest(
  Species ~ .,
  data = iris,
  mtry = grid_search$bestTune$mtry,
  importance = TRUE
)

# Print model summary
print(final_model)
```

---

## 🛠️ Best Practices

### Defining the Parameter Grid

1. **Start with a coarse grid**: Begin with a wide range of values, then refine.

```python
# Coarse initial grid
param_grid_initial = {
    'n_estimators': [10, 100, 1000],
    'max_depth': [None, 10, 30],
    'min_samples_split': [2, 10]
}

# Refined grid based on initial results
param_grid_refined = {
    'n_estimators': [70, 100, 130],
    'max_depth': [20, 25, 30, 35],
    'min_samples_split': [2, 3, 4]
}
```

2. **Use logarithmic scales**: For parameters that span multiple orders of magnitude.

```python
# Logarithmic scale for C and gamma in SVM
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10]
}
```

3. **Leverage domain knowledge**: Start with sensible defaults based on literature or prior experience.

4. **Consider parameter dependencies**: Some parameters only make sense in combination with others.

```python
# Grid with conditional parameters
param_grid = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
    {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}
]
```

5. **Balance breadth and depth**: Cover a wide range of possibilities without making the grid too large.

### Efficient Computation

1. **Use parallelization**: Leverage multiple cores to speed up computation.

```python
# Use all available cores
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
```

2. **Implement early stopping**: Stop evaluating a configuration if early results are poor.

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

halving_search = HalvingGridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    factor=3,  # Reduce candidates by a factor of 3 at each iteration
    resource='n_samples',  # Resource to be increased over iterations
    max_resources='auto'   # Maximum amount of resource
)
```

3. **Use smaller subsets for initial screening**: Start with a subset of data to quickly eliminate poor performers.

```python
# Initial screening on a subset
X_subset, _, y_subset, _ = train_test_split(X, y, test_size=0.8, random_state=42)
initial_grid_search = GridSearchCV(model, param_grid, cv=3)
initial_grid_search.fit(X_subset, y_subset)

# Refined search with the best parameters on full dataset
best_params = initial_grid_search.best_params_
refined_param_grid = {k: [v] for k, v in best_params.items()}
final_grid_search = GridSearchCV(model, refined_param_grid, cv=5)
final_grid_search.fit(X, y)
```

4. **Use appropriate cross-validation**: Adjust the number of folds based on dataset size.

```python
# For large datasets, fewer folds
if len(X) > 10000:
    cv = 3
else:
    cv = 5

grid_search = GridSearchCV(model, param_grid, cv=cv)
```

### Avoiding Overfitting

1. **Use nested cross-validation**: One outer loop for performance estimation, one inner loop for hyperparameter tuning.

```python
from sklearn.model_selection import cross_val_score, KFold

# Outer cross-validation loop
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_scores = []

for train_idx, test_idx in outer_cv.split(X):
    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Inner cross-validation for hyperparameter tuning
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=inner_cv)
    grid_search.fit(X_train, y_train)
    
    # Evaluate best model on test set
    best_model = grid_search.best_estimator_
    score = best_model.score(X_test, y_test)
    outer_scores.append(score)

print(f"Nested CV Score: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")
```

2. **Monitor train vs. validation performance**: Check for signs of overfitting.

```python
# Request training scores
grid_search = GridSearchCV(model, param_grid, cv=5, return_train_score=True)
grid_search.fit(X, y)

# Analyze overfitting
results = pd.DataFrame(grid_search.cv_results_)
results['overfit_degree'] = results['mean_train_score'] - results['mean_test_score']

# Plot train vs. test scores
plt.figure(figsize=(12, 6))
plt.errorbar(range(len(results)), results['mean_train_score'], yerr=results['std_train_score'], 
             label='Training Score', fmt='o-')
plt.errorbar(range(len(results)), results['mean_test_score'], yerr=results['std_test_score'], 
             label='Validation Score', fmt='o-')
plt.legend()
plt.xlabel('Parameter Combination Index')
plt.ylabel('Score')
plt.title('Training vs. Validation Scores')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```

3. **Consider regularization parameters**: Include regularization in your grid search.

```python
# Grid including regularization parameters
param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],  # Regularization strength
    'l1_ratio': [0, 0.25, 0.5, 0.75, 1]  # Mix of L1 and L2 regularization (ElasticNet)
}
```

### Choosing Appropriate Metrics

1. **Match metrics to the problem**: Use domain-appropriate metrics.

```python
# Classification metrics
classification_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Regression metrics
regression_metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']

# Multi-metric evaluation
grid_search = GridSearchCV(
    model,
    param_grid,
    scoring=['accuracy', 'f1', 'roc_auc'],
    refit='f1'  # Use f1 to select the best model
)
```

2. **Use scoring functions for custom metrics**:

```python
from sklearn.metrics import make_scorer, fbeta_score

# Custom F2 score (weighs recall higher than precision)
f2_scorer = make_scorer(fbeta_score, beta=2)

grid_search = GridSearchCV(model, param_grid, scoring=f2_scorer)
```

3. **Consider business impact in metrics**: Align with business objectives.

```python
# Custom scoring function that incorporates business costs
def business_score(y_true, y_pred):
    # Calculate false positives and false negatives
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    # Assign costs: false negative costs $100, false positive costs $10
    total_cost = fn * 100 + fp * 10
    
    # Return negative cost (higher is better for GridSearchCV)
    return -total_cost

# Create scorer
business_scorer = make_scorer(business_score, greater_is_better=True)

# Use in grid search
grid_search = GridSearchCV(model, param_grid, scoring=business_scorer)
```

---

## 🔄 Advanced Grid Search Techniques

### Randomized Grid Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_distributions = {
    'n_estimators': randint(10, 200),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}

# Create randomized search
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=100,  # Number of parameter settings sampled
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Fit randomized search
random_search.fit(X, y)

# Compare with regular grid search
import time

# Time grid search (with smaller grid for demonstration)
small_param_grid = {
    'n_estimators': [10, 50, 100, 150],
    'max_depth': [10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

start_time = time.time()
grid_search = GridSearchCV(model, small_param_grid, cv=5, n_jobs=-1)
grid_search.fit(X, y)
grid_time = time.time() - start_time

# Time randomized search with same computation budget
start_time = time.time()
random_search = RandomizedSearchCV(
    model, param_distributions, n_iter=len(grid_search.cv_results_['params']), cv=5, n_jobs=-1
)
random_search.fit(X, y)
random_time = time.time() - start_time

print(f"Grid Search Time: {grid_time:.2f} seconds")
print(f"Grid Search Best Score: {grid_search.best_score_:.4f}")
print(f"Random Search Time: {random_time:.2f} seconds")
print(f"Random Search Best Score: {random_search.best_score_:.4f}")
```

### Successive Halving and Hyperband

```python
# Using HalvingGridSearchCV from scikit-learn
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [10, 50, 100, 200, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 8]
}

# Create halving grid search
halving_search = HalvingGridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    factor=3,           # Reduction factor
    resource='n_samples',  # Resource to increase progressively
    max_resources='auto',  # Maximum resource to use
    aggressive_elimination=False,  # Less aggressive elimination
    random_state=42
)

# Fit halving search
halving_search.fit(X, y)

# Print results
print("Best Parameters:", halving_search.best_params_)
print("Best Score:", halving_search.best_score_)

# Examine the successive halving process
results = pd.DataFrame(halving_search.cv_results_)
results_per_iteration = results.groupby('iter').agg({
    'mean_test_score': ['mean', 'min', 'max', 'std'],
    'n_resources': 'mean',
    'params': 'count'
})

print("\nSuccessive Halving Process:")
print(results_per_iteration)

# Visualize the successive halving process
plt.figure(figsize=(12, 6))

iterations = results['iter'].unique()
for iteration in iterations:
    subset = results[results['iter'] == iteration]
    plt.scatter(
        subset['param_n_estimators'], 
        subset['mean_test_score'], 
        label=f'Iteration {iteration}',
        alpha=0.7,
        s=100 / (iteration + 1)  # Size decreases with iteration
    )

plt.xlabel('n_estimators')
plt.ylabel('Mean Test Score')
plt.title('Successive Halving: Performance vs. n_estimators')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```

### Bayesian Optimization

```python
# Using scikit-optimize for Bayesian optimization
import skopt
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Define search space
search_space = {
    'n_estimators': Integer(10, 200),
    'max_depth': Integer(5, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.1, 0.9)
}

# Create Bayesian search
bayes_search = BayesSearchCV(
    estimator=model,
    search_spaces=search_space,
    n_iter=50,  # Number of evaluations
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Fit Bayesian search
bayes_search.fit(X, y)

# Print results
print("Best Parameters:", bayes_search.best_params_)
print("Best Score:", bayes_search.best_score_)

# Analyze optimization results
results = pd.DataFrame(bayes_search.cv_results_)

# Plot optimization history
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(results) + 1), -results['mean_test_score'], 'o-')
plt.axhline(y=-bayes_search.best_score_, color='r', linestyle='--', 
           label=f'Best Score: {bayes_search.best_score_:.4f}')
plt.xlabel('Iteration')
plt.ylabel('Negative Score (lower is better)')
plt.title('Bayesian Optimization History')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Plot feature importances from the optimizer
from skopt.plots import plot_evaluations, plot_objective

plt.figure(figsize=(15, 10))
plot_objective(bayes_search.optimizer_results_[0], dimensions=['n_estimators', 'max_depth'])
plt.show()
```

### Custom Grid Search Implementation

```python
def custom_grid_search(model_class, param_grid, X, y, cv=5, scoring='accuracy'):
    """
    Custom grid search implementation with early stopping.
    
    Parameters:
    -----------
    model_class : class
        The model class to use
    param_grid : dict
        Dictionary with parameters names as keys and lists of parameter values
    X : array-like
        Training data
    y : array-like
        Target variable
    cv : int
        Number of cross-validation folds
    scoring : str or callable
        Metric to evaluate models
        
    Returns:
    --------
    dict
        Best parameters, best score, and all results
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import get_scorer
    
    # Initialize variables
    best_score = -np.inf
    best_params = None
    all_results = []
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scorer = get_scorer(scoring)
    
    # Generate all parameter combinations
    from itertools import product
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    # Set early stopping threshold
    early_stop_threshold = 0.05  # Stop if score is 5% worse than current best
    
    # Evaluate each parameter combination
    for i, params_tuple in enumerate(param_combinations):
        params = dict(zip(param_names, params_tuple))
        fold_scores = []
        early_stop = False
        
        # Cross-validation
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            # Evaluate model
            score = scorer(model, X_val, y_val)
            fold_scores.append(score)
            
            # Check early stopping condition
            if best_score > -np.inf and score < best_score * (1 - early_stop_threshold):
                early_stop = True
                break
        
        # If early stopping triggered, skip to next combination
        if early_stop:
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            all_results.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'early_stopped': True
            })
            print(f"Combination {i+1}/{len(param_combinations)} early stopped.")
            continue
        
        # Calculate mean score
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        # Update best score and parameters
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
        
        # Store results
        all_results.append({
            'params': params,
            'mean_score': mean_score,
            'std_score': std_score,
            'early_stopped': False
        })
        
        print(f"Combination {i+1}/{len(param_combinations)}: {params}")
        print(f"  Score: {mean_score:.4f} ± {std_score:.4f}")
    
    # Return results
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': all_results
    }

# Example usage of custom grid search
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

results = custom_grid_search(
    model_class=RandomForestClassifier,
    param_grid=param_grid,
    X=X,
    y=y,
    cv=5,
    scoring='accuracy'
)

print("\nCustom Grid Search Results:")
print("Best Parameters:", results['best_params'])
print("Best Score:", results['best_score'])

# Analyze early stopping benefits
all_results = pd.DataFrame(results['all_results'])
early_stopped = all_results['early_stopped'].sum()
total_combinations = len(all_results)
print(f"Combinations early stopped: {early_stopped}/{total_combinations} ({early_stopped/total_combinations*100:.1f}%)")
```

---

## 🔬 Practical Applications

### Classification: Credit Risk Modeling

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, roc_auc_score

# Load credit dataset (example using a synthetic dataset)
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10, n_redundant=5,
    n_classes=2, weights=[0.7, 0.3], random_state=42
)

# Convert to DataFrame with feature names
feature_names = [f'feature_{i}' for i in range(20)]
X = pd.DataFrame(X, columns=feature_names)

# Add some categorical features for demonstration
X['feature_cat_1'] = np.random.choice(['A', 'B', 'C'], size=X.shape[0])
X['feature_cat_2'] = np.random.choice(['X', 'Y', 'Z'], size=X.shape[0])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numerical and categorical features
numerical_features = feature_names
categorical_features = ['feature_cat_1', 'feature_cat_2']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Create pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Define custom scoring function prioritizing recall (finding all defaults)
def custom_credit_score(y_true, y_pred):
    # Precision and recall
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    # F2 score (weights recall higher than precision)
    beta = 2
    f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    
    return f2

# Create custom scorer
credit_scorer = make_scorer(custom_credit_score)

# Define parameter grid
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__subsample': [0.8, 0.9, 1.0]
}

# Create grid search with multiple metrics
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring={
        'f2': credit_scorer,
        'auc': 'roc_auc',
        'recall': 'recall',
        'precision': 'precision'
    },
    refit='f2',  # Use F2 score to select the best model
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Print best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best F2 Score:", grid_search.best_score_)

# Evaluate best model on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Calculate various metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot ROC curve
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Feature importance analysis
best_classifier = best_model.named_steps['classifier']
feature_importances = best_classifier.feature_importances_

# Get feature names after preprocessing
preprocessor = best_model.named_steps['preprocessor']
preprocessed_features = numerical_features.copy()
for encoder_name, encoder, features in preprocessor.transformers_:
    if encoder_name == 'cat':
        # Get categorical feature names after one-hot encoding
        cat_features = []
        for i, feature in enumerate(features):
            categories = encoder.named_steps['encoder'].categories_[i]
            for category in categories:
                cat_features.append(f"{feature}_{category}")
        preprocessed_features.extend(cat_features)

# Display feature importances
importance_df = pd.DataFrame({
    'Feature': preprocessed_features[:len(feature_importances)],
    'Importance': feature_importances
})
importance_df = importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances')
plt.gca().invert_yaxis()  # Invert y-axis to have highest importance at the top
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```

### Regression: House Price Prediction

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load housing dataset (using California housing dataset)
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing
numerical_features = X.columns.tolist()

# Create preprocessing pipeline with polynomial features
preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(include_bias=False, degree=2))
])

# Create main pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# Define custom scoring function (RMSE)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)

# Define parameter grid
param_grid = {
    'preprocessor__poly__degree': [1, 2],  # Test with and without polynomial features
    'regressor__n_estimators': [100, 200, 300],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__max_depth': [3, 5, 7],
    'regressor__min_samples_split': [2, 5],
    'regressor__subsample': [0.8, 1.0]
}

# Create grid search with multiple metrics
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring={
        'rmse': rmse_scorer,
        'r2': 'r2',
        'mae': 'neg_mean_absolute_error'
    },
    refit='rmse',  # Use RMSE to select the best model
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Print best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best RMSE:", -grid_search.best_score_)  # Negate because RMSE is negated in scorer

# Evaluate best model on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate metrics
test_rmse = rmse(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)
test_mae = mean_absolute_error(y_test, y_pred)

print("\nTest Set Metrics:")
print(f"RMSE: {test_rmse:.4f}")
print(f"R²: {test_r2:.4f}")
print(f"MAE: {test_mae:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted House Prices')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Feature importance analysis
best_regressor = best_model.named_steps['regressor']
feature_importances = best_regressor.feature_importances_

# Check if polynomial features were used
poly_degree = best_model.named_steps['preprocessor'].named_steps['poly'].degree

if poly_degree > 1:
    # Get the polynomial feature names
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    poly.fit(X.values)
    feature_names = poly.get_feature_names_out(X.columns)
else:
    feature_names = X.columns

# Display feature importances (top 20)
n_features_to_show = min(20, len(feature_importances))
importance_df = pd.DataFrame({
    'Feature': feature_names[:len(feature_importances)],
    'Importance': feature_importances
})
importance_df = importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'][:n_features_to_show], importance_df['Importance'][:n_features_to_show])
plt.xlabel('Importance')
plt.title(f'Top {n_features_to_show} Feature Importances')
plt.gca().invert_yaxis()  # Invert y-axis to have highest importance at the top
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```

### Time Series Forecasting

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic time series data
np.random.seed(42)
date_rng = pd.date_range(start='2015-01-01', end='2020-12-31', freq='D')
n = len(date_rng)

# Create time series with trend, seasonality, and noise
time = np.arange(n)
trend = 0.01 * time
seasonality = 10 * np.sin(2 * np.pi * time / 365.25)
noise = np.random.normal(0, 1, n)

y = trend + seasonality + noise
ts_data = pd.DataFrame(date_rng, columns=['date'])
ts_data['value'] = y

# Create features from date
ts_data['year'] = ts_data['date'].dt.year
ts_data['month'] = ts_data['date'].dt.month
ts_data['day'] = ts_data['date'].dt.day
ts_data['dayofweek'] = ts_data['date'].dt.dayofweek
ts_data['dayofyear'] = ts_data['date'].dt.dayofyear
ts_data['quarter'] = ts_data['date'].dt.quarter

# Create lag features
for lag in range(1, 8):  # 1 to 7 day lags
    ts_data[f'lag_{lag}'] = ts_data['value'].shift(lag)

# Drop rows with NaN values (due to lags)
ts_data = ts_data.dropna()

# Define features and target
X = ts_data.drop(['date', 'value'], axis=1)
y = ts_data['value']

# Define time series cross-validation
tscv = TimeSeriesSplit(n_splits=5, test_size=30)

# Create pipeline with preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Define parameter grid
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

# Create grid search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Fit grid search
grid_search.fit(X, y)

# Print best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best RMSE:", np.sqrt(-grid_search.best_score_))

# Visualize cross-validation splits
plt.figure(figsize=(15, 10))
for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
    # Plot training and test indices
    plt.subplot(tscv.n_splits, 1, i+1)
    plt.plot(np.arange(len(train_idx)), [0] * len(train_idx), 'o-', color='blue', label='Training')
    plt.plot(np.arange(len(train_idx), len(train_idx) + len(test_idx)), [0] * len(test_idx), 'o-', color='red', label='Testing')
    plt.title(f'Split {i+1}')
    if i == 0:
        plt.legend()
plt.tight_layout()
plt.show()

# Make predictions using the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X)

# Plot actual vs. predicted
plt.figure(figsize=(15, 6))
plt.plot(ts_data['date'], y, label='Actual', alpha=0.7)
plt.plot(ts_data['date'], y_pred, label='Predicted', alpha=0.7, linestyle='--')
plt.title('Actual vs. Predicted Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Feature importance analysis
best_regressor = best_model.named_steps['regressor']
feature_importances = best_regressor.feature_importances_

# Display feature importances
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})
importance_df = importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.gca().invert_yaxis()  # Invert y-axis to have highest importance at the top
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Forecasting future values
# Create a function to forecast the next n days
def forecast_next_days(model, last_data, n_days=30):
    forecasts = []
    current_data = last_data.copy()
    
    for _ in range(n_days):
        # Make prediction for the next day
        next_prediction = model.predict(current_data.values.reshape(1, -1))[0]
        forecasts.append(next_prediction)
        
        # Update features for the next prediction
        # Shift lag features
        for i in range(7, 1, -1):
            current_data[f'lag_{i}'] = current_data[f'lag_{i-1}']
        current_data['lag_1'] = next_prediction
        
        # Update date features (simplified for demonstration)
        # In practice, you would update these based on the next date
        
    return forecasts

# Get the last row of data
last_row = X.iloc[-1]

# Forecast next 30 days
future_forecasts = forecast_next_days(best_model, last_row, n_days=30)

# Create future dates
last_date = ts_data['date'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')

# Plot historical data and forecasts
plt.figure(figsize=(15, 6))
plt.plot(ts_data['date'], y, label='Historical Data', alpha=0.7)
plt.plot(future_dates, future_forecasts, label='Forecast', alpha=0.7, linestyle='--', color='red')
plt.title('Time Series Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```

---

## 🔄 Comparison with Other Methods

### Grid Search vs. Random Search

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import time

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Calculate the total number of combinations
total_combinations = 1
for param_values in param_grid.values():
    total_combinations *= len(param_values)
print(f"Total number of combinations: {total_combinations}")

# Function to run a search and measure performance
def run_search(search_type, X_train, y_train, X_test, y_test, param_grid, n_iter=None):
    # Create base model
    model = RandomForestClassifier(random_state=42)
    
    # Create search
    if search_type == 'grid':
        search = GridSearchCV(
            model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0
        )
    else:  # random
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=3, scoring='accuracy', n_jobs=-1, 
            verbose=0, random_state=42
        )
    
    # Measure time
    start_time = time.time()
    search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # Evaluate on test set
    best_model = search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    
    # Return results
    return {
        'search_type': search_type,
        'best_params': search.best_params_,
        'best_cv_score': search.best_score_,
        'test_score': test_score,
        'time': search_time,
        'n_evaluations': len(search.cv_results_['params']),
        'cv_results': search.cv_results_
    }

# Run grid search
grid_results = run_search('grid', X_train, y_train, X_test, y_test, param_grid)

# Run random search with different numbers of iterations
random_iterations = [10, 50, 100, 200]
random_results = []

for n_iter in random_iterations:
    result = run_search('random', X_train, y_train, X_test, y_test, param_grid, n_iter=n_iter)
    result['n_iter'] = n_iter
    random_results.append(result)

# Combine all results
all_results = [grid_results] + random_results

# Print summary
print("\nSearch Method Comparison:")
print(f"{'Search Type':<15} {'Iterations':<10} {'CV Score':<10} {'Test Score':<10} {'Time (s)':<10}")
print("-" * 60)
for result in all_results:
    search_name = f"{result['search_type']}"
    if result['search_type'] == 'random':
        search_name += f" ({result['n_iter']})"
    print(f"{search_name:<15} {result['n_evaluations']:<10} {result['best_cv_score']:.4f}     {result['test_score']:.4f}     {result['time']:.2f}")

# Visualize time vs. performance
plt.figure(figsize=(12, 6))

# Extract data for plotting
times = [r['time'] for r in all_results]
cv_scores = [r['best_cv_score'] for r in all_results]
test_scores = [r['test_score'] for r in all_results]
labels = ['Grid Search'] + [f'Random ({n})' for n in random_iterations]

# Plot
plt.scatter(times, test_scores, s=100, label='Test Score')
plt.scatter(times, cv_scores, s=100, label='CV Score')

# Add labels
for i, label in enumerate(labels):
    plt.annotate(label, (times[i], test_scores[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('Time (seconds)')
plt.ylabel('Accuracy')
plt.title('Search Time vs. Performance')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Visualize the convergence of random search
plt.figure(figsize=(12, 6))

# For each random search result, plot the best score vs. number of iterations
for result in random_results:
    cv_results = pd.DataFrame(result['cv_results'])
    cv_results = cv_results.sort_values('mean_test_score', ascending=False)
    
    # Calculate the best score found after each evaluation
    best_scores = [cv_results['mean_test_score'].iloc[:i+1].max() for i in range(len(cv_results))]
    
    plt.plot(range(1, len(best_scores) + 1), best_scores, 
             label=f"Random Search ({result['n_iter']} iterations)")

# Add grid search best score as a reference line
plt.axhline(y=grid_results['best_cv_score'], color='r', linestyle='--', 
           label=f'Grid Search Best Score ({grid_results["best_cv_score"]:.4f})')

plt.xlabel('Number of Evaluations')
plt.ylabel('Best CV Score Found')
plt.title('Convergence of Random Search')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```

### Grid Search vs. Bayesian Optimization

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing
import time

# For Bayesian optimization
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Load dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid for grid search
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 0.9, 1.0]
}

# Define search space for Bayesian optimization
bayes_search_space = {
    'n_estimators': Integer(50, 200),
    'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
    'max_depth': Integer(3, 7),
    'min_samples_split': Integer(2, 10),
    'subsample': Real(0.8, 1.0)
}

# Function to run grid search
def run_grid_search(X_train, y_train, X_test, y_test, param_grid):
    # Create base model
    model = GradientBoostingRegressor(random_state=42)
    
    # Create grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0
    )
    
    # Measure time
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    test_score = -mean_squared_error(y_test, best_model.predict(X_test))
    test_rmse = np.sqrt(-test_score)
    
    # Return results
    return {
        'best_params': grid_search.best_params_,
        'best_cv_score': -grid_search.best_score_,  # Convert back to positive MSE
        'test_mse': -test_score,
        'test_rmse': test_rmse,
        'time': search_time,
        'n_evaluations': len(grid_search.cv_results_['params']),
        'cv_results': grid_search.cv_results_
    }

# Function to run Bayesian optimization
def run_bayes_search(X_train, y_train, X_test, y_test, search_space, n_iter):
    # Create base model
    model = GradientBoostingRegressor(random_state=42)
    
    # Create Bayesian search
    bayes_search = BayesSearchCV(
        model, search_space, n_iter=n_iter, cv=3, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=0, random_state=42
    )
    
    # Measure time
    start_time = time.time()
    bayes_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # Evaluate on test set
    best_model = bayes_search.best_estimator_
    test_score = -mean_squared_error(y_test, best_model.predict(X_test))
    test_rmse = np.sqrt(-test_score)
    
    # Return results
    return {
        'best_params': bayes_search.best_params_,
        'best_cv_score': -bayes_search.best_score_,  # Convert back to positive MSE
        'test_mse': -test_score,
        'test_rmse': test_rmse,
        'time': search_time,
        'n_evaluations': len(bayes_search.cv_results_['params']),
        'cv_results': bayes_search.cv_results_,
        'optimizer_results': bayes_search.optimizer_results_
    }

# Run grid search
print("Running Grid Search...")
grid_results = run_grid_search(X_train, y_train, X_test, y_test, param_grid)

# Calculate total number of combinations in grid search
total_combinations = 1
for param_values in param_grid.values():
    total_combinations *= len(param_values)

# Run Bayesian optimization with different numbers of iterations
print(f"\nRunning Bayesian Optimization...")
bayes_iterations = [10, 30, 60, 90]
bayes_results = []

for n_iter in bayes_iterations:
    print(f"  - With {n_iter} iterations...")
    result = run_bayes_search(X_train, y_train, X_test, y_test, bayes_search_space, n_iter=n_iter)
    result['n_iter'] = n_iter
    bayes_results.append(result)

# Combine all results
all_results = [grid_results] + bayes_results

# Print summary
print("\nSearch Method Comparison:")
print(f"{'Search Type':<20} {'Iterations':<10} {'CV RMSE':<10} {'Test RMSE':<10} {'Time (s)':<10}")
print("-" * 65)
for result in all_results:
    search_name = "Grid Search" if 'n_iter' not in result else f"Bayesian ({result['n_iter']})"
    cv_rmse = np.sqrt(result['best_cv_score'])
    print(f"{search_name:<20} {result['n_evaluations']:<10} {cv_rmse:.4f}     {result['test_rmse']:.4f}     {result['time']:.2f}")

# Visualize time vs. performance
plt.figure(figsize=(12, 6))

# Extract data for plotting
times = [r['time'] for r in all_results]
cv_rmses = [np.sqrt(r['best_cv_score']) for r in all_results]
test_rmses = [r['test_rmse'] for r in all_results]
labels = ['Grid Search'] + [f'Bayesian ({n})' for n in bayes_iterations]

# Plot
plt.scatter(times, test_rmses, s=100, label='Test RMSE')
plt.scatter(times, cv_rmses, s=100, label='CV RMSE')

# Add labels
for i, label in enumerate(labels):
    plt.annotate(label, (times[i], test_rmses[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('Time (seconds)')
plt.ylabel('RMSE (lower is better)')
plt.title('Search Time vs. Performance')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Visualize the convergence of Bayesian optimization
plt.figure(figsize=(12, 6))

for i, result in enumerate(bayes_results):
    # Get the optimization results
    opt_result = result['optimizer_results'][0]
    
    # Extract the function values (negative MSE)
    func_vals = -np.array(opt_result.func_vals)
    
    # Convert to RMSE
    rmses = np.sqrt(-func_vals)
    
    # Calculate the best RMSE found after each iteration
    best_rmses = [rmses[:i+1].min() for i in range(len(rmses))]
    
    plt.plot(range(1, len(best_rmses) + 1), best_rmses, 
             label=f"Bayesian ({result['n_iter']} iterations)")

# Add grid search best score as a reference line
grid_rmse = np.sqrt(grid_results['best_cv_score'])
plt.axhline(y=grid_rmse, color='r', linestyle='--', 
           label=f'Grid Search Best RMSE ({grid_rmse:.4f})')

plt.xlabel('Number of Evaluations')
plt.ylabel('Best RMSE Found (lower is better)')
plt.title('Convergence of Bayesian Optimization')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Analyze the exploration-exploitation trade-off in Bayesian optimization
plt.figure(figsize=(15, 8))

# Choose the Bayesian result with the most iterations for this analysis
bayes_result = bayes_results[-1]
opt_result = bayes_result['optimizer_results'][0]

# Extract data
X_sample = opt_result.x_iters
y_sample = -np.array(opt_result.func_vals)
y_sample_rmse = np.sqrt(-y_sample)

# Extract two important parameters for visualization
param1 = 'learning_rate'
param2 = 'max_depth'

# Get the indices of these parameters
param_names = list(bayes_search_space.keys())
param1_idx = param_names.index(param1)
param2_idx = param_names.index(param2)

# Extract values for these parameters
param1_values = [x[param1_idx] for x in X_sample]
param2_values = [x[param2_idx] for x in X_sample]

# Create scatter plot colored by performance and size by iteration
plt.scatter(param1_values, param2_values, c=y_sample_rmse, cmap='viridis', 
            s=[30 * (i+1) for i in range(len(y_sample))], alpha=0.7)

plt.colorbar(label='RMSE (lower is better)')
plt.xlabel(param1)
plt.ylabel(param2)
plt.title('Bayesian Optimization Parameter Exploration')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```

### Grid Search vs. Evolutionary Algorithms

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import time

# For evolutionary algorithm
from evolutionary_search import EvolutionaryAlgorithmSearchCV

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Run grid search
print("Running Grid Search...")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

start_time = time.time()
grid_search.fit(X_train, y_train)
grid_time = time.time() - start_time

# Run evolutionary search
print("Running Evolutionary Search...")
evolutionary_search = EvolutionaryAlgorithmSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    params=param_grid,
    scoring="accuracy",
    cv=3,
    verbose=1,
    population_size=10,
    gene_mutation_prob=0.10,
    gene_crossover_prob=0.5,
    tournament_size=3,
    generations_number=5,
    n_jobs=-1
)

start_time = time.time()
evolutionary_search.fit(X_train, y_train)
evolutionary_time = time.time() - start_time

# Compare results
print("\nSearch Method Comparison:")
print(f"{'Search Type':<20} {'Best Score':<15} {'Test Accuracy':<15} {'Time (s)':<10}")
print("-" * 65)

# Grid search results
grid_best_model = grid_search.best_estimator_
grid_test_score = grid_best_model.score(X_test, y_test)
print(f"Grid Search{'':<13} {grid_search.best_score_:.4f}{'':<10} {grid_test_score:.4f}{'':<10} {grid_time:.2f}")

# Evolutionary search results
evo_best_model = evolutionary_search.best_estimator_
evo_test_score = evo_best_model.score(X_test, y_test)
print(f"Evolutionary Search{'':<4} {evolutionary_search.best_score_:.4f}{'':<10} {evo_test_score:.4f}{'':<10} {evolutionary_time:.2f}")

# Visualize comparison
methods = ['Grid Search', 'Evolutionary Search']
times = [grid_time, evolutionary_time]
cv_scores = [grid_search.best_score_, evolutionary_search.best_score_]
test_scores = [grid_test_score, evo_test_score]

plt.figure(figsize=(12, 6))

# Plot time comparison
plt.subplot(1, 2, 1)
plt.bar(methods, times)
plt.title('Computation Time')
plt.ylabel('Time (seconds)')
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

# Plot accuracy comparison
plt.subplot(1, 2, 2)
x = np.arange(len(methods))
width = 0.35
plt.bar(x - width/2, cv_scores, width, label='CV Accuracy')
plt.bar(x + width/2, test_scores, width, label='Test Accuracy')
plt.xticks(x, methods)
plt.title('Model Performance')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

plt.tight_layout()
plt.show()

# Analyze parameter distributions
grid_params = pd.DataFrame(grid_search.cv_results_['params'])
grid_scores = grid_search.cv_results_['mean_test_score']

# Analyze relationship between parameters and performance
plt.figure(figsize=(15, 10))

# For each parameter, plot its effect on performance
for i, param_name in enumerate(param_grid.keys()):
    plt.subplot(2, 2, i+1)
    
    # Group by parameter value and calculate mean score
    param_values = grid_params[param_name].values
    unique_values = np.unique(param_values)
    
    # Handle 'None' values
    if None in unique_values:
        param_values = [str(val) for val in param_values]
        unique_values = [str(val) for val in unique_values]
    
    # Calculate mean score for each parameter value
    mean_scores = []
    for val in unique_values:
        mask = np.array(param_values) == val
        mean_scores.append(np.mean(grid_scores[mask]))
    
    plt.bar(unique_values, mean_scores)
    plt.title(f'Effect of {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Mean CV Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

plt.tight_layout()
plt.show()
```

---

## ⚠️ Common Pitfalls

### Computational Complexity

```python
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Function to calculate number of combinations
def calculate_combinations(param_grid):
    n_combinations = 1
    for param_values in param_grid.values():
        n_combinations *= len(param_values)
    return n_combinations

# Function to estimate grid search time
def estimate_grid_search_time(param_grid, n_samples, n_features, cv=5, baseline_time=None):
    """Estimate grid search time based on number of combinations and data size."""
    n_combinations = calculate_combinations(param_grid)
    
    if baseline_time is None:
        # Generate small dataset and measure baseline time
        X_small, y_small = make_classification(n_samples=100, n_features=5, random_state=42)
        
        # Simple parameter grid for baseline
        simple_grid = {'n_estimators': [10], 'max_depth': [3]}
        
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(model, simple_grid, cv=cv)
        
        start_time = time.time()
        grid_search.fit(X_small, y_small)
        baseline_time = time.time() - start_time
    
    # Scaling factors
    samples_factor = n_samples / 100  # Relative to baseline 100 samples
    features_factor = n_features / 5  # Relative to baseline 5 features
    combinations_factor = n_combinations  # Number of parameter combinations
    
    # Estimated time with linear scaling for samples and features, and linear for combinations
    estimated_time = baseline_time * samples_factor * np.sqrt(features_factor) * combinations_factor
    
    return estimated_time, n_combinations

# Visualize how grid search time scales with different factors
# 1. Number of combinations
param_values = [2, 3, 4, 5, 6]
combinations = []
estimated_times = []

for n_values in param_values:
    param_grid = {
        'n_estimators': list(range(10, 10 * n_values + 1, 10)),
        'max_depth': list(range(1, n_values + 1)),
        'min_samples_split': list(range(2, n_values + 2))
    }
    
    time_est, n_combinations = estimate_grid_search_time(param_grid, 1000, 10)
    combinations.append(n_combinations)
    estimated_times.append(time_est)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.bar(param_values, combinations)
plt.yscale('log')
plt.xlabel('Number of Values per Hyperparameter')
plt.ylabel('Number of Combinations (log scale)')
plt.title('Combinations vs. Values per Parameter')
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

plt.subplot(1, 3, 2)
plt.bar(param_values, estimated_times)
plt.yscale('log')
plt.xlabel('Number of Values per Hyperparameter')
plt.ylabel('Estimated Time (seconds, log scale)')
plt.title('Estimated Grid Search Time')
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

# 2. Dataset size effect
dataset_sizes = [1000, 5000, 10000, 50000, 100000]
size_times = []

basic_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

for size in dataset_sizes:
    time_est, _ = estimate_grid_search_time(basic_grid, size, 10)
    size_times.append(time_est)

plt.subplot(1, 3, 3)
plt.bar(dataset_sizes, size_times)
plt.yscale('log')
plt.xlabel('Dataset Size (samples)')
plt.ylabel('Estimated Time (seconds, log scale)')
plt.title('Effect of Dataset Size on Grid Search Time')
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

plt.tight_layout()
plt.show()

# Practical strategies to mitigate computational complexity
print("Strategies to Mitigate Computational Complexity:")
print("1. Use a coarse grid first, then refine around promising values")
print("2. Use RandomizedSearchCV instead of GridSearchCV for large search spaces")
print("3. Reduce dataset size for initial exploration")
print("4. Reduce cross-validation folds for preliminary searches")
print("5. Use early stopping criteria when available")
print("6. Implement parallel processing (n_jobs=-1)")
print("7. Use more efficient alternatives like Bayesian Optimization for expensive models")
print("8. Focus on the most important hyperparameters first")
```

### Overfitting to the Validation Set

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Generate a synthetic dataset
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10, n_redundant=5,
    n_classes=2, random_state=42
)

# Split into train, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [10, 50, 100, 200, 500],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None]
}

# Perform grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_val, y_train_val)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate on validation and test sets
val_score = best_model.score(X_val, y_val)
test_score = best_model.score(X_test, y_test)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Cross-Validation Score: {grid_search.best_score_:.4f}")
print(f"Validation Score: {val_score:.4f}")
print(f"Test Score: {test_score:.4f}")

# Analyze the top N models to check for overfitting
results = pd.DataFrame(grid_search.cv_results_)
results = results.sort_values('rank_test_score')

# Get top 20 models
top_models = results.head(20)

# Evaluate each of these models on the test set
test_scores = []
val_scores = []
cv_scores = []

for params in top_models['params']:
    model = RandomForestClassifier(random_state=42, **params)
    model.fit(X_train, y_train)
    
    # Score on validation and test sets
    val_score = model.score(X_val, y_val)
    test_score = model.score(X_test, y_test)
    
    val_scores.append(val_score)
    test_scores.append(test_score)
    
    # Get the CV score from the results
    cv_score = results[results['params'].apply(lambda x: x == params)]['mean_test_score'].values[0]
    cv_scores.append(cv_score)

# Add scores to the top models DataFrame
top_models['cv_score'] = cv_scores
top_models['val_score'] = val_scores
top_models['test_score'] = test_scores
top_models['cv_test_diff'] = top_models['cv_score'] - top_models['test_score']

# Print top 5 models with all scores
print("\nTop 5 Models Performance:")
print(top_models[['params', 'cv_score', 'val_score', 'test_score', 'cv_test_diff']].head(5))

# Visualize overfitting
plt.figure(figsize=(12, 6))

# Sort by CV score
model_indices = range(len(top_models))
plt.plot(model_indices, top_models['cv_score'], 'o-', label='CV Score')
plt.plot(model_indices, top_models['val_score'], 's-', label='Validation Score')
plt.plot(model_indices, top_models['test_score'], 'd-', label='Test Score')

plt.xlabel('Model Rank')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Plot the learning curves for the best model
def plot_learning_curves(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-Validation Score')
    
    plt.title(title)
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()

# Plot learning curves for the best model
plot_learning_curves(best_model, X_train_val, y_train_val, 'Learning Curves for Best Model')

# Strategies to avoid overfitting to the validation set
print("\nStrategies to Avoid Overfitting to the Validation Set:")
print("1. Use nested cross-validation")
print("2. Reserve a separate holdout test set")
print("3. Use regularization in your models")
print("4. Be cautious about too many hyperparameter combinations")
print("5. Analyze learning curves for signs of overfitting")
print("6. Consider the stability of model performance across different CV folds")
print("7. Be wary of parameters that give perfect scores on validation data")
print("8. Use different random seeds and average results")
```

### Misleading Performance Metrics

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.datasets import make_classification

# Generate an imbalanced dataset
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5, n_redundant=2,
    n_classes=2, weights=[0.9, 0.1], random_state=42  # 90% class 0, 10% class 1
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Print class distribution
train_class_counts = np.bincount(y_train)
test_class_counts = np.bincount(y_test)
print("Class distribution in training set:", train_class_counts)
print("Class distribution in test set:", test_class_counts)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'class_weight': [None, 'balanced']
}

# Define multiple scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score)
}

# Perform grid search with multiple metrics
grid_search_multi = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring=scoring,
    refit='accuracy',  # Use accuracy for refit (default behavior)
    n_jobs=-1,
    return_train_score=True
)

grid_search_multi.fit(X_train, y_train)

# Get results for all metrics
results = pd.DataFrame(grid_search_multi.cv_results_)

# Get the best parameters according to each metric
best_accuracy_idx = results['rank_test_accuracy'].idxmin()
best_precision_idx = results['rank_test_precision'].idxmin()
best_recall_idx = results['rank_test_recall'].idxmin()
best_f1_idx = results['rank_test_f1'].idxmin()
best_roc_auc_idx = results['rank_test_roc_auc'].idxmin()

best_models = {
    'Accuracy': {
        'params': results.loc[best_accuracy_idx, 'params'],
        'cv_score': results.loc[best_accuracy_idx, 'mean_test_accuracy']
    },
    'Precision': {
        'params': results.loc[best_precision_idx, 'params'],
        'cv_score': results.loc[best_precision_idx, 'mean_test_precision']
    },
    'Recall': {
        'params': results.loc[best_recall_idx, 'params'],
        'cv_score': results.loc[best_recall_idx, 'mean_test_recall']
    },
    'F1': {
        'params': results.loc[best_f1_idx, 'params'],
        'cv_score': results.loc[best_f1_idx, 'mean_test_f1']
    },
    'ROC AUC': {
        'params': results.loc[best_roc_auc_idx, 'params'],
        'cv_score': results.loc[best_roc_auc_idx, 'mean_test_roc_auc']
    }
}

# Evaluate each "best model" on the test set
for metric, model_info in best_models.items():
    # Create and train model with the best parameters
    model = RandomForestClassifier(random_state=42, **model_info['params'])
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate all metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    # Store results
    model_info['test_metrics'] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    model_info['confusion_matrix'] = cm
    
    print(f"\nBest model according to {metric}:")
    print(f"Parameters: {model_info['params']}")
    print(f"CV {metric} Score: {model_info['cv_score']:.4f}")
    print(f"Test Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"Confusion Matrix:")
    print(cm)

# Visualize the differences in metrics
plt.figure(figsize=(15, 10))

# Extract metrics for comparison
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
model_names = list(best_models.keys())

for i, metric in enumerate(metrics):
    plt.subplot(2, 3, i+1)
    
    values = [best_models[model]['test_metrics'][metric] for model in model_names]
    plt.bar(model_names, values)
    
    plt.title(f'Test {metric.upper()}')
    plt.ylabel(metric.upper())
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

# Plot confusion matrices
plt.subplot(2, 3, 6)
# Choose two contrasting models: accuracy-optimized vs recall-optimized
acc_cm = best_models['Accuracy']['confusion_matrix']
recall_cm = best_models['Recall']['confusion_matrix']

# Calculate the difference
diff_cm = recall_cm - acc_cm

plt.imshow(diff_cm, cmap='coolwarm', vmin=-np.max(np.abs(diff_cm)), vmax=np.max(np.abs(diff_cm)))
plt.colorbar(label='Difference in counts')
plt.title('Confusion Matrix Difference\n(Recall - Accuracy)')
plt.xticks([0, 1], ['Predicted 0', 'Predicted 1'])
plt.yticks([0, 1], ['Actual 0', 'Actual 1'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(diff_cm[i, j]), ha='center', va='center', 
                 color='white' if abs(diff_cm[i, j]) > np.max(np.abs(diff_cm))/2 else 'black')

plt.tight_layout()
plt.show()

# Best practices for metric selection
print("\nBest Practices for Metric Selection:")
print("1. Choose metrics that align with business objectives")
print("2. For imbalanced datasets, avoid accuracy as the primary metric")
print("3. Consider the cost of false positives vs. false negatives")
print("4. Use multiple metrics to get a complete picture of model performance")
print("5. Understand the trade-offs between different metrics")
print("6. For classification problems with imbalanced data, consider:")
print("   - Precision-Recall curves")
print("   - F1 score or F-beta scores")
print("   - ROC AUC")
print("   - Cohen's Kappa")
print("7. For regression problems, consider:")
print("   - RMSE vs. MAE based on sensitivity to outliers")
print("   - R-squared for explained variance")
print("   - Domain-specific metrics (e.g., MAPE for time series)")
```

---

## ❓ FAQ

### Q1: How many hyperparameter combinations should I try?

**A:** The number of hyperparameter combinations to try depends on several factors:

1. **Computational resources**: More combinations require more computation time and resources.
2. **Model complexity**: Complex models with many hyperparameters need more thorough exploration.
3. **Dataset size**: Larger datasets make each evaluation more expensive.
4. **Project timeline**: Consider your time constraints.
5. **Prior knowledge**: If you have domain knowledge, you might focus on a narrower range.

As a general guideline:

- **Small-scale projects**: 20-50 combinations
- **Medium-scale projects**: 50-200 combinations
- **Large-scale projects**: 200+ combinations, possibly using more efficient methods like Random Search or Bayesian Optimization

If you're using Grid Search, you can calculate the total number of combinations as:

```python
total_combinations = 1
for param_values in param_grid.values():
    total_combinations *= len(param_values)
```

If this number is very large (>1000), consider:
1. Reducing the number of values per hyperparameter
2. Using Random Search instead
3. Using a multi-stage approach (coarse search, then fine-tuning)

Remember that having too many combinations can lead to overfitting the validation set, especially with smaller datasets.

### Q2: How do I choose the right hyperparameter ranges?

**A:** Choosing the right hyperparameter ranges is crucial for effective grid search:

1. **Start with literature and documentation**:
   - Look at recommended values in model documentation
   - Check research papers using similar models
   - Review default values as a starting point

2. **Use logarithmic scales** for parameters that span orders of magnitude:
   - Learning rates: [0.0001, 0.001, 0.01, 0.1, 1.0]
   - Regularization strengths: [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

3. **Consider parameter relationships**:
   - Some parameters have interactions (e.g., learning rate and batch size)
   - Use conditional parameters when appropriate

4. **Start broad, then refine**:
   - Begin with a coarse grid covering a wide range
   - Perform a more focused search around promising values

5. **Use domain knowledge**:
   - Incorporate prior experience with similar problems
   - Consider computational constraints when setting upper bounds

6. **Model-specific guidelines**:

   For Random Forests:
   ```python
   param_grid = {
       'n_estimators': [10, 50, 100, 200, 500],  # Higher often better but diminishing returns
       'max_depth': [None, 10, 20, 30, 40, 50],  # None = unlimited
       'min_samples_split': [2, 5, 10, 20],      # Default = 2
       'min_samples_leaf': [1, 2, 4, 8]          # Default = 1
   }
   ```

   For SVMs:
   ```python
   param_grid = {
       'C': [0.1, 1, 10, 100],              # Regularization parameter
       'gamma': [0.001, 0.01, 0.1, 1],      # Kernel coefficient
       'kernel': ['rbf', 'linear', 'poly']  # Kernel type
   }
   ```

   For Neural Networks:
   ```python
   param_grid = {
       'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
       'activation': ['relu', 'tanh'],
       'alpha': [0.0001, 0.001, 0.01],
       'learning_rate_init': [0.001, 0.01, 0.1]
   }
   ```

7. **Validate your ranges**:
   - If the best performance is at the edge of your range, expand it
   - If performance plateaus, you can narrow the range

### Q3: Should I use Grid Search or Random Search?

**A:** The choice between Grid Search and Random Search depends on your specific scenario:

**Grid Search is better when:**

1. You have a small number of hyperparameters to tune (1-3)
2. You have good prior knowledge about promising parameter ranges
3. You need exhaustive exploration of a well-defined search space
4. Reproducibility of exact results is critical
5. Computational resources are not a major constraint

**Random Search is better when:**

1. You have many hyperparameters (4+)
2. Some parameters are much more important than others
3. You have limited computational resources
4. You want to explore a larger space more efficiently
5. You're in the early stages of model development

**Comparison:**

| Aspect | Grid Search | Random Search |
|--------|-------------|---------------|
| Coverage | Exhaustive within grid | Probabilistic coverage |
| Efficiency | Lower with many parameters | Higher with many parameters |
| Reproducibility | Fully reproducible | Reproducible with fixed seed |
| Implementation | Simpler to set up | Nearly as simple |
| Discovery | May miss optimal regions between grid points | Better at finding important parameters |

**Practical advice:**

1. **Start with Random Search** if you have more than 3-4 hyperparameters
2. Use Grid Search for **final fine-tuning** in a narrow range
3. For the **best of both worlds**, consider:
   - Using Random Search first to identify important parameters
   - Following up with Grid Search focused on those parameters
   - Or using Bayesian Optimization for the most efficient search

A hybrid approach example:

```python
# Step 1: Broad Random Search
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=broad_param_space,
    n_iter=100,
    cv=5,
    random_state=42
)
random_search.fit(X, y)

# Step 2: Focused Grid Search
best_params = random_search.best_params_
focused_param_grid = {
    param: [best_params[param] * 0.8, best_params[param], best_params[param] * 1.2]
    for param in best_params
    if isinstance(best_params[param], (int, float))
}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=focused_param_grid,
    cv=5
)
grid_search.fit(X, y)
```

### Q4: How do I handle mixed continuous and discrete hyperparameters?

**A:** Handling mixed continuous and discrete hyperparameters requires a thoughtful approach:

1. **Grid Search approach**:
   - For continuous parameters, discretize them into a reasonable number of values
   - For integer parameters, select specific values rather than ranges
   - For categorical parameters, include all relevant options

   ```python
   param_grid = {
       # Continuous parameter discretized
       'learning_rate': [0.001, 0.01, 0.1, 0.2],
       
       # Integer parameter
       'max_depth': [3, 5, 7, 10],
       
       # Categorical parameter
       'criterion': ['gini', 'entropy'],
       
       # Boolean parameter
       'bootstrap': [True, False]
   }
   ```

2. **Random Search approach**:
   - Use appropriate distributions for each parameter type
   - Continuous: uniform or log-uniform distributions
   - Integer: randint or discrete uniform distributions
   - Categorical: categorical distribution

   ```python
   from scipy.stats import uniform, randint
   
   param_distributions = {
       # Continuous parameter (log-uniform)
       'learning_rate': uniform(0.001, 0.2),
       
       # Integer parameter
       'max_depth': randint(3, 11),  # Upper bound is exclusive
       
       # Categorical parameter
       'criterion': ['gini', 'entropy'],
       
       # Boolean parameter
       'bootstrap': [True, False]
   }
   ```

3. **Conditional parameters**:
   - Some parameters only make sense with certain values of other parameters
   - For Grid Search, use parameter lists:

   ```python
   param_grid = [
       {'kernel': ['linear'], 'C': [0.1, 1, 10]},
       {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001]}
   ]
   ```

4. **Handling 'None' values**:
   - In Python, 'None' is a special value often used for "no limit"
   - Include it directly in your parameter grid:

   ```python
   param_grid = {
       'max_depth': [None, 10, 20, 30]  # None means unlimited depth
   }
   ```

5. **Mixed types with specialized libraries**:
   - For more complex scenarios, consider Bayesian Optimization libraries
   - These can handle mixed parameter types more elegantly:

   ```python
   from skopt import BayesSearchCV
   from skopt.space import Real, Integer, Categorical
   
   search_space = {
       'learning_rate': Real(0.001, 0.2, prior='log-uniform'),
       'max_depth': Integer(3, 10),
       'criterion': Categorical(['gini', 'entropy']),
       'bootstrap': Categorical([True, False])
   }
   
   opt = BayesSearchCV(
       estimator=model,
       search_spaces=search_space,
       n_iter=50,
       cv=5
   )
   ```

6. **Custom transformations**:
   - Sometimes you need to transform parameters before using them
   - Use a custom model wrapper or pipeline for this:

   ```python
   class ModelWithTransformedParams(BaseEstimator):
       def __init__(self, log_learning_rate=-5, max_depth=5):
           self.log_learning_rate = log_learning_rate
           self.max_depth = max_depth
       
       def fit(self, X, y):
           # Transform log_learning_rate to actual learning rate
           learning_rate = 10 ** self.log_learning_rate
           
           # Create and fit the actual model
           self.model_ = ActualModel(
               learning_rate=learning_rate,
               max_depth=self.max_depth
           )
           self.model_.fit(X, y)
           return self
   ```

### Q5: How do I avoid overfitting when using Grid Search?

**A:** Avoiding overfitting when using Grid Search is crucial for developing models that generalize well:

1. **Use cross-validation correctly**:
   - Use k-fold cross-validation (k=5 or k=10 is common)
   - Ensure stratification for classification problems
   - Consider repeated cross-validation for more stable estimates

   ```python
   from sklearn.model_selection import RepeatedStratifiedKFold
   
   cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
   grid_search = GridSearchCV(model, param_grid, cv=cv)
   ```

2. **Implement nested cross-validation**:
   - Use an outer loop for performance estimation
   - Use an inner loop for hyperparameter tuning

   ```python
   from sklearn.model_selection import cross_val_score, KFold
   
   outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
   inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
   
   outer_scores = []
   
   for train_idx, test_idx in outer_cv.split(X):
       X_train, X_test = X[train_idx], X[test_idx]
       y_train, y_test = y[train_idx], y[test_idx]
       
       grid_search = GridSearchCV(model, param_grid, cv=inner_cv)
       grid_search.fit(X_train, y_train)
       
       best_model = grid_search.best_estimator_
       score = best_model.score(X_test, y_test)
       outer_scores.append(score)
   
   print(f"Nested CV Score: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")
   ```

3. **Maintain a true holdout set**:
   - Keep a separate test set that is never used during model development
   - Only evaluate on this set once, after all tuning is complete

   ```python
   # Split data into train+validation and holdout test
   X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)
   
   # Use X_train_val for grid search with cross-validation
   grid_search.fit(X_train_val, y_train_val)
   
   # Only evaluate final model on X_test
   final_score = grid_search.best_estimator_.score(X_test, y_test)
   ```

4. **Limit the search space**:
   - Don't try too many hyperparameter combinations
   - Focus on parameters known to be important
   - Use domain knowledge to set reasonable ranges

5. **Include regularization parameters**:
   - Make sure to include regularization in your search
   - For many models, this means tuning parameters like:
     - `alpha` in Ridge/Lasso regression
     - `C` in SVM (smaller C = more regularization)
     - `min_samples_leaf` in tree-based models
     - `dropout_rate` in neural networks

   ```python
   param_grid = {
       'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],  # Regularization strength
       # Other parameters...
   }
   ```

6. **Monitor train vs. validation performance**:
   - Request training scores when performing grid search
   - Look for large gaps between training and validation performance

   ```python
   grid_search = GridSearchCV(model, param_grid, cv=5, return_train_score=True)
   
   # After fitting, examine the gap
   results = pd.DataFrame(grid_search.cv_results_)
   results['train_test_gap'] = results['mean_train_score'] - results['mean_test_score']
   
   # Check if the best model has a large gap
   best_idx = results['rank_test_score'] == 1
   best_gap = results.loc[best_idx, 'train_test_gap'].values[0]
   
   if best_gap > 0.1:  # Arbitrary threshold
       print("Warning: Best model may be overfitting!")
   ```

7. **Consider simpler models**:
   - Include simpler models in your comparison
   - Sometimes a well-tuned simple model outperforms a complex one

8. **Examine learning curves**:
   - Generate learning curves to understand the bias-variance tradeoff
   - Check if more data would help or if the model is too complex

   ```python
   from sklearn.model_selection import learning_curve
   
   train_sizes, train_scores, val_scores = learning_curve(
       grid_search.best_estimator_, X, y, cv=5, 
       train_sizes=np.linspace(0.1, 1.0, 10)
   )
   ```

---

<div align="center">

## 🌟 Key Takeaways

**Grid Search:**
- Systematically explores all combinations of hyperparameters to find the optimal model configuration
- Provides a thorough, exhaustive approach to hyperparameter tuning
- Works best with cross-validation to ensure robust performance estimation
- Can be computationally expensive but delivers reliable results within the defined search space
- Forms the foundation for more advanced optimization techniques

**Remember:**
- Start with a coarse grid and refine around promising values
- Consider computational complexity when defining your parameter grid
- Choose appropriate evaluation metrics aligned with your problem objectives
- Use cross-validation properly to avoid overfitting
- Consider alternatives like Random Search for high-dimensional parameter spaces
- Always validate your final model on a separate holdout set

---

### 📖 Happy Hyperparameter Tuning! 🚀

*Created by [@danialasim](https://github.com/danialasim) | Last updated: August 10, 2025*

</div>