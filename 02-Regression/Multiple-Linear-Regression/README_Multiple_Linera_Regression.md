# 📊 Multiple Linear Regression

<div align="center">

![Multiple Linear Regression](https://img.shields.io/badge/Algorithm-Multiple%20Linear%20Regression-blue?style=for-the-badge&logo=python)
![Status](https://img.shields.io/badge/Status-Complete-green?style=for-the-badge)
![Difficulty](https://img.shields.io/badge/Difficulty-Beginner%20to%20Intermediate-yellow?style=for-the-badge)

*Modeling Complex Relationships with Multiple Predictors*

</div>

---

## 📚 Table of Contents

- [What is Multiple Linear Regression?](#what-is-multiple-linear-regression)
- [Mathematical Foundation](#mathematical-foundation)
- [How It Works](#how-it-works)
- [Assumptions](#assumptions)
- [Implementation Guide](#implementation-guide)
- [Model Evaluation](#model-evaluation)
- [Feature Selection](#feature-selection)
- [Pros and Cons](#pros-and-cons)
- [Real-World Examples](#real-world-examples)
- [Advanced Topics](#advanced-topics)
- [FAQ](#faq)

---

## 🎯 What is Multiple Linear Regression?

**Multiple Linear Regression** extends simple linear regression by using multiple independent variables (features) to predict a single dependent variable. It models the linear relationship between several predictors and a continuous target variable.

### Key Characteristics:
- **Multiple Predictors**: Uses two or more independent variables
- **Linear Combination**: Combines features linearly
- **Continuous Output**: Predicts numerical values
- **Additive Effects**: Assumes each feature contributes independently

### The Goal:
Find the **best-fitting hyperplane** in multi-dimensional space that minimizes prediction error.

---

## 🧮 Mathematical Foundation

### The Multiple Linear Equation

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε
```

**Matrix Form:**
```
Y = Xβ + ε
```

Where:
- **y** = Dependent variable (n × 1)
- **X** = Design matrix (n × (p+1)) with intercept column
- **β** = Coefficient vector ((p+1) × 1)
- **ε** = Error vector (n × 1)
- **p** = Number of features
- **n** = Number of observations

### Design Matrix Structure

```
X = [1  x₁₁  x₁₂  ...  x₁ₚ]
    [1  x₂₁  x₂₂  ...  x₂ₚ]
    [⋮   ⋮    ⋮   ⋱    ⋮ ]
    [1  xₙ₁  xₙ₂  ...  xₙₚ]
```

### Parameter Estimation (Ordinary Least Squares)

**Normal Equation:**
```
β̂ = (XᵀX)⁻¹XᵀY
```

**Individual Coefficients:**
- **β₀** = Intercept (value when all predictors = 0)
- **βⱼ** = Change in y for unit change in xⱼ (holding others constant)

---

## ⚙️ How It Works

### Step-by-Step Process:

<div align="center">

```mermaid
graph TD
    A[Input: Multiple Features X₁, X₂, ..., Xₚ] --> B[Create Design Matrix X]
    B --> C[Calculate XᵀX and XᵀY]
    C --> D[Solve Normal Equation: β̂ = (XᵀX)⁻¹XᵀY]
    D --> E[Form Prediction Equation]
    E --> F[Make Predictions: ŷ = Xβ̂]
    F --> G[Evaluate Model Performance]
```

</div>

### Mathematical Example:

**Given Data:**
- House prices (y) predicted from size (x₁) and bedrooms (x₂)
- Data: (size, bedrooms, price) = [(1000, 2, 150k), (1500, 3, 200k), (2000, 4, 280k)]

**Design Matrix:**
```
X = [1  1000  2]    Y = [150000]
    [1  1500  3]        [200000]
    [1  2000  4]        [280000]
```

**Solution:**
```python
# Calculate β̂ = (XᵀX)⁻¹XᵀY
import numpy as np

X = np.array([[1, 1000, 2], [1, 1500, 3], [1, 2000, 4]])
Y = np.array([150000, 200000, 280000])

# Normal equation
XtX = X.T @ X
XtY = X.T @ Y
beta = np.linalg.inv(XtX) @ XtY

print(f"β₀ (intercept): {beta[0]:.2f}")
print(f"β₁ (size coeff): {beta[1]:.2f}")  
print(f"β₂ (bedroom coeff): {beta[2]:.2f}")
```

---

## 📋 Assumptions

Multiple Linear Regression requires the same assumptions as simple linear regression, plus additional considerations:

### 1. **Linearity** 🔵
- Linear relationship between each predictor and target
- **Check**: Partial regression plots

### 2. **Independence** 🟢
- Observations are independent
- **Check**: Domain knowledge and residual analysis

### 3. **Homoscedasticity** 🟡
- Constant variance of residuals
- **Check**: Residuals vs fitted values plot

### 4. **Normality of Residuals** 🟠
- Residuals follow normal distribution
- **Check**: Q-Q plot, Shapiro-Wilk test

### 5. **No Multicollinearity** 🔴
- Predictors are not highly correlated with each other
- **Check**: Variance Inflation Factor (VIF), correlation matrix

### 6. **No Perfect Multicollinearity** ⚫
- No predictor is a perfect linear combination of others
- **Check**: Matrix rank, condition number

---

## 💻 Implementation Guide

### From Scratch Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from scipy import stats

class MultipleLinearRegression:
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coefficients = None
        self.intercept = None
        self.fitted = False
        self.feature_names = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[list] = None) -> 'MultipleLinearRegression':
        """
        Fit the multiple linear regression model.
        
        Parameters:
        X (array-like): Independent variables (n_samples, n_features)
        y (array-like): Dependent variable (n_samples,)
        feature_names (list): Names of features for interpretation
        
        Returns:
        self: Returns the instance itself
        """
        X = np.array(X)
        y = np.array(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Add intercept column if needed
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_design = X
            
        # Solve normal equation: β = (X'X)^(-1)X'y
        try:
            XtX = X_design.T @ X_design
            XtY = X_design.T @ y
            
            # Check for singularity
            if np.linalg.cond(XtX) > 1e12:
                print("Warning: Design matrix is nearly singular. Consider removing correlated features.")
            
            beta = np.linalg.solve(XtX, XtY)  # More stable than matrix inversion
            
            if self.fit_intercept:
                self.intercept = beta[0]
                self.coefficients = beta[1:]
            else:
                self.intercept = 0
                self.coefficients = beta
                
        except np.linalg.LinAlgError:
            raise ValueError("Cannot solve normal equation. Check for perfect multicollinearity.")
        
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model."""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        return self.intercept + X @ self.coefficients
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate residuals."""
        return y - self.predict(X)
    
    def summary(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Provide detailed model summary with statistics."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
            
        X = np.array(X)
        y = np.array(y)
        
        y_pred = self.predict(X)
        residuals = y - y_pred
        n, p = X.shape[0], X.shape[1]
        
        # R-squared and Adjusted R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        # Standard errors and t-statistics
        mse = ss_res / (n - p - 1)
        
        # Design matrix for standard errors
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(n), X])
        else:
            X_design = X
            
        try:
            var_covar_matrix = mse * np.linalg.inv(X_design.T @ X_design)
            std_errors = np.sqrt(np.diag(var_covar_matrix))
            
            if self.fit_intercept:
                intercept_se = std_errors[0]
                coeff_se = std_errors[1:]
                intercept_t = self.intercept / intercept_se
                coeff_t = self.coefficients / coeff_se
            else:
                intercept_se = 0
                coeff_se = std_errors
                intercept_t = 0
                coeff_t = self.coefficients / coeff_se
                
            # P-values (two-tailed test)
            intercept_p = 2 * (1 - stats.t.cdf(abs(intercept_t), n - p - 1)) if self.fit_intercept else 1
            coeff_p = 2 * (1 - stats.t.cdf(np.abs(coeff_t), n - p - 1))
            
        except np.linalg.LinAlgError:
            # Handle singular matrix
            std_errors = np.full(len(self.coefficients), np.nan)
            coeff_t = np.full(len(self.coefficients), np.nan)
            coeff_p = np.full(len(self.coefficients), np.nan)
            intercept_se = intercept_t = intercept_p = np.nan
        
        return {
            'r_squared': r2,
            'adj_r_squared': adj_r2,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'intercept': self.intercept,
            'intercept_se': intercept_se,
            'intercept_t': intercept_t,
            'intercept_p': intercept_p,
            'coefficients': dict(zip(self.feature_names, self.coefficients)),
            'coeff_se': dict(zip(self.feature_names, coeff_se)),
            'coeff_t': dict(zip(self.feature_names, coeff_t)),
            'coeff_p': dict(zip(self.feature_names, coeff_p)),
            'n_observations': n,
            'n_features': p
        }
    
    def __str__(self) -> str:
        if not self.fitted:
            return "MultipleLinearRegression(not fitted)"
        
        equation = f"ŷ = {self.intercept:.3f}"
        for name, coef in zip(self.feature_names, self.coefficients):
            equation += f" + {coef:.3f}×{name}"
        return f"MultipleLinearRegression({equation})"
```

### Using Scikit-learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Sample data: House prices
data = {
    'size_sqft': [1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 1600, 1900],
    'bedrooms': [2, 3, 3, 4, 4, 5, 5, 6, 3, 4],
    'bathrooms': [1, 2, 2, 3, 3, 3, 4, 4, 2, 3],
    'age_years': [10, 5, 15, 20, 8, 12, 3, 1, 7, 11],
    'price': [150000, 200000, 250000, 280000, 320000, 380000, 420000, 450000, 230000, 290000]
}

df = pd.DataFrame(data)

# Prepare features and target
X = df[['size_sqft', 'bedrooms', 'bathrooms', 'age_years']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")

# Model interpretation
print(f"\nModel Equation:")
print(f"Price = {model.intercept_:,.0f}", end="")
for feature, coef in zip(X.columns, model.coef_):
    print(f" + {coef:.2f}×{feature}", end="")
print()

# Feature importance (absolute coefficient values)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print("\nFeature Importance (by absolute coefficient):")
print(feature_importance)
```

### Complete Example with Diagnostics

```python
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Advanced analysis function
def comprehensive_analysis(X, y, feature_names=None):
    """Perform comprehensive multiple linear regression analysis."""
    
    # Fit model
    model = MultipleLinearRegression()
    model.fit(X, y, feature_names)
    
    # Get detailed summary
    summary = model.summary(X, y)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Actual vs Predicted
    y_pred = model.predict(X)
    axes[0, 0].scatter(y, y_pred, alpha=0.6)
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Actual vs Predicted')
    
    # 2. Residuals vs Fitted
    residuals = model.get_residuals(X, y)
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='red', linestyle='--')
    axes[0, 1].set_xlabel('Fitted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Fitted')
    
    # 3. Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title('Q-Q Plot of Residuals')
    
    # 4. Feature correlation heatmap
    corr_matrix = np.corrcoef(X.T)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=feature_names or range(X.shape[1]),
                yticklabels=feature_names or range(X.shape[1]),
                ax=axes[1, 0])
    axes[1, 0].set_title('Feature Correlation Matrix')
    
    # 5. Coefficient plot
    if feature_names:
        axes[1, 1].barh(feature_names, model.coefficients)
        axes[1, 1].set_xlabel('Coefficient Value')
        axes[1, 1].set_title('Feature Coefficients')
    
    # 6. Residual histogram
    axes[1, 2].hist(residuals, bins=15, alpha=0.7, edgecolor='black')
    axes[1, 2].set_xlabel('Residuals')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Residual Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("=" * 60)
    print("MULTIPLE LINEAR REGRESSION SUMMARY")
    print("=" * 60)
    print(f"R-squared: {summary['r_squared']:.4f}")
    print(f"Adjusted R-squared: {summary['adj_r_squared']:.4f}")
    print(f"RMSE: {summary['rmse']:.4f}")
    print(f"Number of observations: {summary['n_observations']}")
    print(f"Number of features: {summary['n_features']}")
    
    print("\nCoefficients:")
    print("-" * 40)
    if model.fit_intercept:
        print(f"Intercept: {summary['intercept']:.4f} (p={summary['intercept_p']:.4f})")
    
    for feature in feature_names or range(X.shape[1]):
        coef = summary['coefficients'][feature]
        p_val = summary['coeff_p'][feature]
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{feature}: {coef:.4f} (p={p_val:.4f}) {significance}")
    
    # Multicollinearity check
    print("\nMulticollinearity Analysis:")
    print("-" * 40)
    if X.shape[1] > 1:
        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data["Feature"] = feature_names or [f"X{i}" for i in range(X.shape[1])]
        vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        print(vif_data.to_string(index=False))
        
        # VIF interpretation
        high_vif = vif_data[vif_data["VIF"] > 5]
        if not high_vif.empty:
            print(f"\n⚠️  Warning: High VIF detected (>5) for: {', '.join(high_vif['Feature'])}")
            print("Consider removing one of the correlated features.")
    
    return model, summary

# Example usage
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 4)
# Create realistic relationships
y = 2*X[:, 0] + 1.5*X[:, 1] - 0.5*X[:, 2] + 0.3*X[:, 3] + np.random.randn(n_samples) * 0.5

feature_names = ['Square_Feet', 'Bedrooms', 'Age', 'Location_Score']
model, summary = comprehensive_analysis(X, y, feature_names)
```

---

## 📊 Model Evaluation

### Key Metrics for Multiple Linear Regression

#### 1. **R² and Adjusted R²**
```python
def calculate_adjusted_r2(r2, n, p):
    """Calculate adjusted R-squared."""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Why use Adjusted R²?
# - R² always increases when adding features
# - Adjusted R² penalizes unnecessary features
# - Better for model comparison
```

#### 2. **F-Statistic**
```python
def f_statistic(y_true, y_pred, n_features):
    """Calculate F-statistic for overall model significance."""
    n = len(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_reg = ss_tot - ss_res
    
    msr = ss_reg / n_features  # Mean Square Regression
    mse = ss_res / (n - n_features - 1)  # Mean Square Error
    
    f_stat = msr / mse
    p_value = 1 - stats.f.cdf(f_stat, n_features, n - n_features - 1)
    
    return f_stat, p_value
```

#### 3. **Individual Coefficient Tests**
```python
def coefficient_significance_test(X, y, coefficients, alpha=0.05):
    """Test significance of individual coefficients."""
    n, p = X.shape
    
    # Add intercept column
    X_design = np.column_stack([np.ones(n), X])
    
    # Calculate residuals
    y_pred = X_design @ np.concatenate([[intercept], coefficients])
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (n - p - 1)
    
    # Variance-covariance matrix
    var_covar = mse * np.linalg.inv(X_design.T @ X_design)
    std_errors = np.sqrt(np.diag(var_covar))
    
    # t-statistics
    all_coeffs = np.concatenate([[intercept], coefficients])
    t_stats = all_coeffs / std_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
    
    # Confidence intervals
    t_critical = stats.t.ppf(1 - alpha/2, n - p - 1)
    ci_lower = all_coeffs - t_critical * std_errors
    ci_upper = all_coeffs + t_critical * std_errors
    
    return {
        'coefficients': all_coeffs,
        'std_errors': std_errors,
        't_statistics': t_stats,
        'p_values': p_values,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }
```

---

## 🔍 Feature Selection

### 1. **Forward Selection**
```python
def forward_selection(X, y, significance_level=0.05):
    """Forward stepwise feature selection."""
    initial_features = []
    best_features = []
    
    features = list(range(X.shape[1]))
    
    while len(features) > 0:
        remaining_features = [f for f in features if f not in best_features]
        
        best_pval = 1
        best_feature = None
        
        for feature in remaining_features:
            test_features = best_features + [feature]
            model = LinearRegression()
            model.fit(X[:, test_features], y)
            
            # Calculate p-value for the new feature
            # (Simplified - in practice, use proper statistical test)
            score = model.score(X[:, test_features], y)
            
            if score > best_pval:  # Simplified selection criterion
                best_pval = score
                best_feature = feature
        
        if best_pval > significance_level:
            best_features.append(best_feature)
        else:
            break
    
    return best_features
```

### 2. **Backward Elimination**
```python
def backward_elimination(X, y, significance_level=0.05):
    """Backward stepwise feature elimination."""
    features = list(range(X.shape[1]))
    
    while len(features) > 0:
        # Fit model with current features
        model = LinearRegression()
        model.fit(X[:, features], y)
        
        # Calculate p-values for all features
        # (Simplified implementation)
        worst_pval = 0
        worst_feature = None
        
        for i, feature in enumerate(features):
            # Remove feature and compare models
            temp_features = [f for f in features if f != feature]
            if len(temp_features) == 0:
                break
                
            temp_model = LinearRegression()
            temp_model.fit(X[:, temp_features], y)
            
            # Calculate significance (simplified)
            score_diff = model.score(X[:, features], y) - temp_model.score(X[:, temp_features], y)
            
            if score_diff < worst_pval:
                worst_pval = score_diff
                worst_feature = feature
        
        if worst_pval < significance_level and worst_feature is not None:
            features.remove(worst_feature)
        else:
            break
    
    return features
```

### 3. **Regularization-based Selection**
```python
from sklearn.linear_model import Lasso, Ridge
from sklearn.feature_selection import SelectFromModel

def lasso_feature_selection(X, y, alpha=0.01):
    """Use Lasso regression for feature selection."""
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    
    # Select features with non-zero coefficients
    selected_features = np.where(lasso.coef_ != 0)[0]
    
    return selected_features, lasso.coef_[selected_features]

def automated_feature_selection(X, y, method='lasso'):
    """Automated feature selection using sklearn."""
    if method == 'lasso':
        selector = SelectFromModel(Lasso(alpha=0.01))
    elif method == 'ridge':
        selector = SelectFromModel(Ridge(alpha=1.0))
    else:
        raise ValueError("Method must be 'lasso' or 'ridge'")
    
    selector.fit(X, y)
    selected_features = selector.get_support(indices=True)
    
    return selected_features
```

---

## ✅ Pros and Cons

<div align="center">

| ✅ **Advantages** | ❌ **Disadvantages** |
|-------------------|---------------------|
| **Handles Multiple Predictors** | **Assumes Linear Relationships** |
| Can model complex relationships | Cannot capture non-linear patterns |
| **Highly Interpretable** | **Sensitive to Multicollinearity** |
| Clear coefficient interpretation | Correlated features cause problems |
| **Fast Training and Prediction** | **Sensitive to Outliers** |
| Computationally efficient | Extreme values affect all coefficients |
| **Statistical Inference** | **Requires Large Sample Size** |
| Confidence intervals, p-values | Need n >> p for stable estimates |
| **Feature Importance** | **Assumption-Heavy** |
| Shows relative feature importance | Many assumptions must be met |

</div>

### When to Use Multiple Linear Regression:

✅ **Good Choice When:**
- Relationships appear linear
- You need interpretable results
- You have multiple relevant predictors
- Sample size is adequate (n > 10×p)
- Features are not highly correlated
- You need statistical inference

❌ **Avoid When:**
- Relationships are clearly non-linear
- Severe multicollinearity exists
- Many irrelevant features present
- Sample size is too small
- Assumptions are severely violated
- You need the highest possible accuracy regardless of interpretability

---

## 🌍 Real-World Examples

### Example 1: House Price Prediction
```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Comprehensive house price dataset
np.random.seed(42)
n_houses = 1000

# Generate realistic house data
size = np.random.normal(2000, 500, n_houses)
bedrooms = np.random.poisson(3, n_houses) + 1
bathrooms = bedrooms * 0.75 + np.random.normal(0, 0.5, n_houses)
age = np.random.exponential(15, n_houses)
lot_size = np.random.normal(8000, 2000, n_houses)

# Create price with realistic relationships
price = (100 * size +           # $100 per sqft
         15000 * bedrooms +      # $15k per bedroom
         20000 * bathrooms +     # $20k per bathroom
         -2000 * age +           # Depreciation
         10 * lot_size +         # $10 per sqft of lot
         np.random.normal(0, 20000, n_houses)  # Noise
        )

# Create DataFrame
house_data = pd.DataFrame({
    'size_sqft': size,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age_years': age,
    'lot_size': lot_size,
    'price': price
})

# Clean data
house_data = house_data[
    (house_data['size_sqft'] > 500) & 
    (house_data['size_sqft'] < 5000) &
    (house_data['bedrooms'] <= 6) &
    (house_data['bathrooms'] > 0) &
    (house_data['age_years'] < 50) &
    (house_data['price'] > 0)
]

print("House Price Prediction Model")
print("=" * 40)

# Prepare features
X = house_data[['size_sqft', 'bedrooms', 'bathrooms', 'age_years', 'lot_size']]
y = house_data['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for better interpretation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)  # Use original scale for interpretability

# Predictions
y_pred = model.predict(X_test)

# Model performance
r2 = model.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Model Performance:")
print(f"R² Score: {r2:.3f}")
print(f"RMSE: ${rmse:,.0f}")

# Feature interpretation
print(f"\nModel Equation:")
print(f"Price = ${model.intercept_:,.0f}")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  + ${coef:,.0f} × {feature}")

print(f"\nInterpretation:")
print(f"• Each additional sqft increases price by ${model.coef_[0]:,.0f}")
print(f"• Each additional bedroom increases price by ${model.coef_[1]:,.0f}")
print(f"• Each additional bathroom increases price by ${model.coef_[2]:,.0f}")
print(f"• Each year of age decreases price by ${-model.coef_[3]:,.0f}")
print(f"• Each additional sqft of lot increases price by ${model.coef_[4]:.0f}")

# Prediction example
new_house = pd.DataFrame({
    'size_sqft': [2200],
    'bedrooms': [4],
    'bathrooms': [3],
    'age_years': [5],
    'lot_size': [9000]
})

predicted_price = model.predict(new_house)[0]
print(f"\nPrediction for new house:")
print(f"Size: {new_house.iloc[0]['size_sqft']} sqft, {new_house.iloc[0]['bedrooms']} bed, {new_house.iloc[0]['bathrooms']} bath")
print(f"Age: {new_house.iloc[0]['age_years']} years, Lot: {new_house.iloc[0]['lot_size']} sqft")
print(f"Predicted Price: ${predicted_price:,.0f}")
```

### Example 2: Student Performance Analysis
```python
# Student performance prediction
np.random.seed(123)
n_students = 500

# Generate student data
study_hours = np.random.gamma(2, 2, n_students)  # Study hours per week
sleep_hours = np.random.normal(7, 1.5, n_students)  # Sleep hours per night
attendance = np.random.beta(5, 1, n_students) * 100  # Attendance percentage
prev_gpa = np.random.normal(3.0, 0.5, n_students)  # Previous GPA

# Calculate current GPA with realistic relationships
current_gpa = (0.1 * study_hours +      # More study = better grades
               0.15 * sleep_hours +      # Good sleep helps
               0.02 * attendance +       # Attendance matters
               0.6 * prev_gpa +          # Previous performance predicts current
               np.random.normal(0, 0.3, n_students)  # Random factors
              )

# Ensure GPA is in valid range
current_gpa = np.clip(current_gpa, 0, 4.0)

student_data = pd.DataFrame({
    'study_hours_weekly': study_hours,
    'sleep_hours_nightly': sleep_hours,
    'attendance_percent': attendance,
    'previous_gpa': prev_gpa,
    'current_gpa': current_gpa
})

print("Student Performance Analysis")
print("=" * 40)

# Model fitting
X = student_data[['study_hours_weekly', 'sleep_hours_nightly', 'attendance_percent', 'previous_gpa']]
y = student_data['current_gpa']

model = LinearRegression()
model.fit(X, y)

# Results
print(f"Model Equation:")
print(f"GPA = {model.intercept_:.3f}")
for feature, coef in zip(X.columns, model.coef_):
    print(f"    + {coef:.3f} × {feature}")

print(f"\nR² Score: {model.score(X, y):.3f}")

# Actionable insights
print(f"\nActionable Insights for Students:")
print(f"• Studying 1 additional hour per week increases GPA by {model.coef_[0]:.3f}")
print(f"• Getting 1 additional hour of sleep increases GPA by {model.coef_[1]:.3f}")
print(f"• Improving attendance by 10% increases GPA by {model.coef_[2]*10:.3f}")
print(f"• Previous GPA is a strong predictor (coefficient: {model.coef_[3]:.3f})")

# Student improvement scenarios
print(f"\nImprovement Scenarios:")
baseline_student = [10, 6, 80, 2.5]  # Low-performing student
improved_student_study = [15, 6, 80, 2.5]  # +5 hours study
improved_student_sleep = [10, 8, 80, 2.5]  # +2 hours sleep
improved_student_attend = [10, 6, 95, 2.5]  # +15% attendance

scenarios = [
    ("Baseline", baseline_student),
    ("+ 5 hours study", improved_student_study),
    ("+ 2 hours sleep", improved_student_sleep),
    ("+ 15% attendance", improved_student_attend)
]

for name, scenario in scenarios:
    predicted_gpa = model.predict([scenario])[0]
    improvement = predicted_gpa - model.predict([baseline_student])[0]
    print(f"{name:20}: GPA = {predicted_gpa:.2f} (Δ = +{improvement:.2f})")
```

### Example 3: Marketing ROI Analysis
```python
# Marketing campaign effectiveness
np.random.seed(456)
n_campaigns = 200

# Marketing spend across different channels
tv_spend = np.random.exponential(50000, n_campaigns)
online_spend = np.random.exponential(30000, n_campaigns)
print_spend = np.random.exponential(20000, n_campaigns)
radio_spend = np.random.exponential(15000, n_campaigns)

# Sales with diminishing returns and interaction effects
sales = (2.5 * np.sqrt(tv_spend) +          # TV has diminishing returns
         4.0 * np.sqrt(online_spend) +      # Online is more efficient
         1.5 * np.sqrt(print_spend) +       # Print is less effective
         2.0 * np.sqrt(radio_spend) +       # Radio is moderate
         0.0001 * tv_spend * online_spend + # Synergy between TV and online
         np.random.normal(0, 50000, n_campaigns)  # Market noise
        )

marketing_data = pd.DataFrame({
    'tv_spend': tv_spend,
    'online_spend': online_spend,
    'print_spend': print_spend,
    'radio_spend': radio_spend,
    'sales': sales
})

print("Marketing ROI Analysis")
print("=" * 40)

# Linear model (note: this assumes linear relationships, which we violated above)
X = marketing_data[['tv_spend', 'online_spend', 'print_spend', 'radio_spend']]
y = marketing_data['sales']

model = LinearRegression()
model.fit(X, y)

print(f"Linear Model Results:")
print(f"R² Score: {model.score(X, y):.3f}")

# ROI calculation
print(f"\nROI per Dollar Spent (Linear Model):")
channels = ['TV', 'Online', 'Print', 'Radio']
for channel, coef in zip(channels, model.coef_):
    roi = coef - 1  # Subtract the $1 investment
    print(f"{channel:10}: ${coef:.2f} sales per $1 spent (ROI: {roi*100:.1f}%)")

# Budget optimization
total_budget = 200000
print(f"\nBudget Optimization for ${total_budget:,}:")

# Simple optimization: allocate based on ROI
roi_values = model.coef_
total_roi = np.sum(roi_values)
optimal_allocation = (roi_values / total_roi) * total_budget

for channel, allocation in zip(channels, optimal_allocation):
    print(f"{channel:10}: ${allocation:,.0f} ({allocation/total_budget*100:.1f}%)")

# Predicted sales with optimal allocation
predicted_sales = model.predict([optimal_allocation])[0]
print(f"\nPredicted Sales with Optimal Allocation: ${predicted_sales:,.0f}")
```

---

## 🔬 Advanced Topics

### 1. **Regularization**

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

def compare_regularization_methods(X, y, test_size=0.2):
    """Compare different regularization techniques."""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Standardize features (important for regularization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Ridge (α=10.0)': Ridge(alpha=10.0),
        'Lasso (α=0.1)': Lasso(alpha=0.1),
        'Lasso (α=1.0)': Lasso(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5)
    }
    
    results = {}
    
    for name, model in models.items():
        # Fit model
        model.fit(X_train_scaled, y_train)
        
        # Calculate scores
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        # Count non-zero coefficients (for sparsity)
        if hasattr(model, 'coef_'):
            non_zero_coefs = np.sum(np.abs(model.coef_) > 1e-5)
        else:
            non_zero_coefs = X.shape[1]
        
        results[name] = {
            'train_r2': train_score,
            'test_r2': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'non_zero_coefs': non_zero_coefs,
            'coefficients': model.coef_ if hasattr(model, 'coef_') else None
        }
    
    # Display results
    print("Regularization Comparison")
    print("=" * 80)
    print(f"{'Model':<20} {'Train R²':<10} {'Test R²':<10} {'CV Mean':<10} {'CV Std':<10} {'Features':<10}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<20} {result['train_r2']:<10.3f} {result['test_r2']:<10.3f} "
              f"{result['cv_mean']:<10.3f} {result['cv_std']:<10.3f} {result['non_zero_coefs']:<10}")
    
    return results

# Example usage with multicollinear data
np.random.seed(42)
n, p = 100, 20
X = np.random.randn(n, p)

# Create multicollinearity
X[:, 1] = X[:, 0] + 0.1 * np.random.randn(n)  # X1 ≈ X0
X[:, 2] = 2 * X[:, 0] + 0.1 * np.random.randn(n)  # X2 ≈ 2*X0

# Generate target with only first 5 features being relevant
true_coefs = np.zeros(p)
true_coefs[:5] = [2, -1.5, 1, 0.5, -0.8]
y = X @ true_coefs + 0.1 * np.random.randn(n)

results = compare_regularization_methods(X, y)
```

### 2. **Cross-Validation and Model Selection**

```python
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline

def comprehensive_model_selection(X, y, feature_names=None):
    """Comprehensive model selection with cross-validation."""
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    # Define parameter grids for different models
    param_grids = [
        {
            'regressor': [LinearRegression()],
        },
        {
            'regressor': [Ridge()],
            'regressor__alpha': [0.1, 1.0, 10.0, 100.0]
        },
        {
            'regressor': [Lasso()],
            'regressor__alpha': [0.01, 0.1, 1.0, 10.0]
        },
        {
            'regressor': [ElasticNet()],
            'regressor__alpha': [0.1, 1.0, 10.0],
            'regressor__l1_ratio': [0.3, 0.5, 0.7]
        }
    ]
    
    # Cross-validation strategy
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    best_score = -np.inf
    best_model = None
    all_results = []
    
    for param_grid in param_grids:
        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, 
            scoring='r2', n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X, y)
        
        # Store results
        result = {
            'model_type': param_grid['regressor'][0].__class__.__name__,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
        all_results.append(result)
        
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
    
    # Display results
    print("Model Selection Results")
    print("=" * 50)
    for result in all_results:
        print(f"{result['model_type']:<15}: R² = {result['best_score']:.4f}")
        print(f"                 Params: {result['best_params']}")
        print()
    
    print(f"Best Model: {best_model.named_steps['regressor'].__class__.__name__}")
    print(f"Best CV Score: {best_score:.4f}")
    
    # Feature importance analysis for best model
    if hasattr(best_model.named_steps['regressor'], 'coef_'):
        coefficients = best_model.named_steps['regressor'].coef_
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(coefficients))]
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        print("\nFeature Importance (Best Model):")
        print("-" * 40)
        print(importance_df.to_string(index=False))
    
    return best_model, all_results

# Example usage
feature_names = ['sqft', 'bedrooms', 'bathrooms', 'age', 'lot_size']
best_model, results = comprehensive_model_selection(X, y, feature_names)
```

### 3. **Outlier Detection and Treatment**

```python
from scipy import stats
from sklearn.preprocessing import RobustScaler

def detect_outliers_multiple_methods(X, y, feature_names=None):
    """Detect outliers using multiple methods."""
    
    n, p = X.shape
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(p)]
    
    outlier_indices = set()
    
    # Method 1: Z-score for each feature
    z_scores = np.abs(stats.zscore(X, axis=0))
    z_outliers = np.where(z_scores > 3)
    outlier_indices.update(z_outliers[0])
    
    # Method 2: IQR method for each feature
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    iqr_outliers = np.where((X < lower_bound) | (X > upper_bound))
    outlier_indices.update(iqr_outliers[0])
    
    # Method 3: Mahalanobis distance
    try:
        cov_matrix = np.cov(X.T)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        mean = np.mean(X, axis=0)
        
        mahal_distances = []
        for i in range(n):
            diff = X[i] - mean
            mahal_dist = np.sqrt(diff.T @ inv_cov_matrix @ diff)
            mahal_distances.append(mahal_dist)
        
        mahal_distances = np.array(mahal_distances)
        threshold = np.percentile(mahal_distances, 95)  # Top 5% as outliers
        mahal_outliers = np.where(mahal_distances > threshold)[0]
        outlier_indices.update(mahal_outliers)
        
    except np.linalg.LinAlgError:
        print("Warning: Could not compute Mahalanobis distance (singular covariance matrix)")
        mahal_distances = None
    
    # Method 4: Cook's distance (requires fitted model)
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate leverage
    X_with_intercept = np.column_stack([np.ones(n), X])
    H = X_with_intercept @ np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
    leverage = np.diag(H)
    
    # Calculate Cook's distance
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = np.mean(residuals**2)
    
    cooks_d = (residuals**2 / ((p + 1) * mse)) * (leverage / (1 - leverage)**2)
    cook_threshold = 4 / n
    cook_outliers = np.where(cooks_d > cook_threshold)[0]
    outlier_indices.update(cook_outliers)
    
    # Summary
    outlier_list = sorted(list(outlier_indices))
    
    print("Outlier Detection Summary")
    print("=" * 40)
    print(f"Total observations: {n}")
    print(f"Outliers detected: {len(outlier_list)} ({len(outlier_list)/n*100:.1f}%)")
    print(f"Outlier indices: {outlier_list[:10]}{'...' if len(outlier_list) > 10 else ''}")
    
    return outlier_list, {
        'z_scores': z_scores,
        'mahalanobis': mahal_distances,
        'cooks_distance': cooks_d,
        'leverage': leverage
    }

def handle_outliers_comparison(X, y, outlier_indices, feature_names=None):
    """Compare model performance with different outlier handling strategies."""
    
    strategies = {
        'No Treatment': (X, y),
        'Remove Outliers': (np.delete(X, outlier_indices, axis=0), np.delete(y, outlier_indices)),
        'Robust Scaling': (RobustScaler().fit_transform(X), y),
    }
    
    # Winsorization (cap extreme values)
    X_winsorized = X.copy()
    for i in range(X.shape[1]):
        lower_percentile = np.percentile(X[:, i], 5)
        upper_percentile = np.percentile(X[:, i], 95)
        X_winsorized[:, i] = np.clip(X_winsorized[:, i], lower_percentile, upper_percentile)
    strategies['Winsorization'] = (X_winsorized, y)
    
    print("\nOutlier Handling Comparison")
    print("=" * 50)
    print(f"{'Strategy':<20} {'R²':<10} {'RMSE':<15} {'N Obs':<10}")
    print("-" * 50)
    
    for strategy_name, (X_strategy, y_strategy) in strategies.items():
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_strategy, y_strategy, test_size=0.2, random_state=42
        )
        
        # Fit model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate
        r2 = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"{strategy_name:<20} {r2:<10.3f} {rmse:<15.1f} {len(X_strategy):<10}")

# Example usage
outliers, outlier_stats = detect_outliers_multiple_methods(X, y, feature_names)
handle_outliers_comparison(X, y, outliers, feature_names)
```

---

## ❓ FAQ

### Q1: How many observations do I need for reliable results?

**A:** The general rule is **n ≥ 10-20 times the number of predictors (p)**:
- **Minimum**: n ≥ 10p (basic stability)
- **Recommended**: n ≥ 20p (good statistical power)
- **Ideal**: n ≥ 50p (robust results)

For example, with 5 predictors, you should have at least 50-100 observations.

### Q2: How do I handle multicollinearity?

**A:** Several strategies:
1. **Remove highly correlated features** (correlation > 0.8)
2. **Use VIF** (Variance Inflation Factor) - remove features with VIF > 5-10
3. **Principal Component Analysis** (PCA) for dimensionality reduction
4. **Regularization** (Ridge, Lasso) to handle correlated features
5. **Domain knowledge** to choose the most meaningful features

### Q3: What if my residuals aren't normal?

**A:** Options include:
1. **Transform the target variable** (log, square root, Box-Cox)
2. **Transform predictor variables** if they have skewed distributions
3. **Use robust regression** methods
4. **Try non-linear models** if the relationship isn't linear
5. **Bootstrap methods** for inference without normality assumption

### Q4: How do I interpret coefficients when features are on different scales?

**A:** Use **standardized coefficients**:
```python
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit model on scaled data
model = LinearRegression()
model.fit(X_scaled, y)

# Standardized coefficients show relative importance
print("Standardized coefficients:", model.coef_)
```

### Q5: Should I include interaction terms?

**A:** Consider interactions when:
- **Domain knowledge** suggests features interact
- **Exploratory analysis** shows interaction patterns
- **Cross-validation** shows improvement with interactions

```python
from sklearn.preprocessing import PolynomialFeatures

# Add interaction terms
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_with_interactions = poly.fit_transform(X)
```

### Q6: How do I choose between Ridge and Lasso?

**A:** 
- **Ridge**: When all features are somewhat relevant, handles multicollinearity well
- **Lasso**: When you want feature selection, creates sparse models
- **ElasticNet**: Combines both, good compromise for most situations

---

## 📚 Additional Resources

### Books:
- **"Applied Linear Statistical Models"** by Kutner, Nachtsheim, Neter & Li
- **"Regression Analysis by Example"** by Chatterjee & Hadi
- **"Introduction to Linear Regression Analysis"** by Montgomery, Peck & Vining

### Online Courses:
- [Stanford CS229 Machine Learning](http://cs229.stanford.edu/)
- [MIT 18.650 Statistics for Applications](https://ocw.mit.edu/courses/mathematics/18-650-statistics-for-applications-fall-2016/)
- [Coursera: Regression Models by Johns Hopkins](https://www.coursera.org/learn/regression-models)

### Python Libraries:
- **scikit-learn**: `sklearn.linear_model`
- **statsmodels**: Detailed statistical output and diagnostics
- **scipy.stats**: Statistical tests and distributions
- **seaborn**: Statistical visualization

---

## 🏗️ Project Structure

```
Multiple-Linear-Regression/
│
├── README.md                          # This comprehensive guide
├── multiple_linear_regression.py      # Implementation from scratch
├── Startup.csv              # Adding dataset for the prblem
|
```

---

<div align="center">

## 🌟 Key Takeaways

**Multiple Linear Regression:**
- Extends simple regression to multiple predictors
- Assumes linear relationships and independent features
- Highly interpretable with statistical inference
- Sensitive to multicollinearity and outliers

**Best Practices:**
- Always check assumptions first
- Handle multicollinearity proactively
- Use cross-validation for model selection
- Consider regularization for stability
- Interpret coefficients carefully

---

### 📖 Master Multiple Predictors with Confidence! 🚀

*Created by [@danialasim](https://github.com/danialasim) | Last updated: July 1, 2025*

</div>