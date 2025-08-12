# 🔍 Principal Component Analysis (PCA)

<div align="center">

![Method](https://img.shields.io/badge/Method-Dimensionality_Reduction-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-green?style=for-the-badge)
![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-yellow?style=for-the-badge)

*A Comprehensive Guide to Principal Component Analysis for Dimensionality Reduction*

</div>

---

## 📚 Table of Contents

- [Introduction to PCA](#introduction-to-pca)
- [Mathematical Foundation](#mathematical-foundation)
- [Implementation Guide](#implementation-guide)
- [Choosing the Number of Components](#choosing-the-number-of-components)
- [Practical Applications](#practical-applications)
- [Variants of PCA](#variants-of-pca)
- [Comparison with Other Methods](#comparison-with-other-methods)
- [Limitations and Considerations](#limitations-and-considerations)
- [Best Practices](#best-practices)
- [FAQ](#faq)

---

## 🎯 Introduction to PCA

Principal Component Analysis (PCA) is a widely used unsupervised statistical technique for dimensionality reduction. It transforms a high-dimensional dataset into a lower-dimensional subspace while preserving as much variance as possible. This transformation helps with visualization, noise reduction, feature extraction, and addressing the curse of dimensionality.

### Key Concepts:

- **Dimensionality Reduction**: Reducing the number of features while retaining important information
- **Principal Components**: New variables that are linear combinations of the original features
- **Variance Preservation**: Maximizing the variance retained in the reduced dimensions
- **Orthogonality**: Principal components are mutually perpendicular in the feature space
- **Data Compression**: Representing data more efficiently with fewer dimensions
- **Feature Extraction**: Creating new features that better capture data characteristics

### Why PCA is Important:

1. **Visualization**: Helps visualize high-dimensional data in 2D or 3D
2. **Computational Efficiency**: Reduces computational requirements for subsequent analyses
3. **Noise Reduction**: Eliminates dimensions with low variance that often represent noise
4. **Multicollinearity**: Addresses issues of correlated features in statistical models
5. **Preprocessing**: Serves as an effective preprocessing step for machine learning algorithms
6. **Data Compression**: Enables more efficient storage and transmission of data

### Brief History:

- **1901**: Karl Pearson introduced the technique that would later become PCA
- **1933**: Harold Hotelling independently developed the method, calling it "Principal Component Analysis"
- **1960s-1970s**: With the advent of computers, PCA became widely applicable to larger datasets
- **1980s-Present**: Extensions and variants of PCA have been developed for specific applications

---

## 🧮 Mathematical Foundation

### The Basics of PCA

Principal Component Analysis finds a new coordinate system (principal components) for the data such that:
1. The greatest variance comes to lie on the first coordinate (first principal component)
2. The second greatest variance on the second coordinate, and so on
3. All coordinates are orthogonal to each other

### Step-by-Step Mathematical Process

#### 1. Standardization

First, standardize the data to have zero mean and unit variance:

$$X_{std} = \frac{X - \mu}{\sigma}$$

Where:
- $X$ is the original data matrix
- $\mu$ is the mean of each feature
- $\sigma$ is the standard deviation of each feature

#### 2. Covariance Matrix Calculation

Calculate the covariance matrix of the standardized data:

$$\Sigma = \frac{1}{n-1} X_{std}^T X_{std}$$

Where:
- $\Sigma$ is the covariance matrix
- $n$ is the number of samples
- $X_{std}$ is the standardized data matrix

#### 3. Eigendecomposition

Compute the eigenvalues and eigenvectors of the covariance matrix:

$$\Sigma v = \lambda v$$

Where:
- $v$ is an eigenvector of $\Sigma$
- $\lambda$ is the corresponding eigenvalue

#### 4. Sorting Eigenvectors

Sort the eigenvectors by their corresponding eigenvalues in descending order, creating a matrix $W$ where columns are the sorted eigenvectors.

#### 5. Projection

Project the data onto the new basis formed by the top $k$ eigenvectors:

$$Y = X_{std} W_k$$

Where:
- $Y$ is the transformed data in the reduced space
- $W_k$ is the matrix of the first $k$ eigenvectors (principal components)

### Variance Explained

The proportion of total variance explained by the $i$-th principal component is:

$$\text{Variance Explained}_i = \frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}$$

Where:
- $\lambda_i$ is the $i$-th eigenvalue
- $d$ is the total number of dimensions in the original dataset

The cumulative variance explained by the first $k$ principal components is:

$$\text{Cumulative Variance Explained}_k = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{j=1}^{d} \lambda_j}$$

### Geometric Interpretation

Geometrically, PCA can be understood as:
1. Shifting the origin of the coordinate system to the center of the data (centering)
2. Rotating the coordinate system to align with the directions of maximum variance
3. Projecting the data onto the subspace spanned by the top principal components

### Alternative Derivation via SVD

PCA can also be derived using Singular Value Decomposition (SVD):

$$X_{std} = U \Sigma V^T$$

Where:
- $U$ is the matrix of left singular vectors
- $\Sigma$ is the diagonal matrix of singular values
- $V$ is the matrix of right singular vectors

The principal components are then given by the right singular vectors $V$, and the projected data can be computed as:

$$Y = X_{std} V_k = U_k \Sigma_k$$

Where:
- $V_k$ contains the first $k$ columns of $V$
- $U_k$ contains the first $k$ columns of $U$
- $\Sigma_k$ is the upper-left $k \times k$ submatrix of $\Sigma$

---

## 💻 Implementation Guide

### Implementation with Python

#### Using scikit-learn

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Sample data
X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)  # Keep all components for this example
X_pca = pca.fit_transform(X_scaled)

# Get the principal components (eigenvectors)
components = pca.components_

# Get the eigenvalues (variance explained by each component)
explained_variance = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_

print("Principal Components (Eigenvectors):")
print(components)
print("\nEigenvalues (Variance along each PC):")
print(explained_variance)
print("\nExplained Variance Ratio:")
print(explained_variance_ratio)
print(f"Total Variance Explained: {np.sum(explained_variance_ratio) * 100:.2f}%")

# Visualize the principal components
plt.figure(figsize=(10, 8))

# Plot original data
plt.subplot(2, 1, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.8)
plt.title('Original Data')
plt.grid(True)

# Calculate the center of the data
center = np.mean(X_scaled, axis=0)

# Plot scaled data with principal components
plt.subplot(2, 1, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.8)

# Plot the principal components
for i, (comp, var) in enumerate(zip(components, explained_variance)):
    comp_line = np.vstack([center, center + comp * np.sqrt(var)])
    plt.plot(comp_line[:, 0], comp_line[:, 1], 'r-', 
             label=f'Principal Component {i+1}')
    plt.arrow(center[0], center[1], comp[0] * np.sqrt(var), comp[1] * np.sqrt(var),
              head_width=0.1, head_length=0.1, fc='red', ec='red')

plt.title('Standardized Data with Principal Components')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Visualize the transformed data
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.8)
plt.title('Data Transformed to Principal Component Space')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.grid(True)
plt.show()
```

#### Custom Implementation from Scratch

```python
def pca_from_scratch(X, n_components=None):
    """
    Perform Principal Component Analysis from scratch.
    
    Parameters:
    -----------
    X : numpy array
        Input data matrix (n_samples, n_features)
    n_components : int, optional
        Number of principal components to keep
        
    Returns:
    --------
    X_pca : numpy array
        Transformed data (n_samples, n_components)
    components : numpy array
        Principal components (n_components, n_features)
    explained_variance : numpy array
        Variance explained by each component
    explained_variance_ratio : numpy array
        Ratio of variance explained to total variance
    """
    # Standardize the data
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Compute the covariance matrix
    cov_matrix = np.cov(X_std, rowvar=False)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select number of components
    if n_components is None:
        n_components = X.shape[1]
    
    # Select top n_components eigenvectors
    components = eigenvectors[:, :n_components]
    
    # Calculate explained variance and ratio
    explained_variance = eigenvalues[:n_components]
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = explained_variance / total_variance
    
    # Project data onto principal components
    X_pca = np.dot(X_std, components)
    
    return X_pca, components, explained_variance, explained_variance_ratio

# Apply the custom PCA function
X_pca_custom, components_custom, variance_custom, variance_ratio_custom = pca_from_scratch(X)

print("\nCustom PCA Implementation Results:")
print("Principal Components:")
print(components_custom)
print("\nExplained Variance:")
print(variance_custom)
print("\nExplained Variance Ratio:")
print(variance_ratio_custom)
print(f"Total Variance Explained: {np.sum(variance_ratio_custom) * 100:.2f}%")
```

#### Implementing PCA with SVD

```python
def pca_with_svd(X, n_components=None):
    """
    Perform PCA using Singular Value Decomposition.
    
    Parameters:
    -----------
    X : numpy array
        Input data matrix (n_samples, n_features)
    n_components : int, optional
        Number of principal components to keep
        
    Returns:
    --------
    X_pca : numpy array
        Transformed data (n_samples, n_components)
    components : numpy array
        Principal components (n_components, n_features)
    explained_variance : numpy array
        Variance explained by each component
    explained_variance_ratio : numpy array
        Ratio of variance explained to total variance
    """
    # Standardize the data
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Perform SVD
    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    
    # Select number of components
    if n_components is None:
        n_components = X.shape[1]
    
    # Calculate variance explained
    explained_variance = (S ** 2) / (X.shape[0] - 1)
    total_variance = np.sum(explained_variance)
    explained_variance_ratio = explained_variance / total_variance
    
    # Select top n_components
    components = Vt[:n_components, :]
    explained_variance = explained_variance[:n_components]
    explained_variance_ratio = explained_variance_ratio[:n_components]
    
    # Project data onto principal components
    X_pca = np.dot(X_std, components.T)
    
    return X_pca, components, explained_variance, explained_variance_ratio

# Apply PCA with SVD
X_pca_svd, components_svd, variance_svd, variance_ratio_svd = pca_with_svd(X)

print("\nPCA with SVD Results:")
print("Principal Components:")
print(components_svd)
print("\nExplained Variance:")
print(variance_svd)
print("\nExplained Variance Ratio:")
print(variance_ratio_svd)
print(f"Total Variance Explained: {np.sum(variance_ratio_svd) * 100:.2f}%")
```

### Implementation in R

```r
# Load necessary libraries
library(ggplot2)
library(dplyr)

# Sample data
X <- matrix(c(
  2.5, 2.4,
  0.5, 0.7,
  2.2, 2.9,
  1.9, 2.2,
  3.1, 3.0,
  2.3, 2.7,
  2.0, 1.6,
  1.0, 1.1,
  1.5, 1.6,
  1.1, 0.9
), ncol = 2, byrow = TRUE)

# Convert to data frame for ggplot
df <- data.frame(X)
colnames(df) <- c("X1", "X2")

# Perform PCA using prcomp
pca_result <- prcomp(X, scale. = TRUE)

# Extract results
components <- pca_result$rotation
scores <- pca_result$x
explained_variance <- pca_result$sdev^2
explained_variance_ratio <- explained_variance / sum(explained_variance)

# Print results
cat("Principal Components (Eigenvectors):\n")
print(components)
cat("\nExplained Variance:\n")
print(explained_variance)
cat("\nExplained Variance Ratio:\n")
print(explained_variance_ratio)
cat(sprintf("\nTotal Variance Explained: %.2f%%\n", sum(explained_variance_ratio) * 100))

# Visualize the data and principal components
p1 <- ggplot(df, aes(x = X1, y = X2)) +
  geom_point(size = 3, alpha = 0.8) +
  labs(title = "Original Data") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Create a data frame for scaled data
X_scaled <- scale(X)
df_scaled <- data.frame(X_scaled)
colnames(df_scaled) <- c("X1", "X2")

# Calculate the center
center <- colMeans(X_scaled)

# Create data frame for plotting principal components
pc_df <- data.frame(
  x1 = c(center[1], center[1] + components[1, 1] * sqrt(explained_variance[1])),
  y1 = c(center[2], center[2] + components[2, 1] * sqrt(explained_variance[1])),
  x2 = c(center[1], center[1] + components[1, 2] * sqrt(explained_variance[2])),
  y2 = c(center[2], center[2] + components[2, 2] * sqrt(explained_variance[2]))
)

p2 <- ggplot(df_scaled, aes(x = X1, y = X2)) +
  geom_point(size = 3, alpha = 0.8) +
  geom_segment(data = pc_df, aes(x = x1[1], y = y1[1], xend = x1[2], yend = y1[2]),
               arrow = arrow(length = unit(0.3, "cm")), color = "red", size = 1) +
  geom_segment(data = pc_df, aes(x = x2[1], y = y2[1], xend = x2[2], yend = y2[2]),
               arrow = arrow(length = unit(0.3, "cm")), color = "blue", size = 1) +
  labs(title = "Standardized Data with Principal Components",
       subtitle = paste("PC1:", round(explained_variance_ratio[1] * 100, 2),
                        "%, PC2:", round(explained_variance_ratio[2] * 100, 2), "%")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

# Create a data frame for transformed data
df_pca <- data.frame(scores)
colnames(df_pca) <- c("PC1", "PC2")

p3 <- ggplot(df_pca, aes(x = PC1, y = PC2)) +
  geom_point(size = 3, alpha = 0.8) +
  labs(title = "Data Transformed to Principal Component Space",
       x = "First Principal Component",
       y = "Second Principal Component") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Print the plots
print(p1)
print(p2)
print(p3)
```

---

## 🔢 Choosing the Number of Components

Determining the optimal number of principal components is a critical step in PCA. Here are several methods to guide this decision:

### 1. Explained Variance Ratio

Retain components that explain a certain percentage of the total variance (e.g., 95%):

```python
def choose_components_by_variance(X, variance_threshold=0.95):
    """Select number of components based on explained variance threshold."""
    pca = PCA()
    pca.fit(X)
    
    # Cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find the number of components that explain at least variance_threshold
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    return n_components, cumulative_variance

# Apply the function
n_components, cumulative_variance = choose_components_by_variance(X_scaled)

# Plot the cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.axvline(x=n_components, color='g', linestyle='--')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.annotate(f'Optimal Components: {n_components}', 
             xy=(n_components, cumulative_variance[n_components-1]),
             xytext=(n_components+0.5, cumulative_variance[n_components-1]-0.1),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
```

### 2. Scree Plot

Visualize the explained variance of each component and look for an "elbow" point:

```python
def plot_scree(pca):
    """Create a scree plot of explained variance."""
    plt.figure(figsize=(10, 6))
    
    # Individual explained variance
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_, alpha=0.7, color='blue')
    plt.step(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), where='mid', color='red', marker='o')
    
    plt.title('Scree Plot of Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True, alpha=0.3)
    plt.legend(['Cumulative Explained Variance', 'Individual Explained Variance'])
    plt.tight_layout()
    plt.show()

# Fit PCA and create scree plot
pca = PCA().fit(X_scaled)
plot_scree(pca)
```

### 3. Kaiser Rule

Keep only components with eigenvalues greater than 1 (for standardized data):

```python
def kaiser_rule(X):
    """Apply Kaiser rule to select components with eigenvalues > 1."""
    pca = PCA()
    pca.fit(X)
    
    # Find components with eigenvalues > 1
    n_components = sum(pca.explained_variance_ > 1)
    
    return n_components, pca.explained_variance_

# Apply Kaiser rule
n_components, eigenvalues = kaiser_rule(X_scaled)

print(f"Number of components selected by Kaiser rule: {n_components}")
print(f"Eigenvalues: {eigenvalues}")

# Visualize eigenvalues
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(eigenvalues) + 1), eigenvalues, alpha=0.7)
plt.axhline(y=1, color='r', linestyle='--')
plt.title('Eigenvalues of Components')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.grid(True, alpha=0.3)
plt.show()
```

### 4. Cross-Validation

Use cross-validation to find the optimal number of components:

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def pca_cv(X, y, max_components=None, cv=5):
    """Find optimal number of components using cross-validation."""
    if max_components is None:
        max_components = X.shape[1]
    
    max_components = min(max_components, X.shape[1])
    
    # Store cross-validation scores
    cv_scores = []
    
    for n in range(1, max_components + 1):
        # Create a pipeline with PCA and a classifier
        pipeline = Pipeline([
            ('pca', PCA(n_components=n)),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
        
        # Perform cross-validation
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        cv_scores.append(np.mean(scores))
    
    # Find the number of components with the highest CV score
    optimal_n = np.argmax(cv_scores) + 1
    
    return optimal_n, cv_scores

# Example with classification dataset
from sklearn.datasets import load_iris
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Find optimal components
optimal_n, cv_scores = pca_cv(X_iris, y_iris)

# Plot cross-validation results
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o')
plt.axvline(x=optimal_n, color='r', linestyle='--')
plt.title('Cross-Validation Accuracy vs. Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Cross-Validation Accuracy')
plt.grid(True)
plt.annotate(f'Optimal Components: {optimal_n}', 
             xy=(optimal_n, cv_scores[optimal_n-1]),
             xytext=(optimal_n+0.5, cv_scores[optimal_n-1]-0.05),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()

print(f"Optimal number of components by cross-validation: {optimal_n}")
```

### 5. Parallel Analysis

Compare eigenvalues to those obtained from random data with the same structure:

```python
def parallel_analysis(X, n_iterations=100, percentile=95):
    """
    Perform parallel analysis to determine number of components.
    
    Parameters:
    -----------
    X : numpy array
        Input data matrix (n_samples, n_features)
    n_iterations : int
        Number of random datasets to generate
    percentile : int
        Percentile of random eigenvalues to compare against
        
    Returns:
    --------
    n_components : int
        Recommended number of components
    """
    n_samples, n_features = X.shape
    
    # Standardize the data
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Calculate eigenvalues of the actual data
    pca = PCA().fit(X_std)
    actual_eigenvalues = pca.explained_variance_
    
    # Generate random data and calculate their eigenvalues
    random_eigenvalues = np.zeros((n_iterations, n_features))
    
    for i in range(n_iterations):
        # Generate random data with the same dimensions
        random_data = np.random.normal(size=(n_samples, n_features))
        random_data = (random_data - np.mean(random_data, axis=0)) / np.std(random_data, axis=0)
        
        # Calculate eigenvalues
        random_pca = PCA().fit(random_data)
        random_eigenvalues[i, :] = random_pca.explained_variance_
    
    # Calculate percentile of random eigenvalues
    random_percentile = np.percentile(random_eigenvalues, percentile, axis=0)
    
    # Compare actual eigenvalues with random eigenvalues
    n_components = sum(actual_eigenvalues > random_percentile)
    
    return n_components, actual_eigenvalues, random_percentile

# Apply parallel analysis
n_components, actual_evals, random_evals = parallel_analysis(X)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(actual_evals) + 1), actual_evals, marker='o', label='Actual Data')
plt.plot(range(1, len(random_evals) + 1), random_evals, marker='x', label='95th Percentile of Random Data')
plt.axvline(x=n_components, color='r', linestyle='--')
plt.title('Parallel Analysis')
plt.xlabel('Component Number')
plt.ylabel('Eigenvalue')
plt.legend()
plt.grid(True)
plt.annotate(f'Components to Retain: {n_components}', 
             xy=(n_components, actual_evals[n_components-1]),
             xytext=(n_components+0.5, actual_evals[n_components-1]+0.5),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()

print(f"Number of components recommended by parallel analysis: {n_components}")
```

---

## 🔬 Practical Applications

PCA has numerous applications across different domains:

### Image Compression

PCA can be used to compress images by reducing the dimensionality of pixel data:

```python
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Load face dataset
faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = faces.data
y = faces.target

# Get image dimensions
h, w = faces.images[0].shape

# Perform PCA with different numbers of components
n_components_list = [10, 50, 100, 200]
fig, axes = plt.subplots(1, len(n_components_list) + 1, figsize=(15, 3))

# Original image
axes[0].imshow(faces.images[0], cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

for i, n_components in enumerate(n_components_list):
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Reconstruct image
    X_reconstructed = pca.inverse_transform(X_pca)
    
    # Reshape and display the reconstructed image
    reconstructed_image = X_reconstructed[0].reshape(h, w)
    axes[i+1].imshow(reconstructed_image, cmap='gray')
    axes[i+1].set_title(f'{n_components} PCs\n({pca.explained_variance_ratio_.sum():.2%})')
    axes[i+1].axis('off')

plt.tight_layout()
plt.show()

# Calculate compression ratio
def calculate_compression_ratio(n_components, original_dim):
    """Calculate compression ratio using PCA."""
    # Original size: n_samples * original_dim
    # Compressed size: n_samples * n_components + n_components * original_dim
    # The second term accounts for storing the principal components
    
    # For a single image:
    original_size = original_dim
    compressed_size = n_components + (n_components * original_dim) / X.shape[0]
    
    return original_size / compressed_size

# Print compression ratios
for n_components in n_components_list:
    ratio = calculate_compression_ratio(n_components, X.shape[1])
    print(f"Compression ratio with {n_components} components: {ratio:.2f}:1")
```

### Noise Reduction

PCA can help reduce noise in data by removing low-variance components:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate a noisy sine wave
np.random.seed(42)
t = np.linspace(0, 10, 100)
clean_signal = np.sin(t)
noise = np.random.normal(0, 0.5, size=len(t))
noisy_signal = clean_signal + noise

# Create a dataset with time-lagged features
def create_time_lagged_features(signal, n_lags=10):
    """Create a dataset with time-lagged features."""
    n_samples = len(signal) - n_lags
    X = np.zeros((n_samples, n_lags))
    
    for i in range(n_samples):
        X[i, :] = signal[i:i+n_lags]
    
    return X

# Create lagged dataset
X = create_time_lagged_features(noisy_signal)

# Apply PCA for noise reduction
pca = PCA(n_components=3)  # Keep only top 3 components
X_pca = pca.fit_transform(X)
X_denoised = pca.inverse_transform(X_pca)

# Reconstruct the denoised signal
denoised_signal = np.zeros_like(noisy_signal)
count = np.zeros_like(noisy_signal)

for i in range(len(X_denoised)):
    denoised_signal[i:i+X.shape[1]] += X_denoised[i, :]
    count[i:i+X.shape[1]] += 1

# Average overlapping reconstructions
denoised_signal /= np.maximum(count, 1)

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, clean_signal)
plt.title('Original Clean Signal')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, noisy_signal)
plt.title('Noisy Signal')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, denoised_signal)
plt.title(f'PCA Denoised Signal (using {pca.n_components_} components)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Calculate signal-to-noise ratio
def calculate_snr(clean, noisy):
    """Calculate Signal-to-Noise Ratio in dB."""
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - noisy) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

original_snr = calculate_snr(clean_signal, noisy_signal)
denoised_snr = calculate_snr(clean_signal, denoised_signal)

print(f"Original SNR: {original_snr:.2f} dB")
print(f"Denoised SNR: {denoised_snr:.2f} dB")
print(f"SNR Improvement: {denoised_snr - original_snr:.2f} dB")
```

### Feature Extraction for Machine Learning

PCA is widely used as a preprocessing step for machine learning models:

```python
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare different numbers of principal components
n_components_list = [2, 5, 8, 10, 13]  # 13 is all components
accuracies = []

for n_components in n_components_list:
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Train SVM
    svm = SVC(kernel='linear')
    svm.fit(X_train_pca, y_train)
    
    # Evaluate
    y_pred = svm.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    
    print(f"\nResults with {n_components} principal components:")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

# Plot accuracy vs. number of components
plt.figure(figsize=(10, 6))
plt.plot(n_components_list, accuracies, marker='o')
plt.title('Accuracy vs. Number of Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(n_components_list)
plt.show()

# Visualize first two principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaler.fit_transform(X))

plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']
for i, color in enumerate(colors):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=0.8,
                label=wine.target_names[i])

plt.title('PCA of Wine Dataset')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.legend()
plt.grid(True)
plt.show()
```

### Anomaly Detection

PCA can be used for anomaly detection by identifying data points with high reconstruction error:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Generate synthetic data with outliers
X, _ = make_blobs(n_samples=300, centers=1, random_state=42, cluster_std=1.0)

# Add outliers
outliers = np.random.uniform(low=-10, high=10, size=(15, 2))
X = np.vstack([X, outliers])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Compute reconstruction error
X_reconstructed = pca.inverse_transform(X_pca)
reconstruction_error = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)

# Determine threshold for anomalies
threshold = np.percentile(reconstruction_error, 95)  # Top 5% are anomalies

# Identify anomalies
anomalies = reconstruction_error > threshold

# Plot the results
plt.figure(figsize=(15, 5))

# Original data
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.7)
plt.title('Original Data')
plt.grid(True)

# Data with anomalies highlighted
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=['blue' if not a else 'red' for a in anomalies], alpha=0.7)
plt.title('Data with Anomalies Highlighted')
plt.grid(True)

# Reconstruction error
plt.subplot(1, 3, 3)
plt.hist(reconstruction_error, bins=50, alpha=0.7)
plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.2f}')
plt.title('Reconstruction Error')
plt.xlabel('Error')
plt.ylabel('Count')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Number of identified anomalies: {sum(anomalies)}")
```

### Visualization of High-Dimensional Data

PCA is commonly used to visualize high-dimensional data in 2D or 3D:

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Create 2D visualization
plt.figure(figsize=(12, 10))

# Create a scatter plot with custom colors
colors = plt.cm.tab10(np.arange(10) / 10)
for i in range(10):
    plt.scatter(
        X_pca[y == i, 0], X_pca[y == i, 1],
        color=colors[i], alpha=0.7,
        label=f'Digit {i}'
    )

plt.title('PCA of Digits Dataset (First 2 Components)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.legend()
plt.grid(True)
plt.show()

# Create 3D visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for i in range(10):
    ax.scatter(
        X_pca[y == i, 0], X_pca[y == i, 1], X_pca[y == i, 2],
        color=colors[i], alpha=0.7,
        label=f'Digit {i}'
    )

ax.set_title('PCA of Digits Dataset (First 3 Components)')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
ax.legend()
plt.show()
```

---

## 🔄 Variants of PCA

Several variants of PCA have been developed to address specific needs and limitations:

### Kernel PCA

Kernel PCA extends PCA to capture non-linear relationships in the data:

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

# Generate non-linear data
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

# Apply standard PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply Kernel PCA with different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
fig, axes = plt.subplots(1, len(kernels) + 1, figsize=(20, 4))

# Original data
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.8)
axes[0].set_title('Original Data')
axes[0].grid(True)

for i, kernel in enumerate(kernels):
    kpca = KernelPCA(n_components=2, kernel=kernel, gamma=10)
    X_kpca = kpca.fit_transform(X)
    
    axes[i+1].scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis', alpha=0.8)
    axes[i+1].set_title(f'Kernel PCA ({kernel.capitalize()})')
    axes[i+1].grid(True)

plt.tight_layout()
plt.show()
```

### Incremental PCA

Incremental PCA processes data in batches, allowing for large datasets that don't fit in memory:

```python
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
import matplotlib.pyplot as plt
import time

# Generate a large dataset
n_samples = 10000
n_features = 100
X = np.random.randn(n_samples, n_features)

# Benchmark standard PCA
start_time = time.time()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
pca_time = time.time() - start_time

# Benchmark Incremental PCA with different batch sizes
batch_sizes = [10, 100, 500, 1000]
ipca_times = []

for batch_size in batch_sizes:
    start_time = time.time()
    ipca = IncrementalPCA(n_components=2, batch_size=batch_size)
    X_ipca = ipca.fit_transform(X)
    ipca_times.append(time.time() - start_time)

# Plot timing comparison
plt.figure(figsize=(10, 6))
plt.bar(['PCA'] + [f'IPCA (batch={bs})' for bs in batch_sizes], 
        [pca_time] + ipca_times, color='skyblue')
plt.title('Computation Time: PCA vs. Incremental PCA')
plt.ylabel('Time (seconds)')
plt.grid(True, alpha=0.3)
plt.show()

# Verify that the results are similar
ipca = IncrementalPCA(n_components=2, batch_size=100)
X_ipca = ipca.fit_transform(X)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2)
plt.title('Standard PCA')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X_ipca[:, 0], X_ipca[:, 1], alpha=0.2)
plt.title('Incremental PCA')
plt.grid(True)

plt.tight_layout()
plt.show()

# Calculate the correlation between PCA and IPCA components
correlation_pc1 = np.corrcoef(X_pca[:, 0], X_ipca[:, 0])[0, 1]
correlation_pc2 = np.corrcoef(X_pca[:, 1], X_ipca[:, 1])[0, 1]

print(f"Correlation between PCA and IPCA for PC1: {abs(correlation_pc1):.4f}")
print(f"Correlation between PCA and IPCA for PC2: {abs(correlation_pc2):.4f}")
```

### Sparse PCA

Sparse PCA produces sparse loadings, making principal components easier to interpret:

```python
from sklearn.decomposition import PCA, SparsePCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Generate synthetic data
np.random.seed(42)
n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)

# Apply standard PCA
pca = PCA(n_components=2)
pca.fit(X)

# Apply Sparse PCA with different alpha values
alphas = [0.01, 0.1, 1, 10]
spca_components = []

for alpha in alphas:
    spca = SparsePCA(n_components=2, alpha=alpha, random_state=42)
    spca.fit(X)
    spca_components.append(spca.components_)

# Visualize the loadings
feature_names = [f'Feature {i+1}' for i in range(n_features)]

fig, axes = plt.subplots(1, len(alphas) + 1, figsize=(20, 5))

# Standard PCA
axes[0].bar(feature_names, pca.components_[0], alpha=0.7, label='PC1')
axes[0].bar(feature_names, pca.components_[1], alpha=0.5, label='PC2')
axes[0].set_title('Standard PCA')
axes[0].set_xticklabels(feature_names, rotation=90)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Sparse PCA with different alphas
for i, (alpha, components) in enumerate(zip(alphas, spca_components)):
    axes[i+1].bar(feature_names, components[0], alpha=0.7, label='PC1')
    axes[i+1].bar(feature_names, components[1], alpha=0.5, label='PC2')
    axes[i+1].set_title(f'Sparse PCA (alpha={alpha})')
    axes[i+1].set_xticklabels(feature_names, rotation=90)
    axes[i+1].legend()
    axes[i+1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Count non-zero loadings
def count_nonzero(components):
    """Count non-zero loadings for each component."""
    return [np.count_nonzero(comp) for comp in components]

nonzero_counts = [count_nonzero(pca.components_)] + [count_nonzero(comp) for comp in spca_components]

# Create a DataFrame for better visualization
nonzero_df = pd.DataFrame(
    nonzero_counts, 
    columns=['PC1', 'PC2'],
    index=['Standard PCA'] + [f'Sparse PCA (alpha={alpha})' for alpha in alphas]
)

print("Number of non-zero loadings in each component:")
print(nonzero_df)

# Plot number of non-zero loadings
plt.figure(figsize=(12, 6))
nonzero_df.plot(kind='bar')
plt.title('Number of Non-Zero Loadings')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.show()
```

### Robust PCA

Robust PCA is less sensitive to outliers in the data:

```python
# Note: A complete implementation of Robust PCA would be complex.
# Here's a simplified version using scikit-learn's RobustScaler with PCA.

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Generate data with outliers
np.random.seed(42)
X = np.random.randn(100, 2)
X = np.vstack([X, np.array([10, 10]), np.array([-10, 10])])  # Add outliers

# Apply standard PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply "Robust" PCA (using RobustScaler + PCA)
robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)
robust_pca = PCA(n_components=2)
X_robust_pca = robust_pca.fit_transform(X_robust)

# Visualize results
plt.figure(figsize=(15, 5))

# Original data
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
plt.title('Original Data with Outliers')
plt.grid(True)

# Standard PCA
plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
plt.title('Standard PCA')
plt.grid(True)

# Robust PCA
plt.subplot(1, 3, 3)
plt.scatter(X_robust_pca[:, 0], X_robust_pca[:, 1], alpha=0.7)
plt.title('Robust PCA')
plt.grid(True)

plt.tight_layout()
plt.show()

# Compare principal components
print("Standard PCA components:")
print(pca.components_)
print("\nRobust PCA components:")
print(robust_pca.components_)
```

### Probabilistic PCA

Probabilistic PCA views PCA from a probabilistic perspective:

```python
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Generate synthetic data
np.random.seed(42)
n_samples = 500
X = np.random.randn(n_samples, 2)
X = X @ np.array([[2, 1], [1, 2]])  # Introduce correlation

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Extract parameters for probabilistic interpretation
mean = np.mean(X, axis=0)
components = pca.components_
explained_variance = pca.explained_variance_

# Generate samples from the probabilistic model
def sample_ppca(n_samples, mean, components, explained_variance, n_components):
    """Generate samples from probabilistic PCA model."""
    # Sample from latent space
    latent = np.random.randn(n_samples, n_components)
    
    # Scale by sqrt of eigenvalues
    latent_scaled = latent * np.sqrt(explained_variance)
    
    # Transform to data space
    X_sampled = latent_scaled @ components + mean
    
    return X_sampled

# Generate samples
X_sampled = sample_ppca(n_samples, mean, components, explained_variance, 2)

# Visualize original data and samples
plt.figure(figsize=(15, 5))

# Original data
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.7, label='Original')
plt.title('Original Data')
plt.grid(True)

# Data transformed to PC space and back
X_reconstructed = pca.inverse_transform(X_pca)
plt.subplot(1, 3, 2)
plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], alpha=0.7, label='Reconstructed')
plt.title('Reconstructed Data')
plt.grid(True)

# Samples from probabilistic model
plt.subplot(1, 3, 3)
plt.scatter(X_sampled[:, 0], X_sampled[:, 1], alpha=0.7, label='Sampled', color='green')
plt.title('Samples from Probabilistic PCA')
plt.grid(True)

plt.tight_layout()
plt.show()

# Create contour plot of the probability density
def plot_probability_contours(X, mean, covariance, title):
    """Plot contours of the probability density function."""
    x, y = np.meshgrid(np.linspace(-6, 6, 100), np.linspace(-6, 6, 100))
    pos = np.dstack((x, y))
    
    rv = multivariate_normal(mean, covariance)
    z = rv.pdf(pos)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label='Data')
    plt.contour(x, y, z, levels=10, cmap='viridis')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

# Calculate data covariance and PPCA covariance
data_cov = np.cov(X, rowvar=False)
ppca_cov = components.T @ np.diag(explained_variance) @ components

# Plot probability contours
plot_probability_contours(X, mean, data_cov, 'Data Distribution')
plot_probability_contours(X, mean, ppca_cov, 'Probabilistic PCA Distribution')
```

---

## 🔄 Comparison with Other Methods

PCA is one of many dimensionality reduction techniques. Here's how it compares to others:

### PCA vs. Linear Discriminant Analysis (LDA)

```python
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import numpy as np

# Load wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply LDA
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)

# Visualize the results
plt.figure(figsize=(16, 6))

# PCA
plt.subplot(1, 2, 1)
for i, target_name in enumerate(wine.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target_name)
plt.title('PCA of Wine Dataset')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.legend()
plt.grid(True)

# LDA
plt.subplot(1, 2, 2)
for i, target_name in enumerate(wine.target_names):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], label=target_name)
plt.title('LDA of Wine Dataset')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Compare classification performance
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Function to evaluate classifier with different dimensionality reduction techniques
def evaluate_classifier(X, y, method_name, n_components_list):
    """Evaluate classifier with different numbers of components."""
    results = []
    
    for n_components in n_components_list:
        if method_name == 'PCA':
            method = PCA(n_components=n_components)
            X_transformed = method.fit_transform(X)
        elif method_name == 'LDA':
            # LDA can have at most n_classes - 1 components
            n_components = min(n_components, len(np.unique(y)) - 1)
            method = LDA(n_components=n_components)
            X_transformed = method.fit_transform(X, y)
        
        clf = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(clf, X_transformed, y, cv=5)
        results.append((n_components, np.mean(scores)))
    
    return results

# Evaluate with different numbers of components
n_components_list = [1, 2, 3, 5, 8, 10, 13]  # 13 is max for Wine dataset
pca_results = evaluate_classifier(X, y, 'PCA', n_components_list)
lda_results = evaluate_classifier(X, y, 'LDA', n_components_list)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot([r[0] for r in pca_results], [r[1] for r in pca_results], 'o-', label='PCA')
plt.plot([r[0] for r in lda_results], [r[1] for r in lda_results], 's-', label='LDA')
plt.title('Classification Accuracy: PCA vs. LDA')
plt.xlabel('Number of Components')
plt.ylabel('Cross-Validation Accuracy')
plt.xticks(n_components_list)
plt.legend()
plt.grid(True)
plt.show()

# Key differences
print("Key Differences between PCA and LDA:")
print("1. PCA is unsupervised, LDA is supervised")
print("2. PCA maximizes variance, LDA maximizes class separation")
print("3. LDA can have at most n_classes - 1 components")
print("4. PCA works better when classes are not well separated")
print("5. LDA often performs better for classification tasks")
```

### PCA vs. t-SNE

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.datasets import load_digits

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply PCA
start_time = time.time()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
pca_time = time.time() - start_time

# Apply t-SNE
start_time = time.time()
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
tsne_time = time.time() - start_time

# Visualize the results
plt.figure(figsize=(16, 6))

# PCA
plt.subplot(1, 2, 1)
for i in range(10):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=str(i), alpha=0.7, s=30)
plt.title(f'PCA of Digits Dataset (Time: {pca_time:.2f}s)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.legend()
plt.grid(True)

# t-SNE
plt.subplot(1, 2, 2)
for i in range(10):
    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=str(i), alpha=0.7, s=30)
plt.title(f't-SNE of Digits Dataset (Time: {tsne_time:.2f}s)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Measure cluster separation
from sklearn.metrics import silhouette_score

# Note: Silhouette score is not always appropriate for t-SNE, 
# but used here for simple comparison
pca_silhouette = silhouette_score(X_pca, y)
tsne_silhouette = silhouette_score(X_tsne, y)

print(f"PCA Silhouette Score: {pca_silhouette:.4f}")
print(f"t-SNE Silhouette Score: {tsne_silhouette:.4f}")

# Key differences
print("\nKey Differences between PCA and t-SNE:")
print("1. PCA is linear, t-SNE is non-linear")
print("2. PCA preserves global structure, t-SNE preserves local structure")
print("3. PCA is deterministic, t-SNE is stochastic")
print("4. PCA is much faster than t-SNE")
print("5. t-SNE does not preserve distances or densities")
print("6. PCA allows for new data projection, t-SNE does not")
print(f"7. Time difference: t-SNE took {tsne_time/pca_time:.1f}x longer than PCA")
```

### PCA vs. Autoencoders

```python
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
import time

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Normalize the data
X = X / 16.0  # Digits are 0-16

# Apply PCA
start_time = time.time()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
pca_time = time.time() - start_time

# Reconstruct from PCA
X_reconstructed_pca = pca.inverse_transform(X_pca)

# Create an autoencoder
def create_autoencoder(input_dim, encoding_dim):
    # Encoder
    input_layer = tf.keras.Input(shape=(input_dim,))
    encoder = layers.Dense(128, activation='relu')(input_layer)
    encoder = layers.Dense(64, activation='relu')(encoder)
    encoder = layers.Dense(encoding_dim, activation='linear')(encoder)
    
    # Decoder
    decoder = layers.Dense(64, activation='relu')(encoder)
    decoder = layers.Dense(128, activation='relu')(decoder)
    decoder = layers.Dense(input_dim, activation='sigmoid')(decoder)
    
    # Autoencoder model
    autoencoder = models.Model(inputs=input_layer, outputs=decoder)
    
    # Encoder model
    encoder_model = models.Model(inputs=input_layer, outputs=encoder)
    
    return autoencoder, encoder_model

# Create and train autoencoder
input_dim = X.shape[1]
encoding_dim = 2  # Same as PCA for comparison
autoencoder, encoder = create_autoencoder(input_dim, encoding_dim)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

start_time = time.time()
history = autoencoder.fit(
    X, X,
    epochs=50,
    batch_size=256,
    shuffle=True,
    verbose=0,
    validation_split=0.2
)
autoencoder_time = time.time() - start_time

# Encode and decode with autoencoder
X_encoded = encoder.predict(X)
X_reconstructed_ae = autoencoder.predict(X)

# Visualize the results
plt.figure(figsize=(20, 15))

# Original images
plt.subplot(2, 3, 1)
for i in range(10):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=str(i), alpha=0.7, s=30)
plt.title(f'PCA Projection (Time: {pca_time:.2f}s)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 2)
for i in range(10):
    plt.scatter(X_encoded[y == i, 0], X_encoded[y == i, 1], label=str(i), alpha=0.7, s=30)
plt.title(f'Autoencoder Encoding (Time: {autoencoder_time:.2f}s)')
plt.xlabel('Encoding Dim 1')
plt.ylabel('Encoding Dim 2')
plt.legend()
plt.grid(True)

# Reconstruction error
plt.subplot(2, 3, 3)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)

# Example reconstructions
def plot_digits(X, title, subplot_index):
    """Plot example digits and their reconstructions."""
    plt.subplot(2, 3, subplot_index)
    n_samples = 5
    for i in range(n_samples):
        plt.subplot(2, 3, subplot_index + i // n_samples)
        idx = i * 100  # Pick well-separated examples
        plt.imshow(X[idx].reshape(8, 8), cmap='gray')
        plt.axis('off')
    plt.title(title)

# Original digits
plot_digits(X, 'Original Digits', 4)

# PCA reconstructions
plot_digits(X_reconstructed_pca, 'PCA Reconstructions', 5)

# Autoencoder reconstructions
plot_digits(X_reconstructed_ae, 'Autoencoder Reconstructions', 6)

plt.tight_layout()
plt.show()

# Quantitative comparison
pca_mse = np.mean((X - X_reconstructed_pca) ** 2)
ae_mse = np.mean((X - X_reconstructed_ae) ** 2)

print(f"PCA Reconstruction MSE: {pca_mse:.6f}")
print(f"Autoencoder Reconstruction MSE: {ae_mse:.6f}")
print(f"Computation Time - PCA: {pca_time:.2f}s, Autoencoder: {autoencoder_time:.2f}s")

# Key differences
print("\nKey Differences between PCA and Autoencoders:")
print("1. PCA is linear, Autoencoders can be non-linear")
print("2. PCA has a closed-form solution, Autoencoders require iterative training")
print("3. PCA is much faster to train than Autoencoders")
print("4. Autoencoders can capture more complex patterns and often achieve better reconstruction")
print("5. Autoencoders have many more parameters and risk overfitting")
print("6. PCA components are orthogonal, Autoencoder dimensions are not constrained to be orthogonal")
print("7. Autoencoders are more flexible in architecture design (can add regularization, etc.)")
```

### PCA vs. Non-negative Matrix Factorization (NMF)

```python
from sklearn.decomposition import PCA, NMF
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
import time

# Load faces dataset
faces = fetch_olivetti_faces(shuffle=True)
X = faces.data
y = faces.target

# Apply PCA
start_time = time.time()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
pca_time = time.time() - start_time

# Apply NMF
start_time = time.time()
nmf = NMF(n_components=2, init='random', random_state=42)
X_nmf = nmf.fit_transform(X)
nmf_time = time.time() - start_time

# Visualize the components
n_components = 6  # Show first 6 components
pca_high_dim = PCA(n_components=n_components)
pca_high_dim.fit(X)

nmf_high_dim = NMF(n_components=n_components, init='random', random_state=42)
nmf_high_dim.fit(X)

plt.figure(figsize=(12, 8))

for i in range(n_components):
    # Plot PCA components
    plt.subplot(2, n_components, i + 1)
    comp = pca_high_dim.components_[i].reshape(64, 64)
    plt.imshow(comp, cmap='viridis')
    plt.title(f'PCA Component {i+1}')
    plt.axis('off')
    
    # Plot NMF components
    plt.subplot(2, n_components, i + n_components + 1)
    comp = nmf_high_dim.components_[i].reshape(64, 64)
    plt.imshow(comp, cmap='viridis')
    plt.title(f'NMF Component {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Compare reconstructions
pca_reconstruction = pca_high_dim.inverse_transform(pca_high_dim.transform(X))
nmf_reconstruction = nmf_high_dim.inverse_transform(nmf_high_dim.transform(X))

plt.figure(figsize=(15, 5))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(X[0].reshape(64, 64), cmap='gray')
plt.title('Original Face')
plt.axis('off')

# PCA reconstruction
plt.subplot(1, 3, 2)
plt.imshow(pca_reconstruction[0].reshape(64, 64), cmap='gray')
plt.title(f'PCA Reconstruction ({n_components} components)')
plt.axis('off')

# NMF reconstruction
plt.subplot(1, 3, 3)
plt.imshow(nmf_reconstruction[0].reshape(64, 64), cmap='gray')
plt.title(f'NMF Reconstruction ({n_components} components)')
plt.axis('off')

plt.tight_layout()
plt.show()

# Quantitative comparison
pca_mse = np.mean((X - pca_reconstruction) ** 2)
nmf_mse = np.mean((X - nmf_reconstruction) ** 2)

print(f"PCA Reconstruction MSE: {pca_mse:.6f}")
print(f"NMF Reconstruction MSE: {nmf_mse:.6f}")
print(f"Computation Time - PCA: {pca_time:.2f}s, NMF: {nmf_time:.2f}s")

# Sparsity comparison
pca_sparsity = np.mean(pca_high_dim.components_ == 0)
nmf_sparsity = np.mean(nmf_high_dim.components_ == 0)

print(f"PCA Component Sparsity: {pca_sparsity:.6f}")
print(f"NMF Component Sparsity: {nmf_sparsity:.6f}")

# Key differences
print("\nKey Differences between PCA and NMF:")
print("1. NMF enforces non-negativity, PCA allows negative values")
print("2. NMF often produces more interpretable components")
print("3. PCA is faster to compute than NMF")
print("4. NMF is better for decomposing additive signals (e.g., images, text)")
print("5. PCA has a unique solution, NMF can have multiple local minima")
print("6. NMF can achieve more sparsity in its components")
print("7. PCA is better for dimensionality reduction, NMF for finding parts-based representation")
```

---

## ⚠️ Limitations and Considerations

While PCA is a powerful technique, it has several limitations to keep in mind:

### Linear Assumptions

PCA assumes linear relationships between variables, which may not hold for all datasets:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate non-linear data: a half circle
t = np.linspace(-np.pi/2, np.pi/2, 200)
x = np.cos(t)
y = np.sin(t)
X = np.column_stack([x, y])

# Add some noise
X += np.random.normal(0, 0.1, X.shape)

# Apply PCA
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(X_pca)

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.8)
plt.title('Original Data')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label='Original')
plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], alpha=0.8, color='red', label='PCA Reconstruction')
plt.title('PCA Reconstruction (1 component)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Note how PCA fails to capture the non-linear structure of the half-circle.")
print("For non-linear data, consider using non-linear dimensionality reduction methods like:")
print("- Kernel PCA")
print("- t-SNE")
print("- UMAP")
print("- Autoencoders")
```

### Sensitivity to Scaling

PCA is sensitive to the scaling of the variables:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Generate data
np.random.seed(42)
n_samples = 100
x = np.random.normal(0, 1, n_samples)
y = x + np.random.normal(0, 0.5, n_samples)

# Scale y to be much larger than x
X_unscaled = np.column_stack([x, y * 100])

# Apply PCA without scaling
pca_unscaled = PCA(n_components=1)
X_pca_unscaled = pca_unscaled.fit_transform(X_unscaled)
X_reconstructed_unscaled = pca_unscaled.inverse_transform(X_pca_unscaled)

# Apply standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unscaled)

# Apply PCA with scaling
pca_scaled = PCA(n_components=1)
X_pca_scaled = pca_scaled.fit_transform(X_scaled)
X_reconstructed_scaled = pca_scaled.inverse_transform(X_pca_scaled)
X_reconstructed_scaled = scaler.inverse_transform(X_reconstructed_scaled)

# Visualize
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.scatter(X_unscaled[:, 0], X_unscaled[:, 1], alpha=0.8)
plt.title('Original Data (Unscaled)')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.scatter(X_unscaled[:, 0], X_unscaled[:, 1], alpha=0.3, label='Original')
plt.scatter(X_reconstructed_unscaled[:, 0], X_reconstructed_unscaled[:, 1], 
           alpha=0.8, color='red', label='PCA Reconstruction')
plt.title('PCA on Unscaled Data')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.8)
plt.title('Scaled Data')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.scatter(X_unscaled[:, 0], X_unscaled[:, 1], alpha=0.3, label='Original')
plt.scatter(X_reconstructed_scaled[:, 0], X_reconstructed_scaled[:, 1], 
           alpha=0.8, color='green', label='PCA Reconstruction (after scaling)')
plt.title('PCA on Scaled Data')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Compare explained variance
print(f"Explained variance without scaling: {pca_unscaled.explained_variance_ratio_[0]:.4f}")
print(f"Explained variance with scaling: {pca_scaled.explained_variance_ratio_[0]:.4f}")
print(f"Principal components without scaling: {pca_unscaled.components_}")
print(f"Principal components with scaling: {pca_scaled.components_}")
```

### Outlier Sensitivity

PCA can be highly sensitive to outliers:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate data
np.random.seed(42)
n_samples = 100
x = np.random.normal(0, 1, n_samples)
y = x + np.random.normal(0, 0.5, n_samples)
X = np.column_stack([x, y])

# Add outliers
X_with_outliers = X.copy()
X_with_outliers[0, :] = [10, 10]  # Add an outlier

# Apply PCA
pca_no_outliers = PCA(n_components=1)
X_pca_no_outliers = pca_no_outliers.fit_transform(X)
X_reconstructed_no_outliers = pca_no_outliers.inverse_transform(X_pca_no_outliers)

pca_with_outliers = PCA(n_components=1)
X_pca_with_outliers = pca_with_outliers.fit_transform(X_with_outliers)
X_reconstructed_with_outliers = pca_with_outliers.inverse_transform(X_pca_with_outliers)

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.8, label='Data')
plt.scatter(X_reconstructed_no_outliers[:, 0], X_reconstructed_no_outliers[:, 1], 
           alpha=0.8, color='green', label='PCA Reconstruction')
# Plot the principal component direction
direction = pca_no_outliers.components_[0]
plt.arrow(0, 0, direction[0]*3, direction[1]*3, head_width=0.2, head_length=0.2, 
          fc='red', ec='red', label='PC1 Direction')
plt.title('PCA without Outliers')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X_with_outliers[:, 0], X_with_outliers[:, 1], alpha=0.8, label='Data with Outlier')
plt.scatter(X_with_outliers[0, 0], X_with_outliers[0, 1], s=100, 
           facecolors='none', edgecolors='red', label='Outlier')
plt.scatter(X_reconstructed_with_outliers[:, 0], X_reconstructed_with_outliers[:, 1], 
           alpha=0.8, color='green', label='PCA Reconstruction')
# Plot the principal component direction
direction = pca_with_outliers.components_[0]
plt.arrow(0, 0, direction[0]*3, direction[1]*3, head_width=0.2, head_length=0.2, 
          fc='red', ec='red', label='PC1 Direction')
plt.title('PCA with Outliers')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Compare principal components
print(f"PC1 without outliers: {pca_no_outliers.components_[0]}")
print(f"PC1 with outliers: {pca_with_outliers.components_[0]}")
print(f"PC1 angle without outliers: {np.arctan2(pca_no_outliers.components_[0, 1], pca_no_outliers.components_[0, 0]) * 180 / np.pi:.2f} degrees")
print(f"PC1 angle with outliers: {np.arctan2(pca_with_outliers.components_[0, 1], pca_with_outliers.components_[0, 0]) * 180 / np.pi:.2f} degrees")
```

### Interpretability Challenges

Principal components can be difficult to interpret in high-dimensional spaces:

```python
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load Boston housing dataset
boston = load_boston()
X = boston.data
feature_names = boston.feature_names

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame of loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=feature_names
)

# Plot the loadings for the first two principal components
plt.figure(figsize=(12, 10))

# PC1 loadings
plt.subplot(2, 1, 1)
loadings_pc1 = loadings.iloc[:, 0].sort_values(ascending=False)
plt.bar(loadings_pc1.index, loadings_pc1.values)
plt.title('Feature Loadings for PC1')
plt.xticks(rotation=90)
plt.grid(True, alpha=0.3)

# PC2 loadings
plt.subplot(2, 1, 2)
loadings_pc2 = loadings.iloc[:, 1].sort_values(ascending=False)
plt.bar(loadings_pc2.index, loadings_pc2.values)
plt.title('Feature Loadings for PC2')
plt.xticks(rotation=90)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Contribution of each feature to first few PCs
n_pcs = 3
plt.figure(figsize=(15, 8))
plt.imshow(np.abs(loadings.iloc[:, :n_pcs]), aspect='auto', cmap='viridis')
plt.colorbar(label='Absolute Loading Value')
plt.xticks(np.arange(n_pcs), loadings.columns[:n_pcs])
plt.yticks(np.arange(len(feature_names)), feature_names)
plt.title('Feature Contributions to Principal Components')
plt.tight_layout()
plt.show()

print("Interpreting principal components can be challenging because:")
print("1. Each PC is a linear combination of all original features")
print("2. Loadings may have similar magnitudes, making it hard to identify dominant features")
print("3. The meaning of each PC depends on domain knowledge")
print("4. High-dimensional spaces are inherently difficult to visualize")
print("\nSome strategies to improve interpretability:")
print("- Examine loadings to understand feature contributions")
print("- Use sparse PCA for more interpretable components")
print("- Apply rotation methods like Varimax")
print("- Plot data in the PC space and label points to identify patterns")
```

### Not Optimized for Prediction

PCA is not optimized for predictive tasks, unlike supervised methods:

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load Boston housing dataset
boston = load_boston()
X, y = boston.data, boston.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA with different numbers of components
n_components_list = range(1, X.shape[1] + 1)
r2_scores = []
mse_scores = []

for n_components in n_components_list:
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train_pca, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_pca)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    r2_scores.append(r2)
    mse_scores.append(mse)

# Also train a model on the original features
model_original = LinearRegression()
model_original.fit(X_train_scaled, y_train)
y_pred_original = model_original.predict(X_test_scaled)
r2_original = r2_score(y_test, y_pred_original)
mse_original = mean_squared_error(y_test, y_pred_original)

# Visualize the results
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(n_components_list, r2_scores, marker='o')
plt.axhline(y=r2_original, color='r', linestyle='--', label=f'Original Features (R² = {r2_original:.4f})')
plt.xlabel('Number of Principal Components')
plt.ylabel('R² Score')
plt.title('R² Score vs. Number of Components')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(n_components_list, mse_scores, marker='o')
plt.axhline(y=mse_original, color='r', linestyle='--', label=f'Original Features (MSE = {mse_original:.4f})')
plt.xlabel('Number of Principal Components')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs. Number of Components')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Find the optimal number of components
optimal_n = n_components_list[np.argmax(r2_scores)]
print(f"Optimal number of components for prediction: {optimal_n}")
print(f"R² with optimal components: {max(r2_scores):.4f}")
print(f"R² with all original features: {r2_original:.4f}")

print("\nReasons why PCA might not be optimal for prediction:")
print("1. PCA maximizes variance, not predictive power")
print("2. Important predictive information might be in low-variance directions")
print("3. Supervised dimensionality reduction methods (like PLS) can perform better for prediction")
print("4. Feature selection methods might retain more predictive information")
```

---

## 🛠️ Best Practices

### When to Use PCA

PCA is particularly useful in certain scenarios:

1. **High-Dimensional Data**: When dealing with datasets with many features
2. **Multicollinearity**: When features are highly correlated
3. **Visualization**: To reduce dimensions for visualization purposes
4. **Noise Reduction**: To filter out noise in the data
5. **Preprocessing**: Before applying machine learning algorithms
6. **Data Compression**: To efficiently store and transmit data

### Preprocessing Recommendations

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.datasets import load_wine

# Load wine dataset
wine = load_wine()
X = wine.data
feature_names = wine.feature_names

# Create different scaling options
scalers = {
    'No Scaling': None,
    'Standard Scaling': StandardScaler(),
    'Min-Max Scaling': MinMaxScaler(),
    'Robust Scaling': RobustScaler()
}

# Apply PCA with different scaling methods
results = {}

plt.figure(figsize=(15, 12))

for i, (name, scaler) in enumerate(scalers.items()):
    # Apply scaling if not None
    if scaler is not None:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Store results
    results[name] = {
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'components': pca.components_
    }
    
    # Plot explained variance
    plt.subplot(2, 2, i+1)
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.7)
    plt.step(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), where='mid', color='red', marker='o')
    plt.title(f'Explained Variance with {name}')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# Compare the first principal components
plt.figure(figsize=(15, 6))

for i, (name, result) in enumerate(results.items()):
    plt.subplot(1, len(scalers), i+1)
    plt.bar(feature_names, result['components'][0], alpha=0.7)
    plt.title(f'First PC with {name}')
    plt.xticks(rotation=90)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Recommendations
print("Best Practices for Preprocessing in PCA:")
print("1. Always standardize the data (zero mean, unit variance) when features have different scales")
print("2. Use robust scaling when data contains outliers")
print("3. Ensure data quality by handling missing values before PCA")
print("4. Consider normalizing data when the magnitude of features is important")
print("5. Perform outlier detection and treatment before applying PCA")
print("6. Log-transform highly skewed features")
print("7. Remove or impute missing values (PCA cannot handle missing data directly)")
```

### Implementing PCA in Production

```python
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris

# Sample data
iris = load_iris()
X, y = iris.data, iris.target

# Create a pipeline with preprocessing and PCA
pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2))
])

# Fit the pipeline
pca_pipeline.fit(X)

# Save the fitted pipeline for production use
joblib.dump(pca_pipeline, 'pca_pipeline.joblib')

# Example of loading and using the pipeline in production
loaded_pipeline = joblib.load('pca_pipeline.joblib')

# Transform new data
X_new = X[:5]  # Just as an example
X_new_transformed = loaded_pipeline.transform(X_new)

print("Transformed data shape:", X_new_transformed.shape)
print("Transformed data:")
print(X_new_transformed)

# Production code example for applying PCA
def apply_pca_transform(data, pipeline_path='pca_pipeline.joblib'):
    """
    Apply PCA transformation to new data using a saved pipeline.
    
    Parameters:
    -----------
    data : numpy array or pandas DataFrame
        New data to transform
    pipeline_path : str
        Path to the saved pipeline
        
    Returns:
    --------
    numpy array
        Transformed data
    """
    # Load the pipeline
    pipeline = joblib.load(pipeline_path)
    
    # Transform the data
    transformed_data = pipeline.transform(data)
    
    return transformed_data

# Example usage
transformed = apply_pca_transform(X_new)
print("\nTransformed using the function:", transformed)

# Best practices for production
print("\nBest Practices for Implementing PCA in Production:")
print("1. Use a Pipeline to combine preprocessing steps with PCA")
print("2. Save the fitted pipeline for consistent application to new data")
print("3. Include all preprocessing steps in the pipeline to avoid inconsistencies")
print("4. Validate the pipeline on a holdout set before deployment")
print("5. Monitor performance metrics in production to detect data drift")
print("6. Consider incremental PCA for large datasets or streaming data")
print("7. Use error handling for input validation and edge cases")
print("8. Document the expected input format and preprocessing requirements")
```

### Interpreting PCA Results

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# Load breast cancer dataset
cancer = load_breast_cancer()
X = cancer.data
feature_names = cancer.feature_names
y = cancer.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for easier visualization
df_pca = pd.DataFrame(data=X_pca[:, :5], columns=[f'PC{i+1}' for i in range(5)])
df_pca['target'] = y
df_pca['diagnosis'] = ['Malignant' if t == 0 else 'Benign' for t in y]

# 1. Scree Plot: Explained Variance
plt.figure(figsize=(12, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.7)
plt.step(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), where='mid', color='red', marker='o')
plt.title('Scree Plot: Explained Variance by Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Biplot: Feature Loadings and Data Points
def biplot(scores, loadings, labels=None, var_explained=None, n_components=2):
    """Create a biplot of PC scores and feature loadings."""
    plt.figure(figsize=(12, 10))
    
    # Plot scores (data points)
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(scores[mask, 0], scores[mask, 1], color=colors[i], alpha=0.7, label=label)
        plt.legend()
    else:
        plt.scatter(scores[:, 0], scores[:, 1], alpha=0.7)
    
    # Plot loadings (feature vectors)
    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, loadings[i, 0] * 5, loadings[i, 1] * 5, color='red', alpha=0.5)
        plt.text(loadings[i, 0] * 5.2, loadings[i, 1] * 5.2, feature, color='red', ha='center', va='center', fontsize=9)
    
    # Add labels and title
    plt.xlabel(f'PC1 ({var_explained[0]:.2%} explained variance)')
    plt.ylabel(f'PC2 ({var_explained[1]:.2%} explained variance)')
    plt.title('Biplot: PC Scores and Feature Loadings')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Create biplot
biplot(X_pca, pca.components_.T, labels=y, var_explained=pca.explained_variance_ratio_)

# 3. Feature Loadings Heatmap
plt.figure(figsize=(12, 8))
loadings = pd.DataFrame(
    pca.components_.T[:, :5],
    columns=[f'PC{i+1}' for i in range(5)],
    index=feature_names
)
plt.imshow(loadings, aspect='auto', cmap='coolwarm')
plt.colorbar(label='Loading Value')
plt.xticks(np.arange(5), loadings.columns)
plt.yticks(np.arange(len(feature_names)), feature_names)
plt.title('Feature Loadings Heatmap')
plt.tight_layout()
plt.show()

# 4. Contribution of Features to Principal Components
def feature_contribution(pca, feature_names, n_components=3):
    """Calculate and visualize feature contributions to principal components."""
    # Calculate absolute loadings
    loadings_df = pd.DataFrame(
        np.abs(pca.components_.T[:, :n_components]),
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_names
    )
    
    # Calculate contribution percentage
    for col in loadings_df.columns:
        loadings_df[col] = loadings_df[col] / loadings_df[col].sum() * 100
    
    # Sort features by contribution to PC1
    loadings_df = loadings_df.sort_values(by='PC1', ascending=False)
    
    # Visualize top contributing features
    plt.figure(figsize=(15, 10))
    for i in range(n_components):
        plt.subplot(1, n_components, i+1)
        top_features = loadings_df.iloc[:, i].sort_values(ascending=False).head(10)
        plt.barh(top_features.index, top_features.values, color=f'C{i}')
        plt.title(f'Top Features for PC{i+1}')
        plt.xlabel('Contribution (%)')
        plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    return loadings_df

# Analyze feature contributions
contribution_df = feature_contribution(pca, feature_names)

# 5. Correlation between Original Features and Principal Components
def feature_pc_correlation(X_scaled, X_pca, feature_names, n_components=3):
    """Calculate correlation between original features and principal components."""
    # Create DataFrames
    df_features = pd.DataFrame(X_scaled, columns=feature_names)
    df_pca = pd.DataFrame(X_pca[:, :n_components], columns=[f'PC{i+1}' for i in range(n_components)])
    
    # Combine DataFrames
    combined_df = pd.concat([df_features, df_pca], axis=1)
    
    # Calculate correlation
    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    correlation = combined_df.corr().loc[feature_names, pc_columns]
    
    # Visualize correlation
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.xticks(np.arange(n_components), pc_columns)
    plt.yticks(np.arange(len(feature_names)), feature_names)
    plt.title('Correlation between Features and Principal Components')
    plt.tight_layout()
    plt.show()
    
    return correlation

# Calculate feature-PC correlation
correlation = feature_pc_correlation(X_scaled, X_pca, feature_names)

# Interpretation guidelines
print("Guidelines for Interpreting PCA Results:")
print("1. Explained Variance: Higher explained variance indicates more information retention")
print("2. Scree Plot: Look for an 'elbow' to determine the optimal number of components")
print("3. Loadings: Examine the weights of original features in each principal component")
print("4. Biplot: Identify relationships between features and principal components")
print("5. Feature Contributions: Identify which features contribute most to each PC")
print("6. PC Scores: Data points with similar scores on PCs are similar in the original space")
print("7. Correlation: High correlation between a feature and PC indicates strong influence")
```

---

## ❓ FAQ

### Q1: When should I use PCA instead of other dimensionality reduction techniques?

**A:** PCA is a good choice when:

1. **You want a linear transformation** of your data
2. **Preserving variance is important** for your analysis
3. **Computational efficiency is a concern** (PCA is faster than many alternatives)
4. **You need a deterministic result** (unlike stochastic methods like t-SNE)
5. **You need to project new data** onto the same lower-dimensional space
6. **Your data has approximately linear relationships** between features
7. **Interpretability of the components** is desired

Consider alternatives when:
- **Your data has non-linear patterns** (use Kernel PCA, t-SNE, or UMAP)
- **You're working with sparse data** (use Truncated SVD/LSA)
- **You want non-negative components** (use NMF)
- **Classification performance is the goal** (use LDA)
- **You need local structure preservation** (use t-SNE or UMAP)

### Q2: How many principal components should I retain?

**A:** There are several approaches to determine the optimal number of components:

1. **Explained Variance Threshold**: Keep components that explain a cumulative percentage of variance (e.g., 95%)
2. **Scree Plot**: Look for an "elbow" in the explained variance plot
3. **Kaiser Rule**: Retain components with eigenvalues greater than 1 (for standardized data)
4. **Parallel Analysis**: Compare eigenvalues with those from random data
5. **Cross-Validation**: Select the number of components that optimizes a downstream task

The best approach depends on your specific goals:
- For visualization: Usually 2-3 components
- For noise reduction: Components that explain most of the variance
- For data compression: Based on acceptable information loss
- For preprocessing: Based on downstream model performance

### Q3: How do I handle categorical variables with PCA?

**A:** PCA is designed for continuous variables, but there are several approaches for handling categorical data:

1. **One-hot encoding** categorical variables before applying PCA
2. **Multiple Correspondence Analysis (MCA)**, a variant of PCA for categorical data
3. **Converting ordinal categorical variables** to numeric values (if appropriate)
4. **Creating dummy variables** and then applying PCA
5. **Gower distance** with Principal Coordinates Analysis for mixed data types

Remember that one-hot encoding can lead to high-dimensional sparse data, which may not be ideal for PCA. Consider:
- Using a dimensionality reduction technique designed for categorical data
- Treating categorical and continuous variables separately
- Using an autoencoder with appropriate embeddings for categorical variables

### Q4: Can PCA help with multicollinearity in regression?

**A:** Yes, PCA can help address multicollinearity in regression models in several ways:

1. **Principal Component Regression (PCR)**: Replace original features with principal components in regression
2. **Removing multicollinearity**: Since PCs are orthogonal to each other
3. **Feature selection**: Identifying and removing redundant features
4. **Regularization**: Similar effect to L2 regularization in some cases

Benefits of using PCA for regression with multicollinearity:
- More stable coefficient estimates
- Reduced variance in predictions
- Improved numerical stability
- Clearer interpretation of feature importance

However, PCR may sacrifice interpretability of individual feature effects and doesn't optimize for prediction directly (unlike Partial Least Squares regression).

### Q5: How do I interpret the principal components?

**A:** Interpreting principal components involves examining:

1. **Component loadings**: The weights of original features in each PC
2. **Explained variance**: How much information each PC captures
3. **Feature correlation with PCs**: How original features correlate with each PC
4. **Biplots**: Visualizing both data points and feature vectors

Strategies for interpretation:
- Focus on features with the highest (absolute) loadings in each PC
- Group features with similar loadings to identify patterns
- Name components based on features that load strongly on them
- Examine data points with extreme scores on each component
- Compare loading patterns across components
- Use domain knowledge to give meaning to the mathematical relationships

Remember that interpretation becomes more difficult as the number of original features increases, and component rotation methods (like Varimax) can sometimes help with interpretability.

---

<div align="center">

## 🌟 Key Takeaways

**Principal Component Analysis:**
- Reduces dimensionality while preserving variance in the data
- Creates new uncorrelated variables (principal components) from linear combinations of original features
- Helps with visualization, noise reduction, and preprocessing for machine learning
- Works best with standardized, continuous data that has approximately linear relationships
- Is computationally efficient and deterministic compared to many alternatives
- Provides insights into the structure and patterns in high-dimensional data

**Remember:**
- Always preprocess your data appropriately before applying PCA
- Choose the number of components based on your specific analysis goals
- Consider the limitations of PCA, especially for non-linear data
- Interpret results carefully, especially in high-dimensional spaces
- Compare with other dimensionality reduction techniques for your specific task
- Use PCA as part of a comprehensive data analysis strategy, not in isolation

---

### 📖 Happy Dimensionality Reduction! 🚀

*Created by [@danialasim](https://github.com/danialasim) | Last updated: August 10, 2025*

</div>