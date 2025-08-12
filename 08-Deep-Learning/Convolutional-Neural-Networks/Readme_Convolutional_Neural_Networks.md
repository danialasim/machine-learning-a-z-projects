# 🧠 Convolutional Neural Networks (CNNs)

<div align="center">

![Model](https://img.shields.io/badge/Model-Convolutional_Neural_Networks-blue?style=for-the-badge&logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Complete-green?style=for-the-badge)
![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-yellow?style=for-the-badge)

*A Comprehensive Guide to the Architecture Revolutionizing Computer Vision*

</div>

---

## 📚 Table of Contents

- [What are Convolutional Neural Networks?](#what-are-convolutional-neural-networks)
- [Mathematical Foundation](#mathematical-foundation)
- [Architecture Components](#architecture-components)
- [Popular CNN Architectures](#popular-cnn-architectures)
- [Implementation Guide](#implementation-guide)
- [Training CNNs](#training-cnns)
- [Visualization Techniques](#visualization-techniques)
- [Transfer Learning](#transfer-learning)
- [Performance Optimization](#performance-optimization)
- [Applications](#applications)
- [Advanced Topics](#advanced-topics)
- [Challenges and Limitations](#challenges-and-limitations)
- [Tools and Frameworks](#tools-and-frameworks)
- [FAQ](#faq)

---

## 🎯 What are Convolutional Neural Networks?

Convolutional Neural Networks (CNNs) are specialized deep learning architectures designed to process data with grid-like topology, such as images. Inspired by the visual cortex of animals, CNNs use spatial relationships between elements to reduce the number of parameters and improve computational efficiency compared to fully connected networks.

### Key Characteristics:

- **Local Connectivity**: Neurons connect only to a small region of the input
- **Spatial Hierarchy**: Extract features at different levels of abstraction
- **Parameter Sharing**: Same weights apply across different locations
- **Translation Invariance**: Recognize patterns regardless of their position
- **Downsampling**: Progressively reduce spatial dimensions
- **Feature Learning**: Automatically discover relevant patterns in data
- **Multi-scale Analysis**: Capture features at multiple scales

### Historical Development:

- **1959**: Hubel and Wiesel discover oriented receptive fields in the visual cortex
- **1980**: Neocognitron introduced by Fukushima as a hierarchical neural network
- **1989**: LeCun et al. develop LeNet-5 for handwritten digit recognition
- **1998**: Gradient-based learning applied to document recognition
- **2012**: AlexNet wins ImageNet competition, sparking the CNN revolution
- **2014-2015**: VGGNet, GoogLeNet, and ResNet introduce new architectural innovations
- **2016-Present**: Specialized architectures (DenseNet, EfficientNet, etc.) achieve state-of-the-art results

### Advantages Over Traditional Neural Networks:

1. **Parameter Efficiency**: Reduced number of parameters through weight sharing
2. **Feature Hierarchy**: Automatically extract features at multiple levels
3. **Spatial Awareness**: Preserve and utilize spatial relationships
4. **Translation Invariance**: Identify objects regardless of their position
5. **Robust Feature Learning**: Less sensitive to small input transformations

---

## 🧮 Mathematical Foundation

### The Convolution Operation

The fundamental operation in CNNs is convolution, which is a mathematical operation that combines two functions to produce a third function:

$$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau$$

In the discrete case, for 2D data like images:

$$(I * K)(i, j) = \sum_{m} \sum_{n} I(i+m, j+n) K(m, n)$$

Where:
- $I$ is the input (e.g., image)
- $K$ is the kernel (filter)
- $*$ represents the convolution operation

### Convolutional Layer

The output of a convolutional layer for a single filter can be expressed as:

$$Z^{[l]}_{ijk} = \sum_{m=0}^{f_h-1} \sum_{n=0}^{f_w-1} \sum_{c=0}^{C_{in}-1} W^{[l]}_{mnc} \cdot A^{[l-1]}_{(i+m)(j+n)c} + b^{[l]}$$

Where:
- $Z^{[l]}_{ijk}$ is the output feature map at position $(i,j)$ for filter $k$ in layer $l$
- $W^{[l]}_{mnc}$ are the weights at position $(m,n)$ for input channel $c$ in layer $l$
- $A^{[l-1]}_{(i+m)(j+n)c}$ is the activation from previous layer at position $(i+m, j+n)$ for channel $c$
- $f_h$ and $f_w$ are the height and width of the filter
- $C_{in}$ is the number of input channels
- $b^{[l]}$ is the bias term

### Output Dimensions

When applying a convolution with a filter of size $f \times f$ to an input of size $n \times n$:

- Without padding, stride = 1: Output size = $(n - f + 1) \times (n - f + 1)$
- With padding $p$, stride = 1: Output size = $(n + 2p - f + 1) \times (n + 2p - f + 1)$
- With padding $p$, stride $s$: Output size = $\lfloor\frac{n + 2p - f}{s} + 1\rfloor \times \lfloor\frac{n + 2p - f}{s} + 1\rfloor$

### Pooling Operations

Max Pooling (for a $2 \times 2$ window with stride 2):

$$A^{[l]}_{ij} = \max_{0 \leq m,n \leq 1} A^{[l-1]}_{(2i+m)(2j+n)}$$

Average Pooling (for a $2 \times 2$ window with stride 2):

$$A^{[l]}_{ij} = \frac{1}{4} \sum_{m=0}^{1} \sum_{n=0}^{1} A^{[l-1]}_{(2i+m)(2j+n)}$$

### Backpropagation in CNNs

For the convolutional layer, the gradient of the loss with respect to weights is:

$$\frac{\partial \mathcal{L}}{\partial W^{[l]}_{mnc}} = \sum_{i} \sum_{j} \frac{\partial \mathcal{L}}{\partial Z^{[l]}_{ijk}} \cdot A^{[l-1]}_{(i+m)(j+n)c}$$

And for the input activations:

$$\frac{\partial \mathcal{L}}{\partial A^{[l-1]}_{ijk}} = \sum_{m} \sum_{n} \sum_{c} \frac{\partial \mathcal{L}}{\partial Z^{[l]}_{(i-m)(j-n)c}} \cdot W^{[l]}_{mnk}$$

For max pooling, gradients flow only through the maximum value in each pooling window.

---

## 🏗️ Architecture Components

### Convolutional Layer

The core building block of CNNs that applies convolution operations to the input:

```python
import tensorflow as tf
from tensorflow.keras import layers

# Creating a convolutional layer
conv_layer = layers.Conv2D(
    filters=32,           # Number of output filters
    kernel_size=(3, 3),   # Filter size
    strides=(1, 1),       # Step size
    padding='same',       # Padding strategy ('valid' or 'same')
    activation='relu',    # Activation function
    kernel_initializer='he_normal'  # Weight initialization
)
```

Key parameters:
- **Filters**: Number of feature maps (kernels)
- **Kernel Size**: Dimensions of each filter (e.g., 3×3, 5×5)
- **Stride**: Step size when sliding the filter
- **Padding**: Strategy for handling boundaries ('valid' or 'same')
- **Dilation**: Spacing between kernel elements

### Pooling Layer

Reduces spatial dimensions to decrease computational load and provide translation invariance:

```python
# Max pooling layer
max_pool = layers.MaxPooling2D(
    pool_size=(2, 2),     # Pooling window size
    strides=(2, 2),       # Step size
    padding='valid'       # Padding strategy
)

# Average pooling layer
avg_pool = layers.AveragePooling2D(
    pool_size=(2, 2),
    strides=None          # None defaults to pool_size
)
```

Types of pooling:
- **Max Pooling**: Takes maximum value in each window
- **Average Pooling**: Takes average of values in each window
- **Global Pooling**: Applies pooling operation across entire feature map

### Activation Functions

Non-linear transformations applied after convolutions:

```python
# ReLU activation
relu_activation = layers.Activation('relu')

# Leaky ReLU
leaky_relu = layers.LeakyReLU(alpha=0.1)

# Parametric ReLU
prelu = layers.PReLU()
```

Common activations in CNNs:
- **ReLU**: Most commonly used, $f(x) = \max(0, x)$
- **Leaky ReLU**: Allows small gradient for negative inputs
- **PReLU**: Trainable version of Leaky ReLU
- **ELU**: Exponential Linear Unit
- **Swish/SiLU**: $f(x) = x \cdot \sigma(x)$

### Normalization Layers

Stabilize and accelerate training:

```python
# Batch normalization
batch_norm = layers.BatchNormalization()

# Group normalization
group_norm = layers.experimental.GroupNormalization(groups=32)
```

Types of normalization:
- **Batch Normalization**: Normalizes across batch dimension
- **Layer Normalization**: Normalizes across channel dimension
- **Group Normalization**: Normalizes across groups of channels
- **Instance Normalization**: Normalizes each sample independently

### Fully Connected Layer

Processes flattened feature maps, typically for classification:

```python
# Flatten the feature maps
flatten = layers.Flatten()

# Fully connected (dense) layer
dense = layers.Dense(
    units=128,            # Number of neurons
    activation='relu',    # Activation function
    kernel_regularizer=tf.keras.regularizers.l2(0.001)  # L2 regularization
)
```

### Dropout

Regularization technique to prevent overfitting:

```python
# Dropout layer
dropout = layers.Dropout(rate=0.5)  # Drop 50% of the inputs
```

### Basic CNN Architecture

Typical CNN structure:
```
Input → [Conv → Activation → (Norm) → Pool] × N → Flatten → [FC → Activation] × M → Output
```

Where:
- N is the number of convolutional blocks
- M is the number of fully connected layers

---

## 🌟 Popular CNN Architectures

### LeNet-5 (1998)

The original CNN architecture for handwritten digit recognition:

```python
def LeNet5(input_shape=(32, 32, 1), num_classes=10):
    model = tf.keras.Sequential([
        layers.Conv2D(6, kernel_size=(5, 5), padding='valid', activation='tanh', input_shape=input_shape),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),
        layers.Conv2D(16, kernel_size=(5, 5), padding='valid', activation='tanh'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),
        layers.Flatten(),
        layers.Dense(120, activation='tanh'),
        layers.Dense(84, activation='tanh'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
```

Key characteristics:
- 7 layers (not counting input)
- Used tanh activation functions
- Average pooling instead of max pooling
- ~60K parameters

### AlexNet (2012)

The architecture that sparked the deep learning revolution:

```python
def AlexNet(input_shape=(227, 227, 3), num_classes=1000):
    model = tf.keras.Sequential([
        layers.Conv2D(96, kernel_size=(11, 11), strides=4, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(3, 3), strides=2),
        layers.BatchNormalization(),
        
        layers.Conv2D(256, kernel_size=(5, 5), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(3, 3), strides=2),
        layers.BatchNormalization(),
        
        layers.Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(3, 3), strides=2),
        
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
```

Key characteristics:
- 8 layers (5 convolutional, 3 fully connected)
- ReLU activations
- Data augmentation
- Dropout regularization
- Local Response Normalization (LRN)
- ~60M parameters

### VGG-16 (2014)

Known for its simplicity and uniform architecture:

```python
def VGG16(input_shape=(224, 224, 3), num_classes=1000):
    model = tf.keras.Sequential([
        # Block 1
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),
        
        # Block 2
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),
        
        # Block 3
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),
        
        # Block 4
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),
        
        # Block 5
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),
        
        # Classification block
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
```

Key characteristics:
- 16 layers (13 convolutional, 3 fully connected)
- Uniform 3×3 filters with stride 1
- Max pooling with 2×2 windows and stride 2
- Same padding throughout
- ~138M parameters

### GoogLeNet/Inception (2014)

Introduced the Inception module for efficient computation:

```python
def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, 
                    filters_5x5_reduce, filters_5x5, filters_pool_proj):
    # 1x1 convolution branch
    conv_1x1 = layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)
    
    # 3x3 convolution branch
    conv_3x3_reduce = layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_3x3 = layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(conv_3x3_reduce)
    
    # 5x5 convolution branch
    conv_5x5_reduce = layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_5x5 = layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(conv_5x5_reduce)
    
    # Pooling branch
    pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = layers.Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')(pool)
    
    # Concatenate all branches
    output = layers.Concatenate(axis=-1)([conv_1x1, conv_3x3, conv_5x5, pool_proj])
    
    return output

def GoogLeNet(input_shape=(224, 224, 3), num_classes=1000):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial convolutions
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # Inception modules
    x = inception_module(x, 64, 96, 128, 16, 32, 32)  # 3a
    x = inception_module(x, 128, 128, 192, 32, 96, 64)  # 3b
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = inception_module(x, 192, 96, 208, 16, 48, 64)  # 4a
    x = inception_module(x, 160, 112, 224, 24, 64, 64)  # 4b
    x = inception_module(x, 128, 128, 256, 24, 64, 64)  # 4c
    x = inception_module(x, 112, 144, 288, 32, 64, 64)  # 4d
    x = inception_module(x, 256, 160, 320, 32, 128, 128)  # 4e
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = inception_module(x, 256, 160, 320, 32, 128, 128)  # 5a
    x = inception_module(x, 384, 192, 384, 48, 128, 128)  # 5b
    
    # Classification block
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model
```

Key characteristics:
- 22 layers (with parameters)
- Inception modules with parallel convolutions
- 1×1 convolutions for dimensionality reduction
- Global average pooling instead of fully connected layers
- Auxiliary classifiers during training
- ~7M parameters

### ResNet (2015)

Introduced residual connections to enable training of very deep networks:

```python
def residual_block(x, filters, kernel_size=3, strides=1, use_conv_shortcut=False):
    shortcut = x
    
    # First convolution
    y = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    
    # Second convolution
    y = layers.Conv2D(filters, kernel_size, padding='same')(y)
    y = layers.BatchNormalization()(y)
    
    # Shortcut connection
    if use_conv_shortcut or strides > 1:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding='same')(x)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Add shortcut to output
    y = layers.add([y, shortcut])
    y = layers.Activation('relu')(y)
    
    return y

def ResNet50(input_shape=(224, 224, 3), num_classes=1000):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    # Block 1
    x = residual_block(x, 64, use_conv_shortcut=True)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    # Block 2
    x = residual_block(x, 128, strides=2, use_conv_shortcut=True)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    
    # Block 3
    x = residual_block(x, 256, strides=2, use_conv_shortcut=True)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    
    # Block 4
    x = residual_block(x, 512, strides=2, use_conv_shortcut=True)
    x = residual_block(x, 512)
    x = residual_block(x, 512)
    
    # Classification block
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model
```

Key characteristics:
- Skip connections (residual connections)
- Batch normalization after each convolution
- Global average pooling
- Variants: ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
- ~25M parameters for ResNet-50

### DenseNet (2017)

Creates dense connections between layers:

```python
def dense_block(x, blocks, growth_rate):
    for i in range(blocks):
        # Bottleneck layer
        y = layers.BatchNormalization()(x)
        y = layers.Activation('relu')(y)
        y = layers.Conv2D(4 * growth_rate, 1, padding='same')(y)
        
        # 3x3 convolution
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        y = layers.Conv2D(growth_rate, 3, padding='same')(y)
        
        # Concatenate input with output
        x = layers.Concatenate()([x, y])
    
    return x

def transition_layer(x, reduction):
    # Reduce number of feature maps
    filters = int(tf.keras.backend.int_shape(x)[-1] * reduction)
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 1, padding='same')(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    
    return x

def DenseNet121(input_shape=(224, 224, 3), num_classes=1000):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Dense blocks with transition layers
    x = dense_block(x, blocks=6, growth_rate=32)
    x = transition_layer(x, reduction=0.5)
    
    x = dense_block(x, blocks=12, growth_rate=32)
    x = transition_layer(x, reduction=0.5)
    
    x = dense_block(x, blocks=24, growth_rate=32)
    x = transition_layer(x, reduction=0.5)
    
    x = dense_block(x, blocks=16, growth_rate=32)
    
    # Classification block
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model
```

Key characteristics:
- Each layer connected to every other layer
- Growth rate controls how many features each layer adds
- Transition layers reduce feature map dimensions
- Efficient feature reuse
- ~7M parameters for DenseNet-121

### MobileNet (2017)

Designed for mobile and embedded vision applications:

```python
def depthwise_separable_conv(x, filters, stride=1):
    # Depthwise convolution
    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)  # ReLU6
    
    # Pointwise convolution
    x = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    
    return x

def MobileNetV1(input_shape=(224, 224, 3), num_classes=1000, alpha=1.0):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(int(32 * alpha), kernel_size=3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    
    # Depthwise separable convolutions
    x = depthwise_separable_conv(x, int(64 * alpha))
    x = depthwise_separable_conv(x, int(128 * alpha), stride=2)
    x = depthwise_separable_conv(x, int(128 * alpha))
    x = depthwise_separable_conv(x, int(256 * alpha), stride=2)
    x = depthwise_separable_conv(x, int(256 * alpha))
    x = depthwise_separable_conv(x, int(512 * alpha), stride=2)
    
    # 5 blocks of depthwise separable convolutions
    for _ in range(5):
        x = depthwise_separable_conv(x, int(512 * alpha))
    
    x = depthwise_separable_conv(x, int(1024 * alpha), stride=2)
    x = depthwise_separable_conv(x, int(1024 * alpha))
    
    # Classification block
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model
```

Key characteristics:
- Depthwise separable convolutions
- Width multiplier (alpha) to control model size
- ReLU6 activation
- ~4.2M parameters for alpha=1.0
- MobileNetV2 introduced inverted residuals and linear bottlenecks

### EfficientNet (2019)

Introduced compound scaling for efficient deep networks:

```python
def mbconv_block(x, expand_ratio, filters, kernel_size, strides, se_ratio=0.25, drop_rate=0.2):
    # Input
    inputs = x
    in_channels = inputs.shape[-1]
    
    # Expansion
    if expand_ratio != 1:
        x = layers.Conv2D(in_channels * expand_ratio, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)
    
    # Depthwise convolution
    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    # Squeeze and Excitation
    if se_ratio:
        se = layers.GlobalAveragePooling2D()(x)
        se_channels = max(1, int(in_channels * expand_ratio * se_ratio))
        se = layers.Dense(se_channels, activation='swish')(se)
        se = layers.Dense(in_channels * expand_ratio, activation='sigmoid')(se)
        se = layers.Reshape((1, 1, in_channels * expand_ratio))(se)
        x = layers.Multiply()([x, se])
    
    # Pointwise convolution
    x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    # Skip connection
    if strides == 1 and in_channels == filters:
        if drop_rate:
            x = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1))(x)
        x = layers.Add()([inputs, x])
    
    return x

# This is a simplified version of EfficientNetB0
def EfficientNetB0(input_shape=(224, 224, 3), num_classes=1000):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    # MBConv blocks
    # Block 1
    x = mbconv_block(x, expand_ratio=1, filters=16, kernel_size=3, strides=1)
    
    # Block 2
    x = mbconv_block(x, expand_ratio=6, filters=24, kernel_size=3, strides=2)
    x = mbconv_block(x, expand_ratio=6, filters=24, kernel_size=3, strides=1)
    
    # Block 3
    x = mbconv_block(x, expand_ratio=6, filters=40, kernel_size=5, strides=2)
    x = mbconv_block(x, expand_ratio=6, filters=40, kernel_size=5, strides=1)
    
    # Block 4
    x = mbconv_block(x, expand_ratio=6, filters=80, kernel_size=3, strides=2)
    x = mbconv_block(x, expand_ratio=6, filters=80, kernel_size=3, strides=1)
    x = mbconv_block(x, expand_ratio=6, filters=80, kernel_size=3, strides=1)
    
    # Block 5
    x = mbconv_block(x, expand_ratio=6, filters=112, kernel_size=5, strides=1)
    x = mbconv_block(x, expand_ratio=6, filters=112, kernel_size=5, strides=1)
    x = mbconv_block(x, expand_ratio=6, filters=112, kernel_size=5, strides=1)
    
    # Block 6
    x = mbconv_block(x, expand_ratio=6, filters=192, kernel_size=5, strides=2)
    x = mbconv_block(x, expand_ratio=6, filters=192, kernel_size=5, strides=1)
    x = mbconv_block(x, expand_ratio=6, filters=192, kernel_size=5, strides=1)
    x = mbconv_block(x, expand_ratio=6, filters=192, kernel_size=5, strides=1)
    
    # Block 7
    x = mbconv_block(x, expand_ratio=6, filters=320, kernel_size=3, strides=1)
    
    # Final convolution
    x = layers.Conv2D(1280, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    # Classification block
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model
```

Key characteristics:
- Mobile Inverted Bottleneck Convolution (MBConv) blocks
- Squeeze-and-Excitation modules
- Compound scaling (depth, width, resolution)
- Swish activation function
- Variants: B0 to B7 with increasing scale
- ~5.3M parameters for EfficientNetB0

---

## 💻 Implementation Guide

### Basic CNN Implementation with TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# One-hot encode the labels
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Define CNN architecture
def create_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Classification block
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Create and compile model
model = create_cnn_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_cnn_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Train the model with data augmentation
history = model.fit(
    tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        .shuffle(10000)
        .batch(128)
        .map(lambda x, y: (data_augmentation(x, training=True), y))
        .prefetch(tf.data.AUTOTUNE),
    epochs=50,
    validation_data=(test_images, test_labels),
    callbacks=callbacks
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.tight_layout()
plt.show()

# Visualize predictions
def visualize_predictions(model, images, labels, class_names, num_images=5):
    # Make predictions
    predictions = model.predict(images[:num_images])
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels[:num_images], axis=1)
    
    plt.figure(figsize=(12, 2 * num_images))
    for i in range(num_images):
        plt.subplot(num_images, 2, 2*i+1)
        plt.imshow(images[i])
        plt.title(f"True: {class_names[true_classes[i]]}")
        plt.axis('off')
        
        plt.subplot(num_images, 2, 2*i+2)
        plt.bar(range(len(class_names)), predictions[i])
        plt.xticks(range(len(class_names)), class_names, rotation=90)
        plt.title(f"Predicted: {class_names[predicted_classes[i]]}")
    
    plt.tight_layout()
    plt.show()

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Visualize some predictions
visualize_predictions(model, test_images, test_labels, class_names)
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# Define CNN model
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4),
        )
        
        # Classification block
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Create model
model = CNN().to(device)
print(model)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# Training function
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 99:
            print(f'Epoch: {epoch} | Batch: {batch_idx+1} | Loss: {running_loss/100:.3f} | Acc: {100.*correct/total:.2f}%')
            running_loss = 0.0
    
    return running_loss / len(train_loader), correct / total

# Testing function
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Statistics
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = correct / total
    
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {100.*accuracy:.2f}%')
    
    return test_loss, accuracy

# Training loop
epochs = 50
train_losses = []
train_accs = []
test_losses = []
test_accs = []
best_acc = 0.0

for epoch in range(1, epochs + 1):
    train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
    test_loss, test_acc = test(model, device, test_loader, criterion)
    
    # Update learning rate
    scheduler.step(test_loss)
    
    # Save statistics
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_cnn_model.pth')
        print(f'Model saved with accuracy: {100.*best_acc:.2f}%')

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Training Accuracy')
plt.plot(test_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.tight_layout()
plt.show()
```

### Custom Layers and Operations

Here's how to implement custom CNN operations:

```python
import tensorflow as tf
from tensorflow.keras import layers, backend as K

# Custom convolution layer with adjustable dilation rate
class DilatedConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super(DilatedConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.kwargs = kwargs
    
    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            **self.kwargs
        )
    
    def call(self, inputs):
        return self.conv(inputs)

# Custom spatial attention module
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
    
    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding='same',
            activation='sigmoid'
        )
    
    def call(self, inputs):
        # Average pooling across channels
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        # Max pooling across channels
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        # Concatenate pooled features
        pooled = tf.concat([avg_pool, max_pool], axis=-1)
        # Apply convolution to generate attention map
        attention = self.conv(pooled)
        # Apply attention map to input feature maps
        return inputs * attention

# Custom residual block with pre-activation
class PreActResidualBlock(layers.Layer):
    def __init__(self, filters, strides=1, **kwargs):
        super(PreActResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.strides = strides
    
    def build(self, input_shape):
        input_channels = input_shape[-1]
        
        # Pre-activation layers
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')
        
        # First convolution
        self.conv1 = layers.Conv2D(self.filters, 3, strides=self.strides, padding='same')
        
        # Second pre-activation
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('relu')
        
        # Second convolution
        self.conv2 = layers.Conv2D(self.filters, 3, padding='same')
        
        # Shortcut connection
        self.shortcut = layers.Lambda(lambda x: x)
        if self.strides > 1 or input_channels != self.filters:
            self.shortcut = layers.Conv2D(self.filters, 1, strides=self.strides, padding='same')
    
    def call(self, inputs):
        # Pre-activation
        x = self.bn1(inputs)
        x = self.relu1(x)
        
        # Shortcut should come after pre-activation
        shortcut = self.shortcut(x)
        
        # First convolution
        x = self.conv1(x)
        
        # Second pre-activation
        x = self.bn2(x)
        x = self.relu2(x)
        
        # Second convolution
        x = self.conv2(x)
        
        # Add shortcut
        x = layers.add([x, shortcut])
        
        return x

# Custom Squeeze-and-Excitation block
class SqueezeExcitation(layers.Layer):
    def __init__(self, ratio=16):
        super(SqueezeExcitation, self).__init__()
        self.ratio = ratio
    
    def build(self, input_shape):
        channels = input_shape[-1]
        self.global_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(channels // self.ratio, activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')
        self.reshape = layers.Reshape((1, 1, channels))
    
    def call(self, inputs):
        x = self.global_pool(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.reshape(x)
        return inputs * x

# Example of using custom layers in a model
def custom_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = PreActResidualBlock(64)(x)
    x = PreActResidualBlock(64)(x)
    
    # Dilated convolution
    x = DilatedConv2D(128, 3, dilation_rate=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Attention mechanism
    x = SpatialAttention()(x)
    
    # Squeeze and Excitation
    x = SqueezeExcitation()(x)
    
    # Classification
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)
```

---

## 🔄 Training CNNs

### Data Preparation and Augmentation

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,                     # Normalize pixel values
    rotation_range=20,                  # Randomly rotate images
    width_shift_range=0.2,              # Randomly shift horizontally
    height_shift_range=0.2,             # Randomly shift vertically
    shear_range=0.2,                    # Shear transformations
    zoom_range=0.2,                     # Zoom transformations
    horizontal_flip=True,               # Randomly flip horizontally
    fill_mode='nearest',                # Fill strategy for created pixels
    brightness_range=[0.8, 1.2],        # Random brightness adjustment
    channel_shift_range=0.2             # Random channel shifts
)

# Only rescaling for validation
validation_datagen = ImageDataGenerator(rescale=1./255)

# Alternative with tf.data pipeline
def prepare_dataset(images, labels, batch_size=32, is_training=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if is_training:
        # Cache the dataset for better performance
        dataset = dataset.cache()
        # Shuffle with a buffer size equal to dataset size
        dataset = dataset.shuffle(buffer_size=len(images))
        
        # Define augmentation function
        def augment(image, label):
            # Cast image to float32
            image = tf.cast(image, tf.float32)
            # Random flip
            image = tf.image.random_flip_left_right(image)
            # Random brightness
            image = tf.image.random_brightness(image, max_delta=0.2)
            # Random contrast
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            # Normalize
            image = image / 255.0
            return image, label
        
        # Apply augmentation
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # Simple normalization for validation/test
        dataset = dataset.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Visualize augmented images
def visualize_augmentation(datagen, images, num_rows=2, num_cols=5):
    # Get a batch of images
    image_batch = images[:1]  # Take one image
    
    # Create an iterator
    aug_iterator = datagen.flow(image_batch, batch_size=1)
    
    # Plot original and augmented images
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(num_rows, num_cols, 1)
    plt.imshow(image_batch[0].astype('uint8'))
    plt.title('Original')
    plt.axis('off')
    
    # Augmented versions
    for i in range(1, num_rows * num_cols):
        aug_img = next(aug_iterator)[0].astype('uint8')
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(aug_img)
        plt.title(f'Augmented {i}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
```

### Learning Rate Schedules

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Constant learning rate
def constant_schedule(epoch, lr):
    return 0.001

# Step decay learning rate
def step_decay(epoch, lr):
    drop_rate = 0.5
    epochs_drop = 10.0
    return 0.001 * (drop_rate ** np.floor((1 + epoch) / epochs_drop))

# Exponential decay learning rate
def exponential_decay(epoch, lr):
    k = 0.1
    return 0.001 * np.exp(-k * epoch)

# Cosine annealing learning rate
def cosine_annealing(epoch, lr):
    epochs = 50
    return 0.001 * (1 + np.cos(np.pi * epoch / epochs)) / 2

# Warm restarts
def warm_restarts(epoch, lr):
    max_lr = 0.001
    min_lr = 0.0001
    cycle_length = 10
    
    # Calculate where in the cycle we are
    cycle = np.floor(1 + epoch / cycle_length)
    x = epoch - (cycle - 1) * cycle_length
    
    # Calculate learning rate
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(x * np.pi / cycle_length))

# One-cycle learning rate
def one_cycle(epoch, lr, epochs=50, max_lr=0.005, min_lr=0.0001):
    # First half: linear increase from min_lr to max_lr
    # Second half: cosine decay from max_lr to min_lr/10
    if epoch < epochs / 2:
        return min_lr + (max_lr - min_lr) * (epoch / (epochs / 2))
    else:
        return min_lr/10 + (max_lr - min_lr/10) * (1 + np.cos(np.pi * (epoch - epochs/2) / (epochs/2))) / 2

# Visualize learning rate schedules
def plot_lr_schedules(epochs=50):
    lr_schedules = {
        'Constant': constant_schedule,
        'Step Decay': step_decay,
        'Exponential Decay': exponential_decay,
        'Cosine Annealing': cosine_annealing,
        'Warm Restarts': warm_restarts,
        'One Cycle': one_cycle
    }
    
    plt.figure(figsize=(12, 6))
    
    for name, schedule in lr_schedules.items():
        lrs = [schedule(epoch, 0) for epoch in range(epochs)]
        plt.plot(lrs, label=name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedules')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Create callbacks for different schedules
def get_lr_callback(schedule_type):
    if schedule_type == 'constant':
        return tf.keras.callbacks.LearningRateScheduler(constant_schedule)
    elif schedule_type == 'step':
        return tf.keras.callbacks.LearningRateScheduler(step_decay)
    elif schedule_type == 'exponential':
        return tf.keras.callbacks.LearningRateScheduler(exponential_decay)
    elif schedule_type == 'cosine':
        return tf.keras.callbacks.LearningRateScheduler(cosine_annealing)
    elif schedule_type == 'restarts':
        return tf.keras.callbacks.LearningRateScheduler(warm_restarts)
    elif schedule_type == 'one_cycle':
        return tf.keras.callbacks.LearningRateScheduler(one_cycle)
    elif schedule_type == 'reduce_on_plateau':
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
```

### Fine-tuning

Fine-tuning pre-trained CNN models allows you to leverage knowledge from models trained on large datasets:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2

def create_transfer_learning_model(base_model_name, input_shape=(224, 224, 3), num_classes=10, trainable_layers=0):
    """
    Create a transfer learning model based on a pre-trained CNN.
    
    Parameters:
    -----------
    base_model_name : str
        Name of the base model ('resnet50', 'vgg16', or 'mobilenetv2')
    input_shape : tuple
        Input shape (height, width, channels)
    num_classes : int
        Number of output classes
    trainable_layers : int
        Number of layers to make trainable from the top
        
    Returns:
    --------
    tf.keras.Model
        Transfer learning model
    """
    # Create base model
    if base_model_name.lower() == 'resnet50':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name.lower() == 'vgg16':
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name.lower() == 'mobilenetv2':
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Freeze all layers in the base model
    base_model.trainable = False
    
    # If trainable_layers > 0, unfreeze the specified number of layers from the top
    if trainable_layers > 0:
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True
    
    # Create model
    inputs = tf.keras.Input(shape=input_shape)
    
    # Preprocess input according to the base model
    if base_model_name.lower() == 'resnet50':
        x = tf.keras.applications.resnet50.preprocess_input(inputs)
    elif base_model_name.lower() == 'vgg16':
        x = tf.keras.applications.vgg16.preprocess_input(inputs)
    elif base_model_name.lower() == 'mobilenetv2':
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Feed through base model
    x = base_model(x, training=False)
    
    # Add classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model

# Example usage
def fine_tuning_example():
    # Create model
    model = create_transfer_learning_model(
        base_model_name='mobilenetv2',
        input_shape=(224, 224, 3),
        num_classes=10,
        trainable_layers=10  # Fine-tune the top 10 layers
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Summary of trainable layers
    print("Trainable layers:")
    for i, layer in enumerate(model.layers):
        if layer.trainable:
            print(f"{i}: {layer.name}")
    
    # Count parameters
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Total parameters: {trainable_params + non_trainable_params:,}")
    
    return model

# Progressive fine-tuning approach
def progressive_fine_tuning(model, train_dataset, validation_dataset, initial_epochs=10, fine_tuning_epochs=20):
    """
    Perform progressive fine-tuning: first train only the top layers, then fine-tune more layers.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Pre-trained model with frozen base
    train_dataset : tf.data.Dataset
        Training dataset
    validation_dataset : tf.data.Dataset
        Validation dataset
    initial_epochs : int
        Number of epochs for initial training
    fine_tuning_epochs : int
        Number of epochs for fine-tuning
        
    Returns:
    --------
    model : tf.keras.Model
        Fine-tuned model
    history : dict
        Combined training history
    """
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Phase 1: Train only the top layers (head)
    print("Phase 1: Training only the top layers...")
    history1 = model.fit(
        train_dataset,
        epochs=initial_epochs,
        validation_data=validation_dataset,
        callbacks=callbacks
    )
    
    # Phase 2: Unfreeze more layers and continue training
    print("Phase 2: Fine-tuning more layers...")
    
    # Unfreeze the base model for fine-tuning
    base_model = model.layers[1]  # Assuming the base model is the second layer
    base_model.trainable = True
    
    # Recompile model with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
    history2 = model.fit(
        train_dataset,
        epochs=initial_epochs + fine_tuning_epochs,
        initial_epoch=initial_epochs,
        validation_data=validation_dataset,
        callbacks=callbacks
    )
    
    # Combine histories
    history = {}
    history['loss'] = history1.history['loss'] + history2.history['loss']
    history['val_loss'] = history1.history['val_loss'] + history2.history['val_loss']
    history['accuracy'] = history1.history['accuracy'] + history2.history['accuracy']
    history['val_accuracy'] = history1.history['val_accuracy'] + history2.history['val_accuracy']
    
    return model, history
```

### Different Fine-tuning Strategies

1. **Feature Extraction**
   - Freeze the entire pre-trained network
   - Replace and train only the classifier head
   - Fast and requires less data

2. **Fine-tuning the Top Layers**
   - Freeze most of the base model
   - Unfreeze and train the top few layers along with the classifier head
   - Balance between adaptation and overfitting

3. **Full Fine-tuning**
   - Unfreeze the entire network
   - Train all layers with a very small learning rate
   - Requires more data and computational resources
   - Risk of overfitting on small datasets

4. **Progressive Fine-tuning**
   - Start with feature extraction (frozen base)
   - Gradually unfreeze and train deeper layers
   - Often yields the best results

5. **Discriminative Fine-tuning**
   - Apply different learning rates to different layers
   - Lower learning rates for earlier layers, higher for later layers
   - Preserves low-level features while adapting high-level features

```python
# Discriminative fine-tuning example
def create_discriminative_fine_tuning_model(base_model_name='resnet50', num_classes=10):
    # Load pre-trained model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Create model
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    
    # Enable training for all layers
    base_model.trainable = True
    
    # Group layers for different learning rates
    layer_groups = []
    
    # Group 1: First layers (lowest learning rate)
    layer_groups.append(base_model.layers[:50])
    
    # Group 2: Middle layers
    layer_groups.append(base_model.layers[50:100])
    
    # Group 3: Final layers (highest learning rate)
    layer_groups.append(base_model.layers[100:] + [model.layers[-3], model.layers[-2]])
    
    # Group 4: Output layer
    layer_groups.append([model.layers[-1]])
    
    # Define learning rates for each group
    learning_rates = [1e-6, 1e-5, 1e-4, 1e-3]
    
    # Create optimizers for each group
    optimizers = []
    for lr in learning_rates:
        optimizers.append(tf.keras.optimizers.Adam(learning_rate=lr))
    
    return model, layer_groups, optimizers

# Custom training loop for discriminative fine-tuning
def train_with_discriminative_learning_rates(model, layer_groups, optimizers, train_dataset, val_dataset, epochs=10):
    # Loss function
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    # Metrics
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    
    # Training history
    history = {
        'loss': [],
        'val_loss': [],
        'accuracy': [],
        'val_accuracy': []
    }
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training phase
        train_loss = 0.0
        num_batches = 0
        
        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                # Forward pass
                logits = model(x_batch, training=True)
                # Calculate loss
                loss_value = loss_fn(y_batch, logits)
            
            # Get gradients
            gradients = tape.gradient(loss_value, model.trainable_weights)
            
            # Apply gradients with different optimizers for different layer groups
            gradient_groups = []
            weight_groups = []
            
            # Group gradients and weights
            for group in layer_groups:
                group_weights = []
                for layer in group:
                    group_weights.extend(layer.trainable_weights)
                weight_groups.append(group_weights)
            
            for i, weights in enumerate(weight_groups):
                if not weights:
                    continue
                
                group_grads = [g for g, w in zip(gradients, model.trainable_weights) if w in weights]
                optimizers[i].apply_gradients(zip(group_grads, weights))
            
            # Update metrics
            train_acc_metric.update_state(y_batch, logits)
            train_loss += loss_value
            num_batches += 1
        
        # Calculate training metrics
        train_acc = train_acc_metric.result().numpy()
        train_loss = (train_loss / num_batches).numpy()
        
        # Reset metrics
        train_acc_metric.reset_states()
        
        # Validation phase
        val_loss = 0.0
        num_val_batches = 0
        
        for x_batch, y_batch in val_dataset:
            # Forward pass
            val_logits = model(x_batch, training=False)
            # Calculate loss
            val_loss_value = loss_fn(y_batch, val_logits)
            
            # Update metrics
            val_acc_metric.update_state(y_batch, val_logits)
            val_loss += val_loss_value
            num_val_batches += 1
        
        # Calculate validation metrics
        val_acc = val_acc_metric.result().numpy()
        val_loss = (val_loss / num_val_batches).numpy()
        
        # Reset metrics
        val_acc_metric.reset_states()
        
        # Print metrics
        print(f"loss: {train_loss:.4f}, accuracy: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_accuracy: {val_acc:.4f}")
        
        # Update history
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
    
    return history
```

---

## 👁️ Visualization Techniques

Visualizing CNNs helps understand what they learn and how they make decisions:

### Feature Maps Visualization

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def visualize_feature_maps(model, image, layer_name=None):
    """
    Visualize feature maps of a CNN for a given image.
    
    Parameters:
    -----------
    model : tf.keras.Model
        CNN model
    image : numpy.ndarray
        Input image (should match model's input shape)
    layer_name : str, optional
        Name of the layer to visualize. If None, use the first convolutional layer.
    """
    # Prepare the image
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Ensure the image has the correct shape
    input_shape = model.input_shape[1:3]
    if image.shape[1:3] != input_shape:
        image = tf.image.resize(image, input_shape).numpy()
    
    # Create a model that outputs the feature maps
    if layer_name:
        layer = model.get_layer(layer_name)
    else:
        # Find the first convolutional layer
        for layer in model.layers:
            if 'conv' in layer.name.lower():
                break
    
    # Create a feature map model
    feature_map_model = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
    
    # Get feature maps
    feature_maps = feature_map_model.predict(image)
    
    # Plot feature maps
    feature_maps = feature_maps[0]  # Remove batch dimension
    n_features = feature_maps.shape[-1]
    
    # Calculate grid size
    size = int(np.ceil(np.sqrt(n_features)))
    
    plt.figure(figsize=(20, 15))
    
    # Plot original image
    plt.subplot(size, size, 1)
    plt.imshow(image[0].astype('uint8'))
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot feature maps
    for i in range(n_features):
        plt.subplot(size, size, i + 2)
        plt.imshow(feature_maps[:, :, i], cmap='viridis')
        plt.title(f'Feature Map {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'Feature Maps for Layer: {layer.name}')
    plt.subplots_adjust(top=0.9)
    plt.show()
```

### Filter Visualization

```python
def visualize_filters(model, layer_name=None):
    """
    Visualize filters of a convolutional layer.
    
    Parameters:
    -----------
    model : tf.keras.Model
        CNN model
    layer_name : str, optional
        Name of the convolutional layer to visualize.
        If None, use the first convolutional layer.
    """
    # Get convolutional layer
    if layer_name:
        layer = model.get_layer(layer_name)
    else:
        # Find the first convolutional layer
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                break
    
    # Get filters
    filters = layer.get_weights()[0]
    
    # Number of filters and their dimensions
    n_filters, filter_width, filter_height, n_channels = filters.shape
    
    # Calculate grid size
    size = int(np.ceil(np.sqrt(n_filters)))
    
    plt.figure(figsize=(20, 15))
    
    # Plot filters
    for i in range(n_filters):
        filter_img = filters[i]
        
        # Normalize filter values to [0, 1]
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
        
        plt.subplot(size, size, i + 1)
        
        if n_channels == 3:
            # RGB filter
            plt.imshow(filter_img)
        else:
            # Grayscale filter (take mean across channels)
            plt.imshow(np.mean(filter_img, axis=2), cmap='gray')
        
        plt.title(f'Filter {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'Filters for Layer: {layer.name}')
    plt.subplots_adjust(top=0.9)
    plt.show()
```

### Class Activation Maps (CAM)

```python
def compute_class_activation_map(model, img, class_idx, layer_name=None):
    """
    Compute Class Activation Map for a specific class.
    
    Parameters:
    -----------
    model : tf.keras.Model
        CNN model with global average pooling
    img : numpy.ndarray
        Input image
    class_idx : int
        Index of the class to visualize
    layer_name : str, optional
        Name of the final convolutional layer
        
    Returns:
    --------
    numpy.ndarray
        Class activation map
    """
    # Prepare the image
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    
    # Find the last convolutional layer
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
    
    # Get the gradients of the predicted class with respect to the output feature map
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[
            model.get_layer(layer_name).output,
            model.output
        ]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, class_idx]
    
    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Global average pooling of the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the feature maps with the gradient values
    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    
    # Average across channels
    cam = np.mean(conv_outputs, axis=-1)
    
    # Normalize CAM
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)
    
    return cam

def visualize_cam(model, img, class_idx, layer_name=None, alpha=0.5):
    """
    Visualize Class Activation Map overlaid on the input image.
    
    Parameters:
    -----------
    model : tf.keras.Model
        CNN model
    img : numpy.ndarray
        Input image
    class_idx : int
        Index of the class to visualize
    layer_name : str, optional
        Name of the final convolutional layer
    alpha : float, default=0.5
        Transparency of the heatmap overlay
    """
    # Compute CAM
    cam = compute_class_activation_map(model, img, class_idx, layer_name)
    
    # Resize CAM to match input image size
    cam_resized = tf.image.resize(
        np.expand_dims(cam, axis=-1),
        (img.shape[1], img.shape[2])
    ).numpy()
    cam_resized = np.squeeze(cam_resized)
    
    # Convert to heatmap
    import cv2
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superimpose heatmap on original image
    if len(img.shape) == 4:
        img = img[0]  # Remove batch dimension
    
    superimposed_img = heatmap * alpha + img
    superimposed_img = superimposed_img / np.max(superimposed_img)
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cam_resized, cmap='jet')
    plt.title('Class Activation Map')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title('CAM Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
```

### Guided Backpropagation

```python
@tf.custom_gradient
def guided_relu(x):
    """Custom gradient for guided backpropagation."""
    def grad(dy):
        return tf.cast(dy > 0, tf.float32) * tf.cast(x > 0, tf.float32) * dy
    return tf.nn.relu(x), grad

class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.guided_model = self.build_guided_model()
    
    def build_guided_model(self):
        """Replace ReLU activations with guided ReLU for backpropagation."""
        model_copy = tf.keras.models.clone_model(self.model)
        model_copy.set_weights(self.model.get_weights())
        
        # Create a new model with the same architecture but guided ReLU
        layer_dict = {}
        for layer in model_copy.layers:
            if isinstance(layer, tf.keras.layers.ReLU):
                layer_dict[layer.name] = lambda x: guided_relu(x)
        
        return tf.keras.models.Model(
            inputs=model_copy.inputs,
            outputs=[model_copy.outputs, 
                     tf.gradients(model_copy.outputs[0][:, tf.argmax(model_copy.outputs[0])], 
                                 model_copy.inputs)[0]]
        )
    
    def guided_backprop(self, image):
        """Compute guided backpropagation for an image."""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Forward and backward pass
        with tf.GradientTape() as tape:
            inputs = tf.convert_to_tensor(image, dtype=tf.float32)
            tape.watch(inputs)
            outputs, grads = self.guided_model(inputs)
        
        # Convert gradients to numpy
        if grads is None:
            grads = np.zeros_like(image)
        else:
            grads = grads.numpy()
        
        # Process gradients
        grads = np.maximum(grads, 0)
        grads = (grads - np.min(grads)) / (np.max(grads) - np.min(grads) + 1e-8)
        
        return grads

def visualize_guided_backprop(model, image, class_idx=None):
    """
    Visualize guided backpropagation for an image.
    
    Parameters:
    -----------
    model : tf.keras.Model
        CNN model
    image : numpy.ndarray
        Input image
    class_idx : int, optional
        Class index to visualize. If None, use the predicted class.
    """
    # Prepare the image
    if len(image.shape) == 3:
        image_display = image.copy()
        image = np.expand_dims(image, axis=0)
    else:
        image_display = image[0].copy()
    
    # Get the predicted class if not provided
    if class_idx is None:
        preds = model.predict(image)
        class_idx = np.argmax(preds[0])
    
    # Create guided backprop model
    gb_model = GuidedBackprop(model)
    gb = gb_model.guided_backprop(image)
    
    # Remove batch dimension
    gb = gb[0]
    
    # Visualize guided backpropagation
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_display)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.sum(gb, axis=-1), cmap='viridis')
    plt.title('Guided Backpropagation')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    gb_rgb = np.abs(gb)
    gb_rgb = gb_rgb / np.max(gb_rgb)
    plt.imshow(gb_rgb)
    plt.title('Guided Backpropagation (RGB)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
```

### t-SNE Visualization of Feature Embeddings

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def visualize_embeddings(model, images, labels, layer_name=None, perplexity=30, n_iter=1000):
    """
    Visualize embeddings of a CNN layer using t-SNE.
    
    Parameters:
    -----------
    model : tf.keras.Model
        CNN model
    images : numpy.ndarray
        Input images
    labels : numpy.ndarray
        Labels for each image
    layer_name : str, optional
        Name of the layer to extract embeddings from.
        If None, use the layer before the final dense layer.
    perplexity : int, default=30
        Perplexity parameter for t-SNE
    n_iter : int, default=1000
        Number of iterations for t-SNE
    """
    # Find the layer to extract embeddings from
    if layer_name is None:
        # Find the last layer before the final dense layer
        for i in range(len(model.layers) - 1, 0, -1):
            if not isinstance(model.layers[i], tf.keras.layers.Dense):
                layer_name = model.layers[i].name
                break
    
    # Create a model that outputs the embeddings
    embedding_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(layer_name).output)
    
    # Extract embeddings
    embeddings = embedding_model.predict(images)
    
    # Flatten embeddings if needed
    if len(embeddings.shape) > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    
    # Reduce dimensionality with t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot t-SNE visualization
    plt.figure(figsize=(12, 10))
    
    # Create a scatter plot colored by class
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=f'Class {label}',
            alpha=0.6
        )
    
    plt.title(f't-SNE Visualization of {layer_name} Embeddings')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

---

## 🔄 Transfer Learning

Transfer learning is a powerful technique for applying CNNs to new tasks with limited data:

### Types of Transfer Learning

1. **Feature Extraction**: Use pre-trained CNN as a fixed feature extractor
2. **Fine-tuning**: Adapt a pre-trained CNN by updating some or all weights
3. **Domain Adaptation**: Adapt models across different but related domains
4. **Knowledge Distillation**: Train a smaller network to mimic a larger one

### Transfer Learning Workflow

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2, EfficientNetB0
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

def create_transfer_learning_model(base_model_name='resnet50', input_shape=(224, 224, 3), 
                                  num_classes=10, dropout_rate=0.5, trainable=False):
    """
    Create a transfer learning model using a pre-trained CNN.
    
    Parameters:
    -----------
    base_model_name : str
        Name of the base model ('resnet50', 'vgg16', 'mobilenetv2', 'efficientnet')
    input_shape : tuple
        Input shape (height, width, channels)
    num_classes : int
        Number of output classes
    dropout_rate : float
        Dropout rate for regularization
    trainable : bool
        Whether to make the base model trainable
        
    Returns:
    --------
    tf.keras.Model
        Transfer learning model
    """
    # Load pre-trained base model
    if base_model_name.lower() == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input = tf.keras.applications.resnet50.preprocess_input
    elif base_model_name.lower() == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input = tf.keras.applications.vgg16.preprocess_input
    elif base_model_name.lower() == 'mobilenetv2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    elif base_model_name.lower() == 'efficientnet':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Freeze/unfreeze base model
    base_model.trainable = trainable
    
    # Create model
    inputs = tf.keras.Input(shape=input_shape)
    
    # Preprocess input
    x = preprocess_input(inputs)
    
    # Pass through base model
    x = base_model(x, training=False)
    
    # Add classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Add a dense layer if using a deep network like ResNet or EfficientNet
    if base_model_name.lower() in ['resnet50', 'efficientnet']:
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model

def train_transfer_learning(model, train_dataset, validation_dataset, learning_rate=0.001, epochs=20):
    """
    Train a transfer learning model.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Transfer learning model
    train_dataset : tf.data.Dataset
        Training dataset
    validation_dataset : tf.data.Dataset
        Validation dataset
    learning_rate : float
        Initial learning rate
    epochs : int
        Number of training epochs
        
    Returns:
    --------
    history : dict
        Training history
    """
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=callbacks
    )
    
    return history

def load_and_preprocess_data(dataset_name='cifar10', img_size=(224, 224), batch_size=32):
    """
    Load and preprocess a dataset for transfer learning.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset ('cifar10' or 'cifar100')
    img_size : tuple
        Target image size for resizing
    batch_size : int
        Batch size for training
        
    Returns:
    --------
    tuple
        (train_dataset, validation_dataset, test_dataset, num_classes)
    """
    # Load dataset
    if dataset_name.lower() == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        num_classes = 10
    elif dataset_name.lower() == 'cifar100':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    # Split training data into train and validation
    val_size = int(0.1 * len(x_train))
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]
    
    # Define data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ])
    
    # Create training dataset with augmentation
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(x_train))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(
        lambda x, y: (tf.image.resize(x, img_size), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Create validation dataset
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    validation_dataset = validation_dataset.batch(batch_size)
    validation_dataset = validation_dataset.map(
        lambda x, y: (tf.image.resize(x, img_size), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Create test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.map(
        lambda x, y: (tf.image.resize(x, img_size), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, validation_dataset, test_dataset, num_classes

def compare_transfer_learning_models():
    """
    Compare different transfer learning approaches on a dataset.
    """
    # Load and preprocess data
    train_dataset, validation_dataset, test_dataset, num_classes = load_and_preprocess_data(
        dataset_name='cifar10',
        img_size=(224, 224),
        batch_size=32
    )
    
    # Define models to compare
    models_to_compare = [
        ('ResNet50 (Feature Extraction)', 'resnet50', False),
        ('ResNet50 (Fine-tuning)', 'resnet50', True),
        ('MobileNetV2 (Feature Extraction)', 'mobilenetv2', False),
        ('MobileNetV2 (Fine-tuning)', 'mobilenetv2', True),
    ]
    
    # Results storage
    results = {}
    
    for name, base_model, trainable in models_to_compare:
        print(f"\nTraining {name}...")
        
        # Create model
        model = create_transfer_learning_model(
            base_model_name=base_model,
            input_shape=(224, 224, 3),
            num_classes=num_classes,
            trainable=trainable
        )
        
        # Train model
        learning_rate = 0.0001 if trainable else 0.001  # Lower learning rate for fine-tuning
        history = train_transfer_learning(
            model,
            train_dataset,
            validation_dataset,
            learning_rate=learning_rate,
            epochs=20
        )
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(test_dataset)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Store results
        results[name] = {
            'model': model,
            'history': history.history,
            'test_accuracy': test_accuracy
        }
    
    # Plot accuracy comparison
    plt.figure(figsize=(15, 5))
    
    # Training curves
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        plt.plot(result['history']['accuracy'], label=f"{name} (Train)")
        plt.plot(result['history']['val_accuracy'], label=f"{name} (Val)", linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Final test accuracy
    plt.subplot(1, 2, 2)
    model_names = list(results.keys())
    test_accuracies = [results[name]['test_accuracy'] for name in model_names]
    
    plt.bar(model_names, test_accuracies, color=['blue', 'orange', 'green', 'red'])
    plt.xlabel('Model')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Add accuracy values on bars
    for i, v in enumerate(test_accuracies):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return results
```

### Transfer Learning Strategies

```python
def progressive_transfer_learning(base_model_name='resnet50', input_shape=(224, 224, 3), 
                                 num_classes=10, unfreeze_strategy='top_layers'):
    """
    Implement progressive transfer learning with different unfreezing strategies.
    
    Parameters:
    -----------
    base_model_name : str
        Name of the base model
    input_shape : tuple
        Input shape
    num_classes : int
        Number of output classes
    unfreeze_strategy : str
        Strategy for unfreezing layers ('top_layers', 'gradual', 'block_wise')
        
    Returns:
    --------
    tf.keras.Model
        Configured model ready for progressive training
    """
    # Load pre-trained base model
    if base_model_name.lower() == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input = tf.keras.applications.resnet50.preprocess_input
    elif base_model_name.lower() == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input = tf.keras.applications.vgg16.preprocess_input
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")
    
    # Initially freeze all layers
    base_model.trainable = False
    
    # Create model
    inputs = tf.keras.Input(shape=input_shape)
    x = preprocess_input(inputs)
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Configure unfreezing based on strategy
    if unfreeze_strategy == 'top_layers':
        # Unfreeze the top N layers
        n_layers_to_unfreeze = 20
        base_model.trainable = True
        for layer in base_model.layers[:-n_layers_to_unfreeze]:
            layer.trainable = False
    
    elif unfreeze_strategy == 'gradual':
        # Prepare for gradual unfreezing during training
        # (will be handled in the training function)
        pass
    
    elif unfreeze_strategy == 'block_wise':
        # Unfreeze specific blocks (e.g., for ResNet)
        base_model.trainable = True
        
        # Example for ResNet: unfreeze only the last block
        for layer in base_model.layers:
            # Freeze all layers except those in the last block
            if 'conv5' in layer.name or 'bn5' in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False
    
    else:
        raise ValueError(f"Unknown unfreezing strategy: {unfreeze_strategy}")
    
    return model, base_model

def train_with_progressive_unfreezing(model, base_model, train_dataset, validation_dataset, 
                                     epochs_per_stage=5, learning_rates=[1e-3, 5e-4, 1e-4]):
    """
    Train a model with progressive unfreezing of layers.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Model to train
    base_model : tf.keras.Model
        Base model for accessing layers
    train_dataset : tf.data.Dataset
        Training dataset
    validation_dataset : tf.data.Dataset
        Validation dataset
    epochs_per_stage : int
        Number of epochs for each unfreezing stage
    learning_rates : list
        Learning rates for each stage
        
    Returns:
    --------
    dict
        Combined training history
    """
    # Determine the total number of layers in the base model
    n_layers = len(base_model.layers)
    
    # Define unfreezing stages (e.g., 3 stages for gradual unfreezing)
    # Each stage unfreezes an additional portion of the network
    stage_sizes = [n_layers, int(n_layers * 2/3), int(n_layers * 1/3), 0]
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train progressively
    combined_history = {
        'accuracy': [],
        'val_accuracy': [],
        'loss': [],
        'val_loss': []
    }
    
    for stage, (freeze_until, lr) in enumerate(zip(stage_sizes, learning_rates)):
        print(f"\nStage {stage+1}: Freezing layers up to {freeze_until}, LR={lr}")
        
        # Freeze/unfreeze layers
        base_model.trainable = True
        for i, layer in enumerate(base_model.layers):
            layer.trainable = i >= freeze_until
        
        # Recompile model with appropriate learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print trainable status
        print("Trainable layers:")
        for i, layer in enumerate(base_model.layers):
            if layer.trainable:
                print(f"  {i}: {layer.name}")
        
        # Train for this stage
        history = model.fit(
            train_dataset,
            epochs=epochs_per_stage,
            validation_data=validation_dataset,
            callbacks=callbacks
        )
        
        # Append history
        for key in combined_history:
            combined_history[key].extend(history.history[key])
    
    # Plot combined training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(combined_history['accuracy'], label='Training Accuracy')
    plt.plot(combined_history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Progressive Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(combined_history['loss'], label='Training Loss')
    plt.plot(combined_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Progressive Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Add stage markers
    for ax in plt.gcf().axes:
        for stage in range(1, len(stage_sizes)):
            epoch = stage * epochs_per_stage
            ax.axvline(x=epoch, color='red', linestyle='--', alpha=0.5)
    
    return combined_history
```

---

## ⚡ Performance Optimization

Optimizing CNN performance for efficiency and speed:

### Quantization

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

def quantize_model(model, dataset, num_calibration_examples=100):
    """
    Quantize a TensorFlow model to reduce size and improve inference speed.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Model to quantize
    dataset : tf.data.Dataset
        Dataset for calibration
    num_calibration_examples : int
        Number of examples to use for calibration
        
    Returns:
    --------
    tf.lite.Interpreter
        Quantized TFLite model
    """
    # Define a representative dataset generator for quantization
    def representative_dataset_gen():
        for data, _ in dataset.take(num_calibration_examples):
            yield [data]
    
    # Convert to TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set quantization parameters
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    # Ensure the model is fully quantized
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Convert the model
    quantized_tflite_model = converter.convert()
    
    # Create an interpreter
    interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
    interpreter.allocate_tensors()
    
    return interpreter, quantized_tflite_model

def compare_model_performance(original_model, quantized_interpreter, test_dataset):
    """
    Compare performance between original and quantized models.
    
    Parameters:
    -----------
    original_model : tf.keras.Model
        Original model
    quantized_interpreter : tf.lite.Interpreter
        Quantized TFLite model interpreter
    test_dataset : tf.data.Dataset
        Test dataset
        
    Returns:
    --------
    dict
        Performance comparison results
    """
    # Get input and output details
    input_details = quantized_interpreter.get_input_details()
    output_details = quantized_interpreter.get_output_details()
    
    # Prepare variables for evaluation
    original_correct = 0
    quantized_correct = 0
    total = 0
    
    original_times = []
    quantized_times = []
    
    # Process test dataset
    for images, labels in test_dataset:
        batch_size = images.shape[0]
        total += batch_size
        
        # Convert labels to numpy
        labels_np = labels.numpy()
        if len(labels_np.shape) > 1 and labels_np.shape[1] > 1:
            # One-hot encoded labels
            labels_np = np.argmax(labels_np, axis=1)
        
        # Evaluate original model
        start_time = time.time()
        original_preds = original_model.predict(images)
        original_times.append(time.time() - start_time)
        
        original_pred_classes = np.argmax(original_preds, axis=1)
        original_correct += np.sum(original_pred_classes == labels_np)
        
        # Evaluate quantized model
        start_time = time.time()
        
        # Process each image individually for the quantized model
        quantized_pred_classes = []
        for i in range(batch_size):
            input_data = images[i:i+1].numpy()
            
            # Quantize input data
            input_scale, input_zero_point = input_details[0]['quantization']
            if input_scale != 0:  # Check if quantization is applied
                input_data = input_data / input_scale + input_zero_point
                input_data = input_data.astype(np.int8)
            
            quantized_interpreter.set_tensor(input_details[0]['index'], input_data)
            quantized_interpreter.invoke()
            
            # Get the output and dequantize
            output_data = quantized_interpreter.get_tensor(output_details[0]['index'])
            output_scale, output_zero_point = output_details[0]['quantization']
            if output_scale != 0:  # Check if quantization is applied
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
            
            quantized_pred_classes.append(np.argmax(output_data[0]))
        
        quantized_times.append(time.time() - start_time)
        quantized_correct += np.sum(np.array(quantized_pred_classes) == labels_np)
    
    # Calculate accuracies
    original_accuracy = original_correct / total
    quantized_accuracy = quantized_correct / total
    
    # Calculate average inference times
    original_avg_time = np.mean(original_times)
    quantized_avg_time = np.mean(quantized_times)
    
    # Calculate size difference
    original_size = get_model_size(original_model)
    quantized_size = len(quantized_interpreter._model_size)
    
    # Print results
    print(f"Original Model:")
    print(f"  Accuracy: {original_accuracy:.4f}")
    print(f"  Average Inference Time: {original_avg_time*1000:.2f} ms")
    print(f"  Model Size: {original_size/1024/1024:.2f} MB")
    
    print(f"\nQuantized Model:")
    print(f"  Accuracy: {quantized_accuracy:.4f}")
    print(f"  Average Inference Time: {quantized_avg_time*1000:.2f} ms")
    print(f"  Model Size: {quantized_size/1024/1024:.2f} MB")
    
    print(f"\nComparison:")
    print(f"  Accuracy Difference: {(quantized_accuracy - original_accuracy)*100:.2f}%")
    print(f"  Speedup: {original_avg_time/quantized_avg_time:.2f}x")
    print(f"  Size Reduction: {original_size/quantized_size:.2f}x")
    
    # Visualize comparison
    plt.figure(figsize=(15, 5))
    
    # Accuracy comparison
    plt.subplot(1, 3, 1)
    plt.bar(['Original', 'Quantized'], [original_accuracy, quantized_accuracy], color=['blue', 'orange'])
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    
    # Inference time comparison
    plt.subplot(1, 3, 2)
    plt.bar(['Original', 'Quantized'], [original_avg_time*1000, quantized_avg_time*1000], color=['blue', 'orange'])
    plt.ylabel('Time (ms)')
    plt.title('Average Inference Time')
    
    # Size comparison
    plt.subplot(1, 3, 3)
    plt.bar(['Original', 'Quantized'], [original_size/1024/1024, quantized_size/1024/1024], color=['blue', 'orange'])
    plt.ylabel('Size (MB)')
    plt.title('Model Size')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'original_accuracy': original_accuracy,
        'quantized_accuracy': quantized_accuracy,
        'original_time': original_avg_time,
        'quantized_time': quantized_avg_time,
        'original_size': original_size,
        'quantized_size': quantized_size
    }

def get_model_size(model):
    """Estimate the size of a Keras model in bytes."""
    # Save model to a temporary file
    model_path = 'temp_model.h5'
    model.save(model_path)
    
    # Get file size
    import os
    size = os.path.getsize(model_path)
    
    # Clean up
    os.remove(model_path)
    
    return size
```

### Pruning

```python
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import matplotlib.pyplot as plt
import tempfile

def prune_model(model, train_dataset, validation_dataset, epochs=10):
    """
    Apply pruning to a CNN model to reduce size and improve efficiency.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Model to prune
    train_dataset : tf.data.Dataset
        Training dataset
    validation_dataset : tf.data.Dataset
        Validation dataset
    epochs : int
        Number of training epochs
        
    Returns:
    --------
    tf.keras.Model
        Pruned model
    """
    # Define pruning schedule
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,  # Start with no pruning
        final_sparsity=0.80,   # End with 80% sparsity (80% of weights set to zero)
        begin_step=0,
        end_step=epochs * len(train_dataset)
    )
    
    # Apply pruning to the model
    pruning_params = {
        'pruning_schedule': pruning_schedule
    }
    
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
        model, **pruning_params
    )
    
    # Compile the pruned model
    model_for_pruning.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Add pruning callbacks
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=tempfile.mkdtemp()),
    ]
    
    # Train the pruned model
    history = model_for_pruning.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=callbacks
    )
    
    # Strip pruning to get the final pruned model
    final_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    
    return final_model, history

def evaluate_pruned_model(original_model, pruned_model, test_dataset):
    """
    Compare performance between original and pruned models.
    
    Parameters:
    -----------
    original_model : tf.keras.Model
        Original model
    pruned_model : tf.keras.Model
        Pruned model
    test_dataset : tf.data.Dataset
        Test dataset
        
    Returns:
    --------
    dict
        Performance comparison results
    """
    # Evaluate original model
    original_loss, original_accuracy = original_model.evaluate(test_dataset)
    
    # Evaluate pruned model
    pruned_loss, pruned_accuracy = pruned_model.evaluate(test_dataset)
    
    # Estimate size and compression
    original_size = get_model_size(original_model)
    pruned_size = get_model_size(pruned_model)
    
    # Convert pruned model to TFLite format for accurate size comparison
    converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
    pruned_tflite_model = converter.convert()
    pruned_tflite_size = len(pruned_tflite_model)
    
    # Apply standard compression to both models (simulating gzip)
    import zlib
    
    # Save models to bytes for compression
    original_bytes = original_model.save(tempfile.mkdtemp())
    pruned_bytes = pruned_model.save(tempfile.mkdtemp())
    
    # Compress
    original_compressed_size = len(zlib.compress(str(original_bytes).encode()))
    pruned_compressed_size = len(zlib.compress(str(pruned_bytes).encode()))
    
    # Print results
    print(f"Original Model:")
    print(f"  Accuracy: {original_accuracy:.4f}")
    print(f"  Model Size: {original_size/1024/1024:.2f} MB")
    print(f"  Compressed Size: {original_compressed_size/1024/1024:.2f} MB")
    
    print(f"\nPruned Model:")
    print(f"  Accuracy: {pruned_accuracy:.4f}")
    print(f"  Model Size: {pruned_size/1024/1024:.2f} MB")
    print(f"  TFLite Size: {pruned_tflite_size/1024/1024:.2f} MB")
    print(f"  Compressed Size: {pruned_compressed_size/1024/1024:.2f} MB")
    
    print(f"\nComparison:")
    print(f"  Accuracy Difference: {(pruned_accuracy - original_accuracy)*100:.2f}%")
    print(f"  Size Reduction (Raw): {original_size/pruned_size:.2f}x")
    print(f"  Size Reduction (Compressed): {original_compressed_size/pruned_compressed_size:.2f}x")
    
    # Visualize comparison
    plt.figure(figsize=(12, 5))
    
    # Accuracy comparison
    plt.subplot(1, 2, 1)
    plt.bar(['Original', 'Pruned'], [original_accuracy, pruned_accuracy], color=['blue', 'green'])
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    
        # Size comparison
    plt.subplot(1, 2, 2)
    plt.bar(['Original', 'Pruned', 'Pruned (TFLite)'], 
            [original_size/1024/1024, pruned_size/1024/1024, pruned_tflite_size/1024/1024],
            color=['blue', 'green', 'lime'])
    plt.ylabel('Size (MB)')
    plt.title('Model Size')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'original_accuracy': original_accuracy,
        'pruned_accuracy': pruned_accuracy,
        'original_size': original_size,
        'pruned_size': pruned_size,
        'pruned_tflite_size': pruned_tflite_size,
        'compression_ratio': original_size / pruned_tflite_size
    }
```

### Knowledge Distillation

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def distill_knowledge(teacher_model, train_dataset, validation_dataset, 
                     student_architecture, temperature=5.0, alpha=0.5, epochs=20):
    """
    Implement knowledge distillation to train a smaller student model.
    
    Parameters:
    -----------
    teacher_model : tf.keras.Model
        Pre-trained teacher model
    train_dataset : tf.data.Dataset
        Training dataset
    validation_dataset : tf.data.Dataset
        Validation dataset
    student_architecture : function
        Function that creates the student model architecture
    temperature : float
        Temperature for softening probability distributions
    alpha : float
        Weight for balancing distillation and true label losses
    epochs : int
        Number of training epochs
        
    Returns:
    --------
    tf.keras.Model
        Trained student model
    """
    # Get the input shape and number of classes from the teacher model
    input_shape = teacher_model.input_shape[1:]
    num_classes = teacher_model.output_shape[1]
    
    # Create student model
    student_model = student_architecture(input_shape, num_classes)
    
    # Custom loss function for knowledge distillation
    def distillation_loss(y_true, y_pred):
        # Extract true labels and soft targets
        y_true_hard, y_true_soft = y_true[:, :num_classes], y_true[:, num_classes:]
        
        # Apply temperature scaling
        y_pred_soft = tf.nn.softmax(y_pred / temperature)
        y_true_soft = tf.nn.softmax(y_true_soft / temperature)
        
        # Calculate distillation loss (KL divergence)
        distillation = tf.keras.losses.KLDivergence()(y_true_soft, y_pred_soft) * (temperature ** 2)
        
        # Calculate true label loss
        student_loss = tf.keras.losses.CategoricalCrossentropy()(y_true_hard, y_pred)
        
        # Combine losses
        return alpha * student_loss + (1 - alpha) * distillation
    
    # Compile student model
    student_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=distillation_loss,
        metrics=['accuracy']
    )
    
    # Dataset mapper to create combined targets (true labels + teacher predictions)
    def create_distillation_targets(x, y):
        # Get teacher's predictions
        teacher_preds = teacher_model(x, training=False)
        
        # Combine true labels and teacher's logits
        combined_targets = tf.concat([y, teacher_preds], axis=1)
        
        return x, combined_targets
    
    # Apply the mapping to datasets
    train_distill = train_dataset.map(create_distillation_targets)
    val_distill = validation_dataset.map(create_distillation_targets)
    
    # Train the student model
    history = student_model.fit(
        train_distill,
        epochs=epochs,
        validation_data=val_distill,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
        ]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Knowledge Distillation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Student Model Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Return trained student model
    return student_model, history

def create_small_student_model(input_shape, num_classes):
    """
    Create a smaller student CNN model.
    
    Parameters:
    -----------
    input_shape : tuple
        Input shape (height, width, channels)
    num_classes : int
        Number of output classes
        
    Returns:
    --------
    tf.keras.Model
        Student model
    """
    model = tf.keras.Sequential([
        # First convolutional block - reduced filters
        tf.keras.layers.Conv2D(16, (3, 3), padding='same', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block - reduced filters
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block - reduced filters
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Classification block
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes)
    ])
    
    return model

def compare_teacher_student(teacher_model, student_model, test_dataset):
    """
    Compare performance between teacher and student models.
    
    Parameters:
    -----------
    teacher_model : tf.keras.Model
        Teacher model
    student_model : tf.keras.Model
        Student model
    test_dataset : tf.data.Dataset
        Test dataset
        
    Returns:
    --------
    dict
        Performance comparison results
    """
    # Prepare test dataset with original labels
    test_dataset_original = test_dataset.map(lambda x, y: (x, y[:, :teacher_model.output_shape[1]]))
    
    # Evaluate teacher model
    print("Evaluating teacher model...")
    teacher_loss, teacher_accuracy = teacher_model.evaluate(test_dataset_original)
    
    # Recompile student model with standard categorical crossentropy for evaluation
    student_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Evaluate student model
    print("Evaluating student model...")
    student_loss, student_accuracy = student_model.evaluate(test_dataset_original)
    
    # Compare model sizes
    teacher_size = get_model_size(teacher_model)
    student_size = get_model_size(student_model)
    
    # Compare inference times
    import time
    
    # Get a batch of data for inference time testing
    for x_batch, _ in test_dataset.take(1):
        break
    
    # Warm-up runs
    _ = teacher_model.predict(x_batch)
    _ = student_model.predict(x_batch)
    
    # Measure teacher inference time
    start_time = time.time()
    for _ in range(100):
        _ = teacher_model.predict(x_batch, verbose=0)
    teacher_time = (time.time() - start_time) / 100
    
    # Measure student inference time
    start_time = time.time()
    for _ in range(100):
        _ = student_model.predict(x_batch, verbose=0)
    student_time = (time.time() - start_time) / 100
    
    # Print results
    print(f"Teacher Model:")
    print(f"  Accuracy: {teacher_accuracy:.4f}")
    print(f"  Model Size: {teacher_size/1024/1024:.2f} MB")
    print(f"  Inference Time: {teacher_time*1000:.2f} ms")
    
    print(f"\nStudent Model:")
    print(f"  Accuracy: {student_accuracy:.4f}")
    print(f"  Model Size: {student_size/1024/1024:.2f} MB")
    print(f"  Inference Time: {student_time*1000:.2f} ms")
    
    print(f"\nComparison:")
    print(f"  Accuracy Difference: {(student_accuracy - teacher_accuracy)*100:.2f}%")
    print(f"  Size Reduction: {teacher_size/student_size:.2f}x")
    print(f"  Speed Improvement: {teacher_time/student_time:.2f}x")
    
    # Visualize comparison
    plt.figure(figsize=(15, 5))
    
    # Accuracy comparison
    plt.subplot(1, 3, 1)
    plt.bar(['Teacher', 'Student'], [teacher_accuracy, student_accuracy], color=['blue', 'green'])
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    
    # Size comparison
    plt.subplot(1, 3, 2)
    plt.bar(['Teacher', 'Student'], [teacher_size/1024/1024, student_size/1024/1024], color=['blue', 'green'])
    plt.ylabel('Size (MB)')
    plt.title('Model Size')
    
    # Inference time comparison
    plt.subplot(1, 3, 3)
    plt.bar(['Teacher', 'Student'], [teacher_time*1000, student_time*1000], color=['blue', 'green'])
    plt.ylabel('Time (ms)')
    plt.title('Inference Time')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'teacher_accuracy': teacher_accuracy,
        'student_accuracy': student_accuracy,
        'teacher_size': teacher_size,
        'student_size': student_size,
        'teacher_time': teacher_time,
        'student_time': student_time
    }
```

### Model Compilation and Optimization

```python
import tensorflow as tf
import timeit

def optimize_model_for_inference(model, input_shape=(1, 224, 224, 3)):
    """
    Optimize a model for inference using TensorFlow's optimization techniques.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Model to optimize
    input_shape : tuple
        Input shape including batch dimension
        
    Returns:
    --------
    tf.function
        Optimized model function
    """
    # Convert to a TensorFlow Graph using tf.function
    @tf.function
    def optimized_model(inputs):
        return model(inputs, training=False)
    
    # Create example input for tracing
    example_input = tf.random.normal(input_shape)
    
    # Trace the function with example input to optimize
    optimized_model(example_input)
    
    # Get the concrete function
    concrete_function = optimized_model.get_concrete_function(
        tf.TensorSpec(input_shape, tf.float32)
    )
    
    print("Model optimized for inference")
    
    return optimized_model, concrete_function

def benchmark_inference(model, optimized_model, input_shape=(1, 224, 224, 3), num_runs=100):
    """
    Benchmark inference time for original and optimized models.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Original model
    optimized_model : tf.function
        Optimized model function
    input_shape : tuple
        Input shape including batch dimension
    num_runs : int
        Number of runs for averaging inference time
        
    Returns:
    --------
    dict
        Benchmark results
    """
    # Create random input data
    inputs = tf.random.normal(input_shape)
    
    # Warm-up runs
    _ = model(inputs, training=False)
    _ = optimized_model(inputs)
    
    # Benchmark original model
    original_times = []
    for _ in range(num_runs):
        start_time = timeit.default_timer()
        _ = model(inputs, training=False)
        original_times.append(timeit.default_timer() - start_time)
    
    original_avg_time = sum(original_times) / num_runs
    
    # Benchmark optimized model
    optimized_times = []
    for _ in range(num_runs):
        start_time = timeit.default_timer()
        _ = optimized_model(inputs)
        optimized_times.append(timeit.default_timer() - start_time)
    
    optimized_avg_time = sum(optimized_times) / num_runs
    
    # Print results
    print(f"Original Model Inference Time: {original_avg_time*1000:.4f} ms")
    print(f"Optimized Model Inference Time: {optimized_avg_time*1000:.4f} ms")
    print(f"Speedup: {original_avg_time/optimized_avg_time:.2f}x")
    
    # Visualize results
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(12, 5))
    
    # Average time comparison
    plt.subplot(1, 2, 1)
    plt.bar(['Original', 'Optimized'], 
            [original_avg_time*1000, optimized_avg_time*1000],
            color=['blue', 'green'])
    plt.ylabel('Time (ms)')
    plt.title('Average Inference Time')
    
    # Time distribution
    plt.subplot(1, 2, 2)
    plt.boxplot([np.array(original_times)*1000, np.array(optimized_times)*1000],
               labels=['Original', 'Optimized'])
    plt.ylabel('Time (ms)')
    plt.title('Inference Time Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'original_avg_time': original_avg_time,
        'optimized_avg_time': optimized_avg_time,
        'speedup': original_avg_time / optimized_avg_time,
        'original_times': original_times,
        'optimized_times': optimized_times
    }
```

---

## 🌟 Applications

Convolutional Neural Networks have revolutionized numerous fields:

### Image Classification

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

def image_classification_demo():
    """
    Demonstrate image classification with a CNN.
    """
    # Load dataset (CIFAR-10)
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize pixel values
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    
    # Convert labels to one-hot encoding
    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)
    
    # Create CNN model
    model = models.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Classification layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        train_images, train_labels,
        epochs=15,
        batch_size=128,
        validation_split=0.1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Define class names
    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
                  'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    
    # Make predictions on test images
    predictions = model.predict(test_images)
    
    # Display some predictions
    plt.figure(figsize=(12, 12))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(test_images[i])
        
        true_label = np.argmax(test_labels[i])
        pred_label = np.argmax(predictions[i])
        
        color = 'green' if true_label == pred_label else 'red'
        
        plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}", 
                 color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Show confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(test_labels, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    return model, history
```

### Object Detection

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import cv2

def simple_object_detection_demo(image_path, model_path=None):
    """
    Demonstrate a simple sliding window approach for object detection.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    model_path : str, optional
        Path to a pre-trained classification model
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load or create a model
    if model_path:
        model = tf.keras.models.load_model(model_path)
    else:
        # Use a pre-trained model from TensorFlow
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=True,
            input_shape=(224, 224, 3)
        )
        model = base_model
    
    # Define window parameters
    window_size = (224, 224)
    stride = 112  # 50% overlap
    
    # Prepare to store detection results
    detections = []
    confidence_threshold = 0.7
    
    # Sliding window approach
    for y in range(0, image.shape[0] - window_size[1], stride):
        for x in range(0, image.shape[1] - window_size[0], stride):
            # Extract window
            window = image[y:y+window_size[1], x:x+window_size[0]]
            
            # Preprocess for model
            window_processed = tf.image.resize(window, (224, 224))
            window_processed = tf.keras.applications.mobilenet_v2.preprocess_input(window_processed)
            window_processed = np.expand_dims(window_processed, axis=0)
            
            # Make prediction
            prediction = model.predict(window_processed, verbose=0)
            
            # Get class and confidence
            if hasattr(model, 'predict_proba'):
                confidence = np.max(prediction)
                class_id = np.argmax(prediction)
            else:
                confidence = np.max(prediction)
                class_id = np.argmax(prediction)
            
            # Store high-confidence detections
            if confidence > confidence_threshold:
                detections.append({
                    'box': (x, y, x+window_size[0], y+window_size[1]),
                    'confidence': float(confidence),
                    'class_id': int(class_id)
                })
    
    # Non-maximum suppression to remove overlapping detections
    def iou(box1, box2):
        # Calculate Intersection over Union
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2
        
        xi1, yi1 = max(x1, x1_), max(y1, y1_)
        xi2, yi2 = min(x2, x2_), min(y2, y2_)
        
        intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)
        
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area
    
    # Sort detections by confidence
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Apply non-maximum suppression
    kept_detections = []
    iou_threshold = 0.3
    
    while detections:
        kept_detections.append(detections.pop(0))
        
        detections = [d for d in detections if iou(d['box'], kept_detections[-1]['box']) < iou_threshold]
    
    # Load ImageNet class names
    with open('imagenet_classes.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Visualize detections
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    for detection in kept_detections:
        x1, y1, x2, y2 = detection['box']
        confidence = detection['confidence']
        class_id = detection['class_id']
        
        # Draw bounding box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
        
        # Add label
        plt.text(x1, y1-10, f"{class_names[class_id]}: {confidence:.2f}", 
                color='red', fontsize=12, backgroundcolor='white')
    
    plt.axis('off')
    plt.title('Object Detection Results')
    plt.tight_layout()
    plt.show()
    
    return kept_detections
```

### Image Segmentation

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

def unet_segmentation_model(input_shape=(128, 128, 3), num_classes=1):
    """
    Create a U-Net model for image segmentation.
    
    Parameters:
    -----------
    input_shape : tuple
        Input image shape (height, width, channels)
    num_classes : int
        Number of segmentation classes
        
    Returns:
    --------
    tf.keras.Model
        U-Net model
    """
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    
    # Encoder (downsampling path)
    # Block 1
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Block 2
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Block 3
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Block 4
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bridge
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    
    # Decoder (upsampling path)
    # Block 6
    up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    concat6 = layers.Concatenate()([up6, conv4])
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(concat6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    # Block 7
    up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    concat7 = layers.Concatenate()([up7, conv3])
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(concat7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    # Block 8
    up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    concat8 = layers.Concatenate()([up8, conv2])
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(concat8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    # Block 9
    up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    concat9 = layers.Concatenate()([up9, conv1])
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    # Output layer
    if num_classes == 1:
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
    else:
        outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv9)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def segment_image(model, image, threshold=0.5):
    """
    Segment an image using the provided model.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Segmentation model
    image : numpy.ndarray
        Input image
    threshold : float
        Threshold for binary segmentation
        
    Returns:
    --------
    numpy.ndarray
        Segmentation mask
    """
    # Preprocess image
    h, w = model.input_shape[1:3]
    processed_image = tf.image.resize(image, (h, w))
    processed_image = processed_image / 255.0  # Normalize
    processed_image = np.expand_dims(processed_image, axis=0)
    
    # Generate prediction
    prediction = model.predict(processed_image)[0]
    
    # Apply threshold for binary segmentation
    if prediction.shape[-1] == 1:
        mask = (prediction > threshold).astype(np.uint8)
        mask = np.squeeze(mask, axis=-1)
    else:
        # For multi-class segmentation
        mask = np.argmax(prediction, axis=-1).astype(np.uint8)
    
    # Resize mask to original image size
    mask = tf.image.resize(
        np.expand_dims(mask, axis=-1),
        (image.shape[0], image.shape[1]),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    ).numpy()
    
    mask = np.squeeze(mask, axis=-1)
    
    return mask

def visualize_segmentation(image, mask, alpha=0.5):
    """
    Visualize image segmentation results.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Original image
    mask : numpy.ndarray
        Segmentation mask
    alpha : float
        Transparency of the overlay
    """
    # Create a color mask
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # For binary segmentation
    if len(np.unique(mask)) <= 2:
        color_mask[mask == 1] = [255, 0, 0]  # Red for the foreground
    else:
        # For multi-class segmentation, assign different colors
        colors = [
            [255, 0, 0],     # Red
            [0, 255, 0],     # Green
            [0, 0, 255],     # Blue
            [255, 255, 0],   # Yellow
            [255, 0, 255],   # Magenta
            [0, 255, 255],   # Cyan
            [128, 0, 0],     # Maroon
            [0, 128, 0],     # Green (dark)
            [0, 0, 128],     # Navy
            [128, 128, 0],   # Olive
        ]
        
        for i in range(1, min(len(np.unique(mask)), len(colors) + 1)):
            color_mask[mask == i] = colors[i-1]
    
    # Create an overlay
    overlay = image.copy()
    cv2.addWeighted(color_mask, alpha, overlay, 1 - alpha, 0, overlay)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Segmentation Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title('Segmentation Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
```

### Facial Recognition

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import cv2

def simple_face_recognition_demo(image_path, face_cascade_path=None):
    """
    Demonstrate a simple face recognition system.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    face_cascade_path : str, optional
        Path to the Haar cascade XML file for face detection
    """
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load face cascade
    if face_cascade_path is None:
        # Use OpenCV's default cascade file
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    else:
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Create a simple CNN for face recognition
    def create_face_recognition_model(num_people=10):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_people, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    # In a real application, you would:
    # 1. Have a dataset of faces for people you want to recognize
    # 2. Train the model on these faces
    # 3. Use the trained model for inference
    
    # For this demo, we'll just create a dummy model for illustration
    model = create_face_recognition_model()
    
    # Pretend names for recognized people
    people_names = [
        "John Doe", "Jane Smith", "Alex Johnson", "Sarah Williams", 
        "Michael Brown", "Emma Davis", "James Wilson", "Olivia Taylor",
        "Robert Martinez", "Emily Anderson"
    ]
    
    # Draw boxes around faces and label them
    for (x, y, w, h) in faces:
        # Extract face
        face = rgb[y:y+h, x:x+w]
        
        # Resize to model input size
        face_resized = cv2.resize(face, (100, 100))
        
        # In a real application, you would:
        # 1. Preprocess the face
        # 2. Use the model to identify the person
        # 3. Get the predicted name
        
        # For this demo, we'll randomly choose a name
        person_idx = np.random.randint(0, len(people_names))
        confidence = np.random.uniform(0.7, 0.99)
        
        # Draw rectangle around face
        cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add name and confidence
        label = f"{people_names[person_idx]}: {confidence:.2f}"
        cv2.putText(rgb, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Show the result
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb)
    plt.title('Face Recognition Demo')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return rgb, faces

def siamese_network_for_face_verification(input_shape=(100, 100, 3)):
    """
    Create a Siamese network for face verification.
    
    Parameters:
    -----------
    input_shape : tuple
        Input image shape
        
    Returns:
    --------
    tf.keras.Model
        Siamese network model
    """
    # Base network for feature extraction
    def create_base_network(input_shape):
        input_layer = layers.Input(shape=input_shape)
        
        x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation=None)(x)
        
        # L2 normalization
        x = tf.keras.layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1)
        )(x)
        
        return models.Model(input_layer, x)
    
    # Create base network
    base_network = create_base_network(input_shape)
    
    # Create inputs for both images
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    
    # Get embeddings for both inputs
    embedding_a = base_network(input_a)
    embedding_b = base_network(input_b)
    
    # Calculate distance between embeddings
    distance = tf.keras.layers.Lambda(
        lambda embeddings: tf.reduce_sum(
            tf.square(embeddings[0] - embeddings[1]),
            axis=1,
            keepdims=True
        )
    )([embedding_a, embedding_b])
    
    # Create model
    model = models.Model(inputs=[input_a, input_b], outputs=distance)
    
    # Compile model
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.0001)
    )
    
    return model, base_network
```

### Medical Imaging

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import cv2

def medical_image_analysis_demo():
    """
    Demonstrate CNN application for medical image analysis.
    
    This is a simplified example for educational purposes.
    Real medical image analysis would use actual medical datasets.
    """
    # Create a sample CNN for medical image classification
    def create_medical_image_model(input_shape=(256, 256, 1), num_classes=2):
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Classification block
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    # Generate synthetic medical images for demonstration
    def generate_synthetic_medical_images(num_samples=100, img_size=256):
        # Normal samples
        normal_images = []
        for _ in range(num_samples // 2):
            # Create a synthetic "normal" image
            img = np.ones((img_size, img_size)) * 0.2  # Background
            
            # Add normal tissue structure
            for i in range(5):
                center_x = np.random.randint(50, img_size-50)
                center_y = np.random.randint(50, img_size-50)
                radius = np.random.randint(20, 40)
                
                # Draw circle
                cv2.circle(
                    img, 
                    (center_x, center_y), 
                    radius, 
                    np.random.uniform(0.4, 0.6), 
                    -1
                )
            
            # Add noise
            noise = np.random.normal(0, 0.05, (img_size, img_size))
            img += noise
            img = np.clip(img, 0, 1)
            
            normal_images.append(img)
        
        # Abnormal samples
        abnormal_images = []
        for _ in range(num_samples // 2):
            # Create a synthetic "abnormal" image
            img = np.ones((img_size, img_size)) * 0.2  # Background
            
            # Add normal tissue structure
            for i in range(5):
                center_x = np.random.randint(50, img_size-50)
                center_y = np.random.randint(50, img_size-50)
                radius = np.random.randint(20, 40)
                
                # Draw circle
                cv2.circle(
                    img, 
                    (center_x, center_y), 
                    radius, 
                    np.random.uniform(0.4, 0.6), 
                    -1
                )
            
            # Add abnormality (brighter region)
            center_x = np.random.randint(50, img_size-50)
            center_y = np.random.randint(50, img_size-50)
            radius = np.random.randint(10, 25)
            
            cv2.circle(
                img, 
                (center_x, center_y), 
                radius, 
                np.random.uniform(0.7, 0.9), 
                -1
            )
            
            # Add noise
            noise = np.random.normal(0, 0.05, (img_size, img_size))
            img += noise
            img = np.clip(img, 0, 1)
            
            abnormal_images.append(img)
        
        # Combine data
        X = np.array(normal_images + abnormal_images)
        X = np.expand_dims(X, axis=-1)  # Add channel dimension
        
        # Create labels
        y = np.array([0] * len(normal_images) + [1] * len(abnormal_images))
        y = tf.keras.utils.to_categorical(y, 2)
        
        return X, y
    
    # Generate synthetic data
    X, y = generate_synthetic_medical_images(num_samples=100, img_size=256)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create model
    model = create_medical_image_model()
    
    # Train model (just a few epochs for demonstration)
    history = model.fit(
        X_train, y_train,
        epochs=5,  # Normally you'd train for more epochs
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Visualize results
    plt.figure(figsize=(15, 12))
    
    # Plot training curves
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show some example predictions
    for i in range(4):
        plt.subplot(2, 2, i+3)
        plt.imshow(X_test[i, :, :, 0], cmap='gray')
        
        true_label = "Normal" if true_classes[i] == 0 else "Abnormal"
        pred_label = "Normal" if predicted_classes[i] == 0 else "Abnormal"
        
        color = 'green' if true_classes[i] == predicted_classes[i] else 'red'
        
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Visualize Grad-CAM (attention maps)
    def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
        # Create a model that maps the input image to the activations of the last conv layer
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        # Compute gradient of the predicted class with respect to the output feature map
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        # Gradient of the output neuron with respect to the output feature map
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        # Vector of mean intensity of the gradient over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the gradient values
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    # Visualize Grad-CAM for a few examples
    plt.figure(figsize=(15, 5))
    
    for i in range(3):
        # Get the image
        img = X_test[i]
        img_array = np.expand_dims(img, axis=0)
        
        # Generate heatmap
        heatmap = make_gradcam_heatmap(
            img_array, 
            model, 
            'conv2d_4',  # Adjust based on your model
            pred_index=predicted_classes[i]
        )
        
        # Create a colored heatmap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Convert grayscale image to RGB
        img_rgb = np.repeat(img, 3, axis=-1)
        
        # Superimpose heatmap on original image
        superimposed_img = heatmap_colored * 0.4 + img_rgb * 255
        superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
        
        # Display original image
        plt.subplot(3, 3, i*3+1)
        plt.imshow(img[:, :, 0], cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        # Display heatmap
        plt.subplot(3, 3, i*3+2)
        plt.imshow(heatmap, cmap='jet')
        plt.title('Attention Map')
        plt.axis('off')
        
        # Display superimposed image
        plt.subplot(3, 3, i*3+3)
        plt.imshow(superimposed_img)
        plt.title('Superimposed')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return model, history, (X_train, y_train, X_test, y_test)
```

---

## 🚀 Advanced Topics

### Attention Mechanisms in CNNs

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

class ChannelAttention(layers.Layer):
    """Channel attention module"""
    
    def __init__(self, ratio=16):
        super(ChannelAttention, self).__init__()
        self.ratio = ratio
    
    def build(self, input_shape):
        channel = input_shape[-1]
        
        self.shared_dense_1 = layers.Dense(channel // self.ratio,
                                         activation='relu',
                                         kernel_initializer='he_normal',
                                         use_bias=False)
        self.shared_dense_2 = layers.Dense(channel,
                                         kernel_initializer='he_normal',
                                         use_bias=False)
        
        super(ChannelAttention, self).build(input_shape)
    
    def call(self, inputs):
        # Average pooling
        avg_pool = layers.GlobalAveragePooling2D()(inputs)
        avg_pool = layers.Reshape((1, 1, inputs.shape[-1]))(avg_pool)
        
        # Max pooling
        max_pool = layers.GlobalMaxPooling2D()(inputs)
        max_pool = layers.Reshape((1, 1, inputs.shape[-1]))(max_pool)
        
        # Shared MLP
        avg_out = self.shared_dense_1(avg_pool)
        avg_out = self.shared_dense_2(avg_out)
        
        max_out = self.shared_dense_1(max_pool)
        max_out = self.shared_dense_2(max_out)
        
        # Activation
        channel_attention = layers.add([avg_out, max_out])
        channel_attention = tf.nn.sigmoid(channel_attention)
        
        return inputs * channel_attention

class SpatialAttention(layers.Layer):
    """Spatial attention module"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
    
    def build(self, input_shape):
        self.conv = layers.Conv2D(filters=1,
                                kernel_size=self.kernel_size,
                                padding='same',
                                activation='sigmoid',
                                kernel_initializer='he_normal',
                                use_bias=False)
        
        super(SpatialAttention, self).build(input_shape)
    
    def call(self, inputs):
        # Average pooling along channel axis
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        
        # Max pooling along channel axis
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        
        # Concatenate pooled features
        pooled = tf.concat([avg_pool, max_pool], axis=-1)
        
        # Convolutional layer
        spatial_attention = self.conv(pooled)
        
        return inputs * spatial_attention

class CBAM(layers.Layer):
    """Convolutional Block Attention Module"""
    
    def __init__(self, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def call(self, inputs):
        # Apply channel attention
        x = self.channel_attention(inputs)
        # Apply spatial attention
        x = self.spatial_attention(x)
        return x

def create_attention_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Create a CNN with attention mechanisms.
    
    Parameters:
    -----------
    input_shape : tuple
        Input image shape
    num_classes : int
        Number of output classes
        
    Returns:
    --------
    tf.keras.Model
        CNN with attention
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # First convolutional block
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = CBAM()(x)  # Add attention
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Second convolutional block
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = CBAM()(x)  # Add attention
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Third convolutional block
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = CBAM()(x)  # Add attention
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Classification block
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model
```

### Self-Supervised Learning with CNNs

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

def create_rotation_prediction_model(input_shape=(32, 32, 3)):
    """
    Create a self-supervised learning model that predicts image rotation.
    
    Parameters:
    -----------
    input_shape : tuple
        Input image shape
        
    Returns:
    --------
    tuple
        (base_model, rotation_model)
    """
    # Base model for feature extraction
    base_model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.GlobalAveragePooling2D()
    ])
    
    # Rotation prediction model
    rotation_model = models.Sequential([
        base_model,
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')  # 4 rotations: 0°, 90°, 180°, 270°
    ])
    
    # Compile model
    rotation_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return base_model, rotation_model

def generate_rotated_images(images):
    """
    Generate rotated versions of images for self-supervised learning.
    
    Parameters:
    -----------
    images : numpy.ndarray
        Original images
        
    Returns:
    --------
    tuple
        (rotated_images, rotation_labels)
    """
    # Number of original images
    num_images = images.shape[0]
    
    # Create arrays to store rotated images and labels
    rotated_images = np.zeros((num_images * 4, *images.shape[1:]), dtype=images.dtype)
    rotation_labels = np.zeros(num_images * 4, dtype=np.int32)
    
    for i in range(num_images):
        # Original image (0° rotation)
        rotated_images[i*4] = images[i]
        rotation_labels[i*4] = 0
        
        # 90° rotation
        rotated_images[i*4 + 1] = np.rot90(images[i], k=1)
        rotation_labels[i*4 + 1] = 1
        
        # 180° rotation
        rotated_images[i*4 + 2] = np.rot90(images[i], k=2)
        rotation_labels[i*4 + 2] = 2
        
                # 270° rotation
        rotated_images[i*4 + 3] = np.rot90(images[i], k=3)
        rotation_labels[i*4 + 3] = 3
    
    return rotated_images, rotation_labels

def train_self_supervised_model(images, batch_size=64, epochs=10):
    """
    Train a self-supervised model using rotation prediction.
    
    Parameters:
    -----------
    images : numpy.ndarray
        Training images
    batch_size : int
        Batch size
    epochs : int
        Number of training epochs
        
    Returns:
    --------
    tuple
        (base_model, rotation_model, history)
    """
    # Create models
    base_model, rotation_model = create_rotation_prediction_model(input_shape=images.shape[1:])
    
    # Generate rotated images and labels
    rotated_images, rotation_labels = generate_rotated_images(images)
    
    # Normalize images
    rotated_images = rotated_images.astype('float32') / 255.0
    
    # Train model
    history = rotation_model.fit(
        rotated_images, rotation_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Rotation Prediction Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Rotation Prediction Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return base_model, rotation_model, history

def visualize_self_supervised_features(base_model, images, labels):
    """
    Visualize features learned through self-supervised learning.
    
    Parameters:
    -----------
    base_model : tf.keras.Model
        Pre-trained base model
    images : numpy.ndarray
        Images to extract features from
    labels : numpy.ndarray
        True labels for visualization
        
    Returns:
    --------
    numpy.ndarray
        Extracted features
    """
    # Normalize images
    images_normalized = images.astype('float32') / 255.0
    
    # Extract features
    features = base_model.predict(images_normalized)
    
    # Use t-SNE for dimensionality reduction
    from sklearn.manifold import TSNE
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Plot t-SNE visualization
    plt.figure(figsize=(10, 8))
    
    # Create a scatter plot colored by class
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[i]],
            label=f'Class {label}',
            alpha=0.6
        )
    
    plt.title('t-SNE Visualization of Self-Supervised Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return features
```

### Graph Convolutional Networks

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

class GraphConvolution(layers.Layer):
    """
    Graph Convolutional Layer.
    
    Implementation of the graph convolutional layer described in:
    "Semi-Supervised Classification with Graph Convolutional Networks" (Kipf & Welling, 2017)
    """
    
    def __init__(self, units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    
    def build(self, input_shape):
        # Input shape should be a list: [node_features, adjacency_matrix]
        node_features_shape = input_shape[0]
        
        # Create weight matrix
        self.kernel = self.add_weight(
            shape=(node_features_shape[-1], self.units),
            initializer=self.kernel_initializer,
            name='kernel'
        )
        
        # Create bias
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer='zeros',
                name='bias'
            )
        
        self.built = True
    
    def call(self, inputs):
        # Unpack inputs: [node_features, adjacency_matrix]
        node_features, adjacency_matrix = inputs
        
        # Basic GCN propagation rule: A_hat * X * W
        # where A_hat is the normalized adjacency matrix
        supports = tf.matmul(adjacency_matrix, node_features)
        output = tf.matmul(supports, self.kernel)
        
        # Add bias if needed
        if self.use_bias:
            output += self.bias
        
        # Apply activation if specified
        if self.activation is not None:
            output = self.activation(output)
        
        return output

def normalize_adjacency(adjacency):
    """
    Normalize adjacency matrix for GCN.
    
    Parameters:
    -----------
    adjacency : numpy.ndarray or scipy.sparse.csr_matrix
        Adjacency matrix
        
    Returns:
    --------
    numpy.ndarray
        Normalized adjacency matrix
    """
    # Add self-connections
    if sp.issparse(adjacency):
        adjacency = adjacency.tolil()
        adjacency.setdiag(1)
        adjacency = adjacency.tocsr()
    else:
        adjacency = adjacency.copy()
        np.fill_diagonal(adjacency, 1)
    
    # Calculate degree matrix
    if sp.issparse(adjacency):
        degree = np.array(adjacency.sum(1)).flatten()
    else:
        degree = adjacency.sum(axis=1)
    
    # Calculate D^(-1/2)
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    
    # Calculate normalized adjacency: D^(-1/2) * A * D^(-1/2)
    if sp.issparse(adjacency):
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        normalized_adjacency = d_inv_sqrt_mat.dot(adjacency).dot(d_inv_sqrt_mat)
        normalized_adjacency = normalized_adjacency.toarray()
    else:
        d_inv_sqrt_mat = np.diag(d_inv_sqrt)
        normalized_adjacency = d_inv_sqrt_mat.dot(adjacency).dot(d_inv_sqrt_mat)
    
    return normalized_adjacency

def create_gcn_model(input_shape, adj_shape, hidden_units=16, num_classes=7):
    """
    Create a Graph Convolutional Network model.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of node features
    adj_shape : tuple
        Shape of adjacency matrix
    hidden_units : int
        Number of hidden units
    num_classes : int
        Number of output classes
        
    Returns:
    --------
    tf.keras.Model
        GCN model
    """
    # Input layers
    node_features_input = tf.keras.Input(shape=input_shape)
    adjacency_input = tf.keras.Input(shape=adj_shape)
    
    # First GCN layer
    x = GraphConvolution(hidden_units, activation='relu')([node_features_input, adjacency_input])
    
    # Second GCN layer
    x = GraphConvolution(num_classes)([x, adjacency_input])
    
    # Apply softmax
    outputs = tf.keras.activations.softmax(x, axis=-1)
    
    # Create model
    model = tf.keras.Model(inputs=[node_features_input, adjacency_input], outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_gcn_on_citation_network():
    """
    Example of training a GCN on a citation network dataset.
    
    This is a simplified example for educational purposes.
    In practice, you would use a real citation network dataset.
    """
    # Create a synthetic citation network dataset
    num_nodes = 100
    num_features = 50
    num_classes = 3
    
    # Generate random node features
    node_features = np.random.randn(num_nodes, num_features).astype(np.float32)
    
    # Generate random adjacency matrix (sparse)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    # Add some random edges (more likely between similar nodes)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # Calculate feature similarity
                similarity = np.dot(node_features[i], node_features[j]) / \
                            (np.linalg.norm(node_features[i]) * np.linalg.norm(node_features[j]))
                
                # Add edge with probability based on similarity
                if np.random.rand() < (similarity + 1) / 2 * 0.1:
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1  # Symmetric
    
    # Generate random node labels
    node_labels = np.random.randint(0, num_classes, size=num_nodes)
    
    # Normalize adjacency matrix
    normalized_adjacency = normalize_adjacency(adjacency)
    
    # Split data into train/val/test
    indices = np.random.permutation(num_nodes)
    train_idx = indices[:60]
    val_idx = indices[60:80]
    test_idx = indices[80:]
    
    # Create mask for training
    train_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[train_idx] = True
    
    val_mask = np.zeros(num_nodes, dtype=bool)
    val_mask[val_idx] = True
    
    test_mask = np.zeros(num_nodes, dtype=bool)
    test_mask[test_idx] = True
    
    # Create GCN model
    model = create_gcn_model(
        input_shape=(num_features,),
        adj_shape=(num_nodes,),
        hidden_units=16,
        num_classes=num_classes
    )
    
    # Custom training loop
    for epoch in range(200):
        # Forward pass
        with tf.GradientTape() as tape:
            logits = model([node_features, normalized_adjacency])
            
            # Calculate loss (only on training nodes)
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                node_labels[train_mask], 
                logits[train_mask]
            )
            loss = tf.reduce_mean(loss)
        
        # Backward pass
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Calculate accuracy
        predictions = tf.argmax(logits, axis=1)
        
        train_acc = np.mean(predictions[train_mask].numpy() == node_labels[train_mask])
        val_acc = np.mean(predictions[val_mask].numpy() == node_labels[val_mask])
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
    
    # Final evaluation
    logits = model([node_features, normalized_adjacency])
    predictions = tf.argmax(logits, axis=1)
    
    test_acc = np.mean(predictions[test_mask].numpy() == node_labels[test_mask])
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Visualize node embeddings
    # Extract node embeddings from the first GCN layer
    embedding_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.layers[2].output
    )
    node_embeddings = embedding_model([node_features, normalized_adjacency]).numpy()
    
    # Use t-SNE for visualization
    from sklearn.manifold import TSNE
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(node_embeddings)
    
    # Plot node embeddings colored by class
    plt.figure(figsize=(10, 8))
    
    for class_id in range(num_classes):
        class_mask = node_labels == class_id
        plt.scatter(
            embeddings_2d[class_mask, 0],
            embeddings_2d[class_mask, 1],
            label=f'Class {class_id}',
            alpha=0.7
        )
    
    plt.title('Node Embeddings from GCN')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model, node_embeddings
```

### 3D Convolutional Networks

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

def create_3d_cnn(input_shape=(16, 128, 128, 1), num_classes=10):
    """
    Create a 3D Convolutional Neural Network.
    
    Parameters:
    -----------
    input_shape : tuple
        Input shape (frames, height, width, channels)
    num_classes : int
        Number of output classes
        
    Returns:
    --------
    tf.keras.Model
        3D CNN model
    """
    model = models.Sequential([
        # First 3D convolutional block
        layers.Conv3D(32, kernel_size=(3, 3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv3D(32, kernel_size=(3, 3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling3D(pool_size=(1, 2, 2)),
        
        # Second 3D convolutional block
        layers.Conv3D(64, kernel_size=(3, 3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv3D(64, kernel_size=(3, 3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling3D(pool_size=(1, 2, 2)),
        
        # Third 3D convolutional block
        layers.Conv3D(128, kernel_size=(3, 3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv3D(128, kernel_size=(3, 3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling3D(pool_size=(1, 2, 2)),
        
        # Temporal pooling
        layers.Conv3D(256, kernel_size=(3, 3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        
        # Classification block
        layers.GlobalAveragePooling3D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def generate_synthetic_video_data(num_samples=100, num_frames=16, frame_height=128, frame_width=128, num_classes=10):
    """
    Generate synthetic video data for 3D CNN.
    
    Parameters:
    -----------
    num_samples : int
        Number of video samples
    num_frames : int
        Number of frames per video
    frame_height : int
        Height of each frame
    frame_width : int
        Width of each frame
    num_classes : int
        Number of classes
        
    Returns:
    --------
    tuple
        (videos, labels)
    """
    # Generate random videos
    videos = np.zeros((num_samples, num_frames, frame_height, frame_width, 1), dtype=np.float32)
    
    # Generate random labels
    labels = np.random.randint(0, num_classes, size=num_samples)
    
    # Generate synthetic patterns for each class
    for i in range(num_samples):
        class_id = labels[i]
        
        # Create a basic pattern based on class
        # (in a real dataset, videos of the same class would share certain patterns)
        
        # Create a moving object
        obj_size = 20
        start_x = np.random.randint(0, frame_width - obj_size)
        start_y = np.random.randint(0, frame_height - obj_size)
        
        # Direction depends on class
        dx = (class_id % 3 - 1) * 3  # -3, 0, or 3
        dy = ((class_id // 3) % 3 - 1) * 3  # -3, 0, or 3
        
        # Add some randomness to motion
        dx += np.random.randn() * 0.5
        dy += np.random.randn() * 0.5
        
        # Create the moving object across frames
        for f in range(num_frames):
            # Calculate object position
            pos_x = int(start_x + dx * f)
            pos_y = int(start_y + dy * f)
            
            # Ensure object stays within frame
            pos_x = min(max(pos_x, 0), frame_width - obj_size - 1)
            pos_y = min(max(pos_y, 0), frame_height - obj_size - 1)
            
            # Draw object
            videos[i, f, pos_y:pos_y+obj_size, pos_x:pos_x+obj_size, 0] = 1.0
            
            # Add noise
            videos[i, f, :, :, 0] += np.random.randn(frame_height, frame_width) * 0.1
            videos[i, f, :, :, 0] = np.clip(videos[i, f, :, :, 0], 0, 1)
    
    # Convert labels to one-hot encoding
    labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes)
    
    return videos, labels_one_hot

def visualize_video_sample(video, label=None):
    """
    Visualize a video sample.
    
    Parameters:
    -----------
    video : numpy.ndarray
        Video data with shape (frames, height, width, channels)
    label : int, optional
        Class label
    """
    num_frames = video.shape[0]
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_frames)))
    
    plt.figure(figsize=(15, 15))
    
    for i in range(num_frames):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(video[i, :, :, 0], cmap='gray')
        plt.title(f'Frame {i+1}')
        plt.axis('off')
    
    if label is not None:
        plt.suptitle(f'Class: {label}', fontsize=16)
    
    plt.tight_layout()
    plt.show()

def train_3d_cnn_example():
    """
    Example of training a 3D CNN on synthetic video data.
    """
    # Generate synthetic data
    videos, labels = generate_synthetic_video_data(
        num_samples=100,
        num_frames=16,
        frame_height=64,
        frame_width=64,
        num_classes=5
    )
    
    # Visualize a sample
    sample_idx = np.random.randint(0, len(videos))
    visualize_video_sample(videos[sample_idx], np.argmax(labels[sample_idx]))
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        videos, labels, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = create_3d_cnn(input_shape=videos.shape[1:], num_classes=labels.shape[1])
    
    # Train model (for demonstration, use few epochs)
    history = model.fit(
        X_train, y_train,
        batch_size=8,
        epochs=5,  # Reduced for demonstration
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return model, history, (X_train, y_train, X_test, y_test)
```

---

## 🚧 Challenges and Limitations

Convolutional Neural Networks face several challenges and limitations:

### Computational Requirements

CNNs, especially deep architectures, require significant computational resources:

- **Memory Usage**: Large feature maps and parameters consume substantial memory
- **Training Time**: Deep networks can take days or weeks to train on large datasets
- **Inference Latency**: Real-time applications may struggle with complex models
- **Power Consumption**: Mobile and edge devices have limited battery capacity

Solutions:
- Model quantization and pruning
- Knowledge distillation to smaller networks
- Hardware acceleration (GPUs, TPUs, specialized ASICs)
- Efficient architectures (MobileNet, EfficientNet)

### Data Hunger

CNNs typically require large labeled datasets:

- **Annotation Cost**: Manual labeling is expensive and time-consuming
- **Domain Gaps**: Models trained on one dataset may not generalize to others
- **Rare Classes**: Imbalanced data leads to poor performance on minority classes
- **Variability Coverage**: Datasets may not represent all real-world variations

Solutions:
- Data augmentation techniques
- Transfer learning from pre-trained models
- Self-supervised and semi-supervised approaches
- Synthetic data generation

### Interpretability

Understanding CNN decisions remains challenging:

- **Black Box Nature**: Complex models offer limited explanations for predictions
- **Spurious Correlations**: Models may learn shortcuts instead of meaningful features
- **Adversarial Vulnerability**: Small perturbations can cause misclassifications
- **Trustworthiness**: Critical applications require reliable explanations

Solutions:
- Visualization techniques (CAM, Grad-CAM)
- Attribution methods (integrated gradients, SHAP)
- Attention mechanisms for highlighting important regions
- Model distillation to simpler, more interpretable models

### Architectural Limitations

Current CNN architectures have inherent limitations:

- **Spatial Invariance**: CNNs struggle with variations in scale and orientation
- **Long-range Dependencies**: Standard convolutions have limited receptive fields
- **Geometric Understanding**: 2D convolutions lack explicit 3D reasoning
- **Contextual Reasoning**: Difficulty capturing contextual relationships

Solutions:
- Multi-scale architectures
- Self-attention and non-local operations
- Capsule networks for better spatial relationships
- Graph-based representations for relational reasoning

---

## 🔧 Tools and Frameworks

### TensorFlow/Keras

Google's end-to-end machine learning platform:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Create a simple CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, 
          validation_data=(test_images, test_labels))
```

### PyTorch

Facebook's deep learning framework emphasizing flexibility:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model
model = SimpleCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### Model Deployment Tools

```python
# TensorFlow Lite conversion for mobile deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# ONNX conversion for cross-platform deployment
import torch.onnx

# Export PyTorch model to ONNX
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "model.onnx", 
                 input_names=["input"], output_names=["output"])
```

### Visualization Tools

```python
# TensorBoard for TensorFlow
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="./logs", histogram_freq=1
)

model.fit(train_images, train_labels, epochs=10,
          callbacks=[tensorboard_callback])

# Weights & Biases
import wandb
from wandb.keras import WandbCallback

wandb.init(project="cnn-project")
model.fit(train_images, train_labels, epochs=10,
          callbacks=[WandbCallback()])
```

---

## ❓ FAQ

### Q1: What's the difference between CNNs and traditional neural networks?

**A:** CNNs differ from traditional fully connected neural networks in several key ways:

1. **Parameter Sharing**: CNNs use the same weights across different locations in the input, dramatically reducing parameters.

2. **Local Connectivity**: Each neuron connects only to a small region of the input, unlike fully connected networks where each neuron connects to all inputs.

3. **Spatial Hierarchy**: Through pooling and strided convolutions, CNNs learn features at different scales.

4. **Translation Invariance**: CNNs can recognize patterns regardless of their position in the input.

5. **Parameter Efficiency**: For image data, CNNs require far fewer parameters (often 10-100x fewer) than equivalent fully connected networks.

This architecture makes CNNs particularly well-suited for grid-like data such as images, where spatial relationships between pixels are important and patterns may appear in different locations.

### Q2: How do I choose the right CNN architecture for my problem?

**A:** Selecting the right CNN architecture depends on several factors:

1. **Dataset Size**:
   - **Small datasets** (< 10K images): Use transfer learning with pre-trained models
   - **Medium datasets** (10K-100K images): Fine-tune pre-trained models
   - **Large datasets** (> 100K images): Consider training custom architectures

2. **Computational Resources**:
   - **Limited resources**: MobileNet, EfficientNet, or ShuffleNet
   - **Moderate resources**: ResNet-50, DenseNet-121
   - **High-end resources**: EfficientNetB7, ResNet-152, Vision Transformer

3. **Task Type**:
   - **Classification**: ResNet, EfficientNet, VGG
   - **Object Detection**: SSD, YOLO, Faster R-CNN
   - **Segmentation**: U-Net, DeepLabV3, Mask R-CNN
   - **Face Recognition**: FaceNet, ArcFace
   - **Super-Resolution**: SRCNN, ESRGAN

4. **Inference Speed Requirements**:
   - **Real-time on mobile**: MobileNet, MnasNet
   - **Real-time on desktop/server**: ResNet-50, EfficientNetB0-B2
   - **Batch processing**: Larger models (ResNet-101, EfficientNetB7)

5. **Accuracy Requirements**:
   - **Higher accuracy priority**: EfficientNet, ResNeXt, Vision Transformer
   - **Balanced accuracy/speed**: ResNet, DenseNet
   - **Speed priority**: MobileNet, ShuffleNet

When in doubt, start with a pre-trained ResNet-50 or EfficientNetB0 as a baseline, then experiment with alternatives based on your results.

### Q3: How can I prevent overfitting in CNNs?

**A:** Overfitting occurs when a model performs well on training data but poorly on new data. Here are effective strategies to prevent overfitting in CNNs:

1. **Data Augmentation**:
   ```python
   data_augmentation = tf.keras.Sequential([
       layers.RandomFlip("horizontal"),
       layers.RandomRotation(0.1),
       layers.RandomZoom(0.1),
       layers.RandomTranslation(0.1, 0.1),
       layers.RandomContrast(0.1),
   ])
   ```

2. **Regularization**:
   ```python
   # L2 regularization
   layers.Conv2D(64, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(0.001))
   
   # Dropout
   layers.Dropout(0.5)
   ```

3. **Batch Normalization**:
   ```python
   layers.Conv2D(64, (3, 3))
   layers.BatchNormalization()
   layers.Activation('relu')
   ```

4. **Early Stopping**:
   ```python
   early_stopping = tf.keras.callbacks.EarlyStopping(
       monitor='val_loss',
       patience=10,
       restore_best_weights=True
   )
   ```

5. **Transfer Learning**:
   ```python
   base_model = tf.keras.applications.ResNet50(
       weights='imagenet',
       include_top=False,
       input_shape=(224, 224, 3)
   )
   base_model.trainable = False  # Freeze base model
   ```

6. **Reduce Model Complexity**:
   - Use fewer layers and filters
   - Add global pooling instead of flattening
   
7. **Cross-Validation**:
   ```python
   from sklearn.model_selection import KFold
   
   kf = KFold(n_splits=5, shuffle=True, random_state=42)
   for train_idx, val_idx in kf.split(X):
       X_train, X_val = X[train_idx], X[val_idx]
       y_train, y_val = y[train_idx], y[val_idx]
       # Train and evaluate model
   ```

Typically, a combination of these techniques yields the best results. Start with data augmentation and regularization, then add other methods as needed.

### Q4: What are the trade-offs between different CNN architectures?

**A:** Different CNN architectures offer various trade-offs:

| Architecture | Pros | Cons | Best Use Cases |
|--------------|------|------|---------------|
| **VGG** | Simple, uniform design<br>Good feature extraction<br>Easy to understand | Very large (138M+ parameters)<br>Computationally expensive<br>Slower inference | Feature extraction<br>Transfer learning<br>Baseline model |
| **ResNet** | Solves vanishing gradient<br>Scales well to deep networks<br>Good accuracy/speed trade-off | Still quite large<br>Many variants to choose from | General purpose<br>When moderate depth needed<br>Default choice for many tasks |
| **Inception** | Efficient multi-scale processing<br>Fewer parameters than VGG<br>Good performance | Complex implementation<br>Many design choices<br>Hyperparameter tuning | When input features appear at multiple scales<br>Limited computing budget |
| **DenseNet** | Feature reuse through dense connections<br>Parameter efficient<br>Strong feature propagation | Memory intensive during training<br>Slower convergence | Smaller datasets<br>When parameter efficiency is key |
| **MobileNet** | Very small, fast inference<br>Designed for mobile devices<br>Low memory footprint | Lower accuracy than larger models<br>Struggles with fine details | Mobile applications<br>Edge devices<br>Real-time processing |
| **EfficientNet** | State-of-the-art accuracy<br>Scalable architecture<br>Optimized efficiency | Complex to implement from scratch<br>Newer, less established | When accuracy is critical<br>Production systems<br>SOTA requirements |

The key trade-offs to consider are:
- **Accuracy vs. Speed**: Larger models (EfficientNet, ResNet) are more accurate but slower
- **Size vs. Performance**: Smaller models (MobileNet, ShuffleNet) fit on edge devices but sacrifice accuracy
- **Simplicity vs. Effectiveness**: Simple architectures (VGG) are easier to understand but less efficient
- **Training Time vs. Inference Time**: Some models train slower but infer faster

Choose based on your specific constraints and requirements.

### Q5: How can I visualize and interpret what my CNN is learning?

**A:** Several techniques help visualize and interpret CNN features:

1. **Convolutional Filter Visualization**:
   ```python
   def visualize_filters(model, layer_name):
       layer = model.get_layer(layer_name)
       filters = layer.get_weights()[0]
       
       # Normalize filter values to 0-1
       f_min, f_max = filters.min(), filters.max()
       filters = (filters - f_min) / (f_max - f_min)
       
       # Plot filters
       n_filters = filters.shape[3]
       size = int(np.sqrt(n_filters))
       
       plt.figure(figsize=(20, 20))
       for i in range(n_filters):
           plt.subplot(size, size, i+1)
           plt.imshow(filters[:, :, 0, i])
           plt.axis('off')
       plt.tight_layout()
       plt.show()
   ```

2. **Feature Maps Visualization**:
   ```python
   def visualize_feature_maps(model, img, layer_name):
       # Create a model that outputs the feature maps
       feature_map_model = tf.keras.Model(
           inputs=model.inputs,
           outputs=model.get_layer(layer_name).output
       )
       
       # Get feature maps
       feature_maps = feature_map_model.predict(np.expand_dims(img, axis=0))
       feature_maps = feature_maps[0]
       
       # Plot feature maps
       n_features = feature_maps.shape[-1]
       size = int(np.ceil(np.sqrt(n_features)))
       
       plt.figure(figsize=(20, 20))
       for i in range(n_features):
           plt.subplot(size, size, i+1)
           plt.imshow(feature_maps[:, :, i], cmap='viridis')
           plt.axis('off')
       plt.tight_layout()
       plt.show()
   ```

3. **Class Activation Maps (CAM)**:
   ```python
   def generate_cam(model, img, class_idx):
       # Get the 'pre-softmax' layer
       cam_model = tf.keras.Model(
           inputs=model.input,
           outputs=[model.get_layer('final_conv').output, model.output]
       )
       
       # Get the weights from the softmax layer
       weights = model.layers[-1].get_weights()[0]
       
       # Get feature maps and predictions
       features, results = cam_model.predict(np.expand_dims(img, axis=0))
       features = features[0]
       
       # Create CAM
       cam = np.zeros(features.shape[0:2], dtype=np.float32)
       for i, w in enumerate(weights[:, class_idx]):
           cam += w * features[:, :, i]
       
       # Apply ReLU
       cam = np.maximum(cam, 0)
       
       # Normalize
       cam = cam / np.max(cam)
       
       # Resize to image size
       cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
       
       return cam
   ```

4. **Grad-CAM**:
   ```python
   def grad_cam(model, img, class_idx, layer_name):
       # Create a model that maps the input to the layer and predictions
       grad_model = tf.keras.Model(
           inputs=model.inputs,
           outputs=[model.get_layer(layer_name).output, model.output]
       )
       
       # Calculate gradients
       with tf.GradientTape() as tape:
           conv_outputs, predictions = grad_model(np.expand_dims(img, axis=0))
           loss = predictions[:, class_idx]
       
       # Extract gradients
       grads = tape.gradient(loss, conv_outputs)
       
       # Global average pooling of gradients
       pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
       
       # Weight feature maps with gradients
       conv_outputs = conv_outputs[0]
       heatmap = tf.reduce_sum(
           tf.multiply(pooled_grads, conv_outputs), axis=-1
       ).numpy()
       
       # Apply ReLU and normalize
       heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
       
       return heatmap
   ```

5. **t-SNE Visualization of Features**:
   ```python
   from sklearn.manifold import TSNE
   
   def visualize_embeddings(model, images, labels, layer_name):
       # Create a model that outputs the layer activations
       feature_model = tf.keras.Model(
           inputs=model.input,
           outputs=model.get_layer(layer_name).output
       )
       
       # Extract features
       features = feature_model.predict(images)
       
       # Flatten if needed
       if len(features.shape) > 2:
           features = features.reshape(features.shape[0], -1)
       
       # Apply t-SNE
       tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
       embeddings = tsne.fit_transform(features)
       
       # Plot
       plt.figure(figsize=(10, 10))
       for class_id in np.unique(labels):
           idx = labels == class_id
           plt.scatter(embeddings[idx, 0], embeddings[idx, 1], label=f'Class {class_id}')
       plt.legend()
       plt.show()
   ```

These visualization techniques help understand what features the network learns at different layers and how it makes decisions, making the "black box" more transparent.

---

<div align="center">

## 🌟 Key Takeaways

**Convolutional Neural Networks:**
- Use weight sharing and local connectivity to efficiently process grid-like data
- Learn hierarchical features from low-level patterns to high-level concepts
- Excel at computer vision tasks like classification, detection, and segmentation
- Benefit from architectural innovations like residual connections and attention
- Can be optimized for various deployment scenarios from cloud to edge devices
- Continue to evolve with new architectures, training methods, and applications

**Remember:**
- Choose architectures based on your specific task and constraints
- Data quality and augmentation are often more important than model complexity
- Transfer learning is powerful when working with limited data
- Regularization techniques prevent overfitting and improve generalization
- Visualization tools help understand model behavior and diagnose issues
- CNNs are foundational to modern computer vision but have applications beyond images

---

### 📖 Happy Deep Learning! 🚀

*Created by [@danialasim](https://github.com/danialasim) | Last updated: August 10, 2025*

</div>