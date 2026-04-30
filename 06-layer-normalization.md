# 06. Layer Normalization in Transformers

## 🤔 Opening Questions

1. **What is normalization and why do we need it?**
2. **Why can't we use Batch Normalization in Transformers?**
3. **What's the problem with padding in sequential data?**
4. **How does Layer Normalization solve these issues?**
5. **Where exactly do we apply normalization in Transformers?**

Let's understand why Layer Normalization is critical for Transformers!

---

## 📊 What is Normalization?

### Definition

**Normalization transforms data or model outputs to have specific statistical properties, typically:**
- **Mean = 0**
- **Variance (Standard Deviation) = 1**

<div align="center">
  <img src="https://miro.medium.com/max/1400/1*Hiq-rLFGDpESpr8QNsJ1jg.png" alt="Normalization Concept" width="500">
  <p><em>Normalization: Transforming features to have mean=0 and variance=1</em></p>
</div>

---

### 💭 Why Do We Normalize?

#### Example: Raw Data

```python
# Input features with different scales
f1 = [1, 27, 23, 51]      # Range: 1-51
f2 = [2, 3, 7, 5, 4]      # Range: 2-7

# Problem: Neural networks struggle with vastly different scales!
```

**What happens without normalization:**
```
Layer 1 → Large values → Large activations
       → Exploding gradients ❌

Layer 1 → Small values → Tiny activations  
       → Vanishing gradients ❌
```

---

### ✅ Benefits of Normalization

#### 1. **Improved Training Stability**
```
Without normalization:
  Gradients: [0.0001, 1000, 0.01, 500] → Unstable!
  
With normalization:
  Gradients: [0.5, 0.8, 0.3, 0.6] → Stable! ✓
```

Reduces likelihood of exploding/vanishing gradients.

---

#### 2. **Faster Convergence**

<div align="center">
  <img src="https://miro.medium.com/max/1400/1*8SJwr9ovIZXwOnuaR_8XrQ.png" alt="Convergence with Normalization" width="500">
  <p><em>Normalized models converge faster during training</em></p>
</div>

```
Training Loss:

Without Norm:    With Norm:
Epoch 1: 2.5     Epoch 1: 2.1  ← Faster start
Epoch 5: 1.8     Epoch 5: 0.9  ← Quicker convergence
Epoch 10: 1.2    Epoch 10: 0.3 ← Better final loss
```

More consistent gradient magnitudes → stable updates → faster learning.

---

#### 3. **Mitigating Internal Covariate Shift**

**Internal Covariate Shift:** Distribution of layer inputs changes during training as previous layers update.

```
Initially:
  Layer 2 input: mean=0, std=1
  
After 100 iterations:
  Layer 2 input: mean=5, std=10  ← Shifted!
  
Problem: Layer 2 must constantly adapt to new distributions
```

**Normalization fixes this** by keeping distributions stable.

---

#### 4. **Regularization Effect**

Some normalization techniques (like Batch Norm) add noise, which:
- Reduces overfitting
- Acts like dropout
- Improves generalization

---

## 🔄 Batch Normalization

### How It Works

**Normalize across the BATCH dimension.**

```python
# For a batch of samples
batch_size = 5
features = 3

# Data shape: (batch_size, features)
data = [
    [6.5, 2.41, 3.21],  # Sample 1
    [2.21, 0.4, 3.6],   # Sample 2
    [0, 0, 0],          # Sample 3
    [0, 0, 0],          # Sample 4
    [0, 0, 0]           # Sample 5
]

# Compute statistics ACROSS batch (for each feature)
# Feature 1: mean([6.5, 2.21, 0, 0, 0]) = 1.74
# Feature 2: mean([2.41, 0.4, 0, 0, 0]) = 0.56
# Feature 3: mean([3.21, 3.6, 0, 0, 0]) = 1.36

# Then normalize each feature using batch statistics
```

**Formula:**
```
For each feature j:
  μ_j = (1/batch_size) × Σ x_ij     (mean across batch)
  σ_j² = (1/batch_size) × Σ (x_ij - μ_j)²  (variance)
  
  x̂_ij = (x_ij - μ_j) / √(σ_j² + ε)   (normalize)
  
  y_ij = γ_j × x̂_ij + β_j    (scale and shift, learnable)
```

---

### 🎯 Batch Norm Example

```python
import numpy as np

# Network with batch norm
# Input batch (5 samples, 3 features)
X = np.array([
    [2, 3, 7],
    [1, 1, 2],
    [5, 4, 1],
    [6, 1, 7],
    [7, 1, 3]
])

# Hidden layer computation (before batch norm)
W = np.array([
    [2, 1],
    [3, 0.5],
    [1, 2]
])
b = np.array([1, 2])

Z = X @ W + b  # Shape: (5, 2)

print("Before Batch Norm:")
print(Z)
# Z₁ (feature 1): [2w₁+3w₂+7w₃+b₁, ...]
# Z₂ (feature 2): [2w₁+3w₂+7w₃+b₂, ...]

# Batch normalization
# For each feature (column), compute mean and std across batch (rows)
μ = Z.mean(axis=0)  # Mean for each feature
σ = Z.std(axis=0)   # Std for each feature

Z_norm = (Z - μ) / (σ + 1e-8)

print("\nAfter Batch Norm:")
print(Z_norm)
print(f"Mean: {Z_norm.mean(axis=0)}")  # ~[0, 0]
print(f"Std: {Z_norm.std(axis=0)}")    # ~[1, 1]
```

**Batch Norm normalizes ACROSS samples (vertically).**

---

## 🚨 The Problem: Batch Norm + Sequential Data = Disaster!

### Why Batch Norm Fails in Transformers

**Issue 1: Variable Sequence Lengths**

```
Batch of sentences:
  1. "Hi Ali"                    (2 words)
  2. "How are you today"         (4 words)
  3. "I am good"                 (3 words)

Need to make them same length → PADDING!
```

---

### Padding Creates Zeros

```python
# After padding to max_length = 4
Sentence 1: ["Hi", "Ali", "<PAD>", "<PAD>"]
Sentence 2: ["How", "are", "you", "today"]
Sentence 3: ["I", "am", "good", "<PAD>"]

# Embeddings (simplified, dim=3)
# Real words have non-zero values
# Padding has ZERO vectors

Batch (4 positions, 3 features):
Position 0: [
  [0.2, 0.45, 0.71],   # "Hi"
  [0.1, 0.5, 0.34],    # "How"  
  [0.24, 0.3, 0.9]     # "I"
]

Position 1: [
  [0.21, 0.3, 0.9],    # "Ali"
  [0.1, 0.0, 0.25],    # "are"
  [0.33, 0.56, 0.4]    # "am"
]

Position 2: [
  [0, 0, 0],           # <PAD>
  [0.24, 0.56, 0.4],   # "you"
  [0.11, 0.4, 0.54]    # "good"
]

Position 3: [
  [0, 0, 0],           # <PAD>
  [0.11, 0.4, 0.54],   # "today"
  [0, 0, 0]            # <PAD>
]
```

---

### 💥 Batch Norm Catastrophe

**Computing batch statistics at position 2:**

```python
# Position 2, feature 1
values = [0, 0.24, 0.11]  # Two zeros from padding!

μ = (0 + 0.24 + 0.11) / 3 = 0.117
σ² = variance([0, 0.24, 0.11]) = 0.012

# Normalized:
# Real word "you": (0.24 - 0.117) / √0.012 = 1.12  ✓
# Padding: (0 - 0.117) / √0.012 = -1.07  ← WRONG!
```

**Problems:**
1. ❌ Padding zeros distort mean (pulls it toward 0)
2. ❌ Padding zeros distort variance (reduces it)
3. ❌ Normalized padding becomes non-zero (-1.07)!
4. ❌ Real words get incorrect normalization

---

### 📊 Visual Example: The Disaster

**Review sentiment dataset:**

```
Batch of reviews:
  Review 1: "Hi Ali"                 → Sentiment: 1
  Review 2: "How are you today"      → Sentiment: 0  
  Review 3: "I am good"              → Sentiment: 0
  Review 4: "You?"                   → Sentiment: 1

Embedding dimension: 3
Batch size: 2
```

**With padding (max_length = 4):**

```python
# Batch 1 (Reviews 1 & 2):
# Sentence 1: ["Hi", "Ali", "<PAD>", "<PAD>"]
# Sentence 2: ["How", "are", "you", "today"]

Embeddings after self-attention:
Batch_1 = [
    # Position: 0       1       2       3
    [[0.2, 0.45, 0.71], [0.21, 0.3, 0.9], [0, 0, 0], [0, 0, 0]],  # Review 1
    [[0.1, 0.5, 0.34], [0.1, 0.0, 0.25], [0.24, 0.56, 0.4], [0.11, 0.4, 0.54]]  # Review 2
]

# Batch Norm at position 2, feature 1:
μ₁ = (0 + 0.24) / 2 = 0.12
σ₁ = std([0, 0.24]) = 0.12

# Normalized position 2, feature 1:
# Padding: (0 - 0.12) / 0.12 = -1.0    ← Should be 0!
# "you": (0.24 - 0.12) / 0.12 = 1.0    ← Distorted by padding!
```

**The padding contaminates the statistics!**

---

### 🎯 Why This is Catastrophic

```
Correct behavior:
  Padding should be IGNORED
  Statistics computed only from real words
  
Batch Norm behavior:
  Padding INCLUDED in statistics
  Mean/variance contaminated
  Real words incorrectly normalized
  Padding becomes non-zero
  
Result: MODEL CAN'T LEARN! ❌
```

---

## ✅ Solution: Layer Normalization

### The Key Difference

**Batch Norm:** Normalize across **batch** (different samples, same feature)
**Layer Norm:** Normalize across **features** (same sample, different features)

```
Batch Norm:    Layer Norm:
(vertical)     (horizontal)

Sample 1 →     Sample 1 → [normalize across these]
Sample 2 →     Sample 2 → [normalize across these]
Sample 3 →     Sample 3 → [normalize across these]
  ↓ ↓ ↓
normalize
across
these
```

---

### How Layer Norm Works

**For EACH sample independently, normalize across all its features.**

```python
# Same example as before
# But now normalize horizontally (per sample)

Sample 1 (word "you"): [0.24, 0.56, 0.4]
  μ = (0.24 + 0.56 + 0.4) / 3 = 0.4
  σ = std([0.24, 0.56, 0.4]) = 0.133
  
  normalized = [(0.24-0.4)/0.133, (0.56-0.4)/0.133, (0.4-0.4)/0.133]
             = [-1.2, 1.2, 0.0]

Sample 2 (padding): [0, 0, 0]
  μ = 0
  σ = 0
  
  normalized = [0, 0, 0]  ← Stays zero! ✓
```

**Padding doesn't contaminate other samples!**

---

### 🔢 Layer Norm Formula

```
For each sample (word embedding):

μ = (1/d) × Σ xᵢ              (mean across features)
σ² = (1/d) × Σ (xᵢ - μ)²      (variance across features)

x̂ᵢ = (xᵢ - μ) / √(σ² + ε)    (normalize)

yᵢ = γᵢ × x̂ᵢ + βᵢ             (scale and shift, learnable)

where:
  d = number of features (embedding dimension)
  ε = small constant (e.g., 1e-8) for numerical stability
  γ, β = learnable parameters
```

---

### 🧮 Complete Example

**Sentence:** "Hi Ali How are you today"

```python
import numpy as np

# After self-attention, we have:
# Shape: (batch_size=2, seq_len=4, d_model=3)

# Batch 1: ["Hi", "Ali", "<PAD>", "<PAD>"]
# Batch 2: ["How", "are", "you", "today"]

X = np.array([
    # Review 1
    [[0.2, 0.45, 0.71],   # Hi
     [0.21, 0.3, 0.9],    # Ali
     [0, 0, 0],           # <PAD>
     [0, 0, 0]],          # <PAD>
    
    # Review 2
    [[0.1, 0.5, 0.34],    # How
     [0.1, 0.0, 0.25],    # are
     [0.24, 0.56, 0.4],   # you
     [0.11, 0.4, 0.54]]   # today
])

def layer_norm(x, eps=1e-8):
    """
    Apply layer normalization.
    x: (batch, seq_len, features)
    """
    # Compute mean and std across features (axis=-1)
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    
    # Normalize
    x_norm = (x - mean) / (std + eps)
    
    return x_norm

X_norm = layer_norm(X)

print("Original:")
print(X[0])  # Review 1

print("\nLayer Normalized:")
print(X_norm[0])

print("\nPadding stays zero:")
print(X_norm[0, 2])  # [0, 0, 0] ✓
print(X_norm[0, 3])  # [0, 0, 0] ✓
```

**Output:**
```
Original Review 1:
[[0.2  0.45 0.71]   # Hi
 [0.21 0.3  0.9 ]   # Ali
 [0.   0.   0.  ]   # <PAD>
 [0.   0.   0.  ]]  # <PAD>

Layer Normalized Review 1:
[[-1.21  0.12  1.09]   # Hi (normalized across its 3 features)
 [-1.25 -0.47  1.72]   # Ali (normalized across its 3 features)
 [ 0.    0.    0.  ]   # <PAD> (stays zero!)
 [ 0.    0.    0.  ]]  # <PAD> (stays zero!)
```

---

## 🎯 Key Differences Summary

| Aspect | Batch Normalization | Layer Normalization |
|--------|-------------------|-------------------|
| **Normalize across** | Batch (samples) | Features (dimensions) |
| **For each** | Feature | Sample |
| **Independence** | Samples depend on each other | Each sample independent |
| **Padding effect** | Contaminates statistics ❌ | No contamination ✓ |
| **Sequential data** | Fails ❌ | Works perfectly ✓ |
| **Used in** | CNNs, MLPs | Transformers, RNNs |

---

## 📍 Where is Layer Norm Applied in Transformers?

### Two Main Locations

```
Transformer Block:
  
  Input
    ↓
  [Multi-Head Attention]
    ↓
  Add & Norm ← Layer Norm applied here!
    ↓
  [Feed Forward Network]
    ↓
  Add & Norm ← Layer Norm applied here!
    ↓
  Output
```

**Layer Norm is applied:**
1. After multi-head attention (before residual connection)
2. After feed-forward network (before residual connection)

---

### Complete Flow

```python
# Pseudocode for one transformer block

def transformer_block(x):
    # Multi-head attention
    attn_output = multi_head_attention(x)
    
    # Add & Norm 1
    x = layer_norm(x + attn_output)  # Residual + Layer Norm
    
    # Feed forward
    ff_output = feed_forward(x)
    
    # Add & Norm 2
    x = layer_norm(x + ff_output)  # Residual + Layer Norm
    
    return x
```

---

## ✅ Why Layer Norm is Perfect for Transformers

1. **Padding-safe:** Zeros stay zeros ✓
2. **Sample-independent:** Each word normalized independently ✓
3. **Sequence-length agnostic:** Works for any length ✓
4. **Stable training:** Consistent normalization ✓
5. **Fast convergence:** Helps gradient flow ✓

---

## 🎯 Key Takeaways

1. **Normalization** stabilizes training by standardizing distributions (mean=0, std=1)

2. **Batch Norm** normalizes across samples
   - Good for: CNNs, standard feedforward networks
   - Bad for: Sequential data with padding

3. **Padding problem:**
   - Variable length sequences need padding
   - Padding = zeros
   - Batch Norm includes zeros in statistics
   - Contaminates normalization ❌

4. **Layer Norm** normalizes across features
   - Each sample processed independently
   - Padding stays zero
   - Perfect for Transformers ✓

5. **Applied twice per Transformer block:**
   - After multi-head attention
   - After feed-forward network

---

## 🚀 Next: The Complete Transformer Architecture

Now we understand:
- ✅ Self-Attention
- ✅ Multi-Head Attention  
- ✅ Positional Encoding
- ✅ Layer Normalization

**Next:** Putting everything together in the Encoder-Decoder architecture!

👉 Continue to: [07-encoder-decoder-architecture.md](07-encoder-decoder-architecture.md)

---

## 📓 Resources

🔗 **[Layer Normalization Paper](https://arxiv.org/abs/1607.06450)**
- Original paper by Ba, Kiros, Hinton

🔗 **[Understanding Layer Normalization](https://leimao.github.io/blog/Layer-Normalization/)**
- Detailed mathematical explanation

🔗 **[Batch Norm vs Layer Norm](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739)**
- Visual comparison

---

## 🧮 Implementation

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# Usage
layer_norm = LayerNorm(features=512)
x = torch.randn(32, 10, 512)  # (batch, seq, features)
normalized = layer_norm(x)
```

**That's Layer Normalization explained!** 🎉
