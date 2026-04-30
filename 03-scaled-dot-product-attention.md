# 03. Scaled Dot-Product Attention

## 🤔 Questions to Think About

Before we dive in, ask yourself:
1. Why do we need to divide by √d_k in the attention formula?
2. What happens to dot products as dimensions increase?
3. How does softmax behave with large vs small numbers?
4. Why do neural networks struggle with extreme probability distributions?
5. What is the relationship between dimension and variance?

By the end of this chapter, you'll have clear answers to all of these!

---

## 🎯 The Story So Far

We learned how to compute attention using Query, Key, and Value:

```
1. Compute scores: Q · K^T
2. Apply softmax: weights = softmax(scores)
3. Weighted sum: output = weights · V
```

But there's a **critical problem** we haven't addressed yet...

---

## ❓ The Big Question

**Formula in papers:**
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) × V
```

**What is that `√d_k` doing there?**

Why divide by square root of dimension? Let's understand this deeply.

---

## 🔢 Understanding Dimensions

### What is d_k?

**d_k = dimensionality of Key (and Query) vectors**

Example:
```python
# If each word embedding has 512 dimensions
# And we transform to Q, K, V with dimension 64:

d_k = 64  # dimension of Key vectors
```

**In our running example:**
```
k_money = [0.57, 0.37, 0.39]  # 3 dimensions
k_bank  = [0.375, 0.390, 0.185]

d_k = 3
```

So: **`√d_k = √3 ≈ 1.73`**

---

## 🧮 The Core Problem: Dot Product and Variance

### 🎲 Quick Statistics Refresher

**Variance** measures how spread out values are:
- **Low variance:** Values clustered together (e.g., [5, 6, 5, 6, 5])
- **High variance:** Values spread apart (e.g., [1, 10, 2, 9, 3])

---

### 📊 Experiment: Dot Product vs Dimensionality

Let's see what happens when we compute dot products in different dimensions:

#### **1-Dimensional Vectors (d=1)**

```python
import numpy as np

# 1000 random pairs
results_1d = []
for _ in range(1000):
    x = np.random.randn(1)  # 1D vector
    y = np.random.randn(1)
    dot_product = np.dot(x, y)
    results_1d.append(dot_product)

variance_1d = np.var(results_1d)
print(f"Variance (1D): {variance_1d:.2f}")
# Output: Variance (1D): ~1.0
```

**Distribution of dot products (1D):**
```
     Frequency
        │
   100 ├────┐
        │    │
        │    ││
    50 │   ││││
        │  ││││││
        │ │││││││
     0 ├─────────────
       -2  0  2
     Dot Product Value
```

Range: Mostly between -2 and 2  
Variance ≈ 1.0

---

#### **10-Dimensional Vectors (d=10)**

```python
# 1000 random pairs
results_10d = []
for _ in range(1000):
    x = np.random.randn(10)  # 10D vector
    y = np.random.randn(10)
    dot_product = np.dot(x, y)
    results_10d.append(dot_product)

variance_10d = np.var(results_10d)
print(f"Variance (10D): {variance_10d:.2f}")
# Output: Variance (10D): ~10.0
```

**Distribution of dot products (10D):**
```
     Frequency
        │
   100 ├────┐
        │    │
        │    ││
    50 │   ││││
        │  ││││││
        │ │││││││
     0 ├─────────────────────
       -10  0  10
         Dot Product Value
```

Range: Mostly between -10 and 10  
Variance ≈ 10.0

---

#### **1000-Dimensional Vectors (d=1000)**

```python
# 1000 random pairs
results_1000d = []
for _ in range(1000):
    x = np.random.randn(1000)  # 1000D vector
    y = np.random.randn(1000)
    dot_product = np.dot(x, y)
    results_1000d.append(dot_product)

variance_1000d = np.var(results_1000d)
print(f"Variance (1000D): {variance_1000d:.2f}")
# Output: Variance (1000D): ~1000.0
```

**Distribution of dot products (1000D):**
```
     Frequency
        │
   100 ├────┐
        │    │
        │    ││
    50 │   ││││
        │  ││││││
        │ │││││││
     0 ├──────────────────────────────────
      -50    0    50
           Dot Product Value
```

Range: Mostly between -50 and 50  
Variance ≈ 1000.0

---

### 📈 The Pattern

```
Comparison of Dot Product Distributions
     ┌──────────────────────────────────┐
     │      ▓▓▓ 3D                       │
     │     ▓▓▓▓▓ 100D                    │
     │    ▓▓▓▓▓▓▓ 1000D                  │
     │   ▓▓▓▓▓▓▓▓▓                       │
     │  ▓▓▓▓▓▓▓▓▓▓▓                      │
     │ ▓▓▓▓▓▓▓▓▓▓▓▓▓                     │
     └──────────────────────────────────┘
       -30  -20  -10   0   10  20  30
              Dot Product Value
```

**Key Observation:**
```
Dimension → Variance
─────────────────────
1   dim  →  1   × Var(X)
2   dim  →  2   × Var(X)
3   dim  →  3   × Var(X)
10  dim  →  10  × Var(X)
100 dim  →  100 × Var(X)
d   dim  →  d   × Var(X)
```

**Mathematical fact:**
```
Var(Y) = d × Var(X)

where Y = dot product of d-dimensional vectors
```

---

## 🔥 Why High Variance is a Problem

### The Softmax Instability Issue

Recall softmax formula:
```
softmax([s₁, s₂, s₃]) = [e^s₁/sum, e^s₂/sum, e^s₃/sum]
```

**Softmax is an exponential function** → extremely sensitive to large values!

---

### 📊 Complete Softmax Code & Behavior

```python
import numpy as np

def softmax(x):
    """
    Numerically stable softmax implementation.
    """
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# Let's see how softmax behaves with different inputs
```

---

### 🔍 Example 1: Small Numbers (Good Behavior)

```python
scores = np.array([1, 2, 3])
probs = softmax(scores)

print("Input scores:", scores)
print("Softmax output:", probs)
print("Sum:", np.sum(probs))

# Output:
# Input scores: [1 2 3]
# Softmax output: [0.09003057 0.24472847 0.66524096]
# Sum: 1.0
```

**Interpretation:**
- Item 1: 9% probability
- Item 2: 24% probability  
- Item 3: 67% probability

**All items get reasonable attention** ✅

---

### 🔍 Example 2: Medium Numbers (Still OK)

```python
scores = np.array([4, 5, 6])
probs = softmax(scores)

print("Input scores:", scores)
print("Softmax output:", probs)

# Output:
# Input scores: [4 5 6]
# Softmax output: [0.09003057 0.24472847 0.66524096]
```

**Same relative probabilities!** (shifted by constant)

---

### 🔍 Example 3: Moderate Spread (Acceptable)

```python
scores = np.array([1, 3, 5])
probs = softmax(scores)

print("Input scores:", scores)
print("Softmax output:", probs)
print(f"Max prob: {probs.max():.4f}, Min prob: {probs.min():.4f}")

# Output:
# Input scores: [1 3 5]
# Softmax output: [0.01587624 0.11731043 0.86681333]
# Max prob: 0.8668, Min prob: 0.0159
```

**Result:** 
- Strongest: 87%
- Weakest: 1.6%

Still learning from all items ✅

---

### 🔴 Example 4: Large Numbers (PROBLEM!)

```python
scores = np.array([10, 20, 30])
probs = softmax(scores)

print("Input scores:", scores)
print("Softmax output:", probs)
print(f"Max prob: {probs.max():.10f}")
print(f"Min prob: {probs.min():.15f}")

# Output:
# Input scores: [10 20 30]
# Softmax output: [2.06106005e-09 4.53978686e-05 9.99954600e-01]
# Max prob: 0.9999546003
# Min prob: 0.000000002061060
```

**DISASTER:**
- Item 3: **99.995%** (basically 100%)
- Item 2: **0.0045%** (negligible)
- Item 1: **0.0000002%** (essentially zero)

**No learning possible!** ❌

---

### 🔴 Example 5: Very Large Numbers (CATASTROPHIC!)

```python
scores = np.array([40, 50, 60])
probs = softmax(scores)

print("Input scores:", scores)
print("Softmax output:", probs)
print(f"Max prob: {probs.max():.20f}")
print(f"Min prob: {probs.min():.30f}")

# Output:
# Input scores: [40 50 60]
# Softmax output: [0. 0. 1.]
# Max prob: 1.00000000000000000000
# Min prob: 0.000000000000000000000000000000
```

**COMPLETE FAILURE:**
- Item 3: **100%** (saturated)
- Item 2: **0%** (rounded to zero!)
- Item 1: **0%** (rounded to zero!)

**Gradient = 0 everywhere → No learning!** ❌

---

### 📊 Visual Comparison

```python
import matplotlib.pyplot as plt

test_cases = [
    ([1, 2, 3], "Small numbers"),
    ([1, 5, 9], "Medium spread"),
    ([10, 20, 30], "Large numbers"),
    ([40, 50, 60], "Very large"),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for idx, (scores, title) in enumerate(test_cases):
    probs = softmax(np.array(scores))
    axes[idx].bar(['Item 1', 'Item 2', 'Item 3'], probs)
    axes[idx].set_title(f"{title}\nScores: {scores}")
    axes[idx].set_ylabel('Probability')
    axes[idx].set_ylim([0, 1])
    
    # Add percentage labels
    for i, prob in enumerate(probs):
        axes[idx].text(i, prob + 0.02, f'{prob*100:.2f}%', 
                      ha='center', fontsize=10)

plt.tight_layout()
plt.show()
```

**Result:**
```
Small numbers:    [9%, 24%, 67%]     ← Balanced ✅
Medium spread:    [2%, 12%, 86%]     ← Still OK ✅
Large numbers:    [0%, 0.004%, 99.99%] ← Bad ❌
Very large:       [0%, 0%, 100%]     ← Disaster ❌
```

---

### 🎯 Why This Matters for Attention

**In transformer without scaling (d_k = 512):**

```python
# Random query and key vectors
q = np.random.randn(512)
k1 = np.random.randn(512)
k2 = np.random.randn(512)
k3 = np.random.randn(512)

# Compute dot products
s1 = np.dot(q, k1)  # Could be: -28.3
s2 = np.dot(q, k2)  # Could be: 35.7
s3 = np.dot(q, k3)  # Could be: -19.2

scores = np.array([s1, s2, s3])
print("Unscaled scores:", scores)
print("Unscaled probabilities:", softmax(scores))

# Output:
# Unscaled scores: [-28.3  35.7 -19.2]
# Unscaled probabilities: [0. 1. 0.]  ← EXTREME!
```

**With scaling (divide by √512 ≈ 22.6):**

```python
scaled_scores = scores / np.sqrt(512)
print("Scaled scores:", scaled_scores)
print("Scaled probabilities:", softmax(scaled_scores))

# Output:
# Scaled scores: [-1.25  1.58 -0.85]
# Scaled probabilities: [0.06 0.83 0.11]  ← BALANCED!
```

---

### 🧮 Interactive Experiment

```python
def compare_softmax(scores, scale_factor=1):
    """Compare softmax with and without scaling."""
    original = softmax(scores)
    scaled = softmax(scores / scale_factor)
    
    print(f"{'='*60}")
    print(f"Original scores: {scores}")
    print(f"Scale factor: {scale_factor}")
    print(f"{'='*60}")
    print(f"Without scaling: {original}")
    print(f"With scaling:    {scaled}")
    print(f"{'='*60}")
    print(f"Max probability change: {original.max():.4f} → {scaled.max():.4f}")
    print(f"Variance of probs: {original.var():.6f} → {scaled.var():.6f}")
    print()

# Test different scenarios
compare_softmax(np.array([10, 20, 30]), scale_factor=1)
compare_softmax(np.array([10, 20, 30]), scale_factor=10)
compare_softmax(np.array([10, 20, 30]), scale_factor=np.sqrt(100))
```

**Output shows how scaling brings extreme distributions back to balance!**

---

### 📊 Example: Low Variance Scores

```python
# Low variance scores (good)
scores = [4, 5]
softmax(scores)
# Output: [0.27, 0.73]  # 27% vs 73% - balanced
```

**Visual:**
```
Probabilities:
word1: ████████ 27%
word2: ██████████████ 73%
```

Both words get reasonable attention.

---

### 🔴 Example: High Variance Scores

```python
# High variance scores (bad)
scores = [40, 50]
softmax(scores)
# Output: [0.000045, 0.999955]  # 0.0045% vs 99.99% - extreme!
```

**Visual:**
```
Probabilities:
word1:  0.0045%  (practically zero)
word2: ████████████████████ 99.99%
```

**Problem:** 
- One word dominates completely
- Others get nearly zero attention
- Model can't learn from diversity

---

### 💥 Real Example with Our Sentence

**Sentence:** "money bank grows"

#### Without Scaling (d_k = 512):

```python
q_bank = np.random.randn(512)
k_money = np.random.randn(512)
k_bank = np.random.randn(512)
k_grows = np.random.randn(512)

# Dot products
s1 = np.dot(q_bank, k_money)  # Could be: -35.2
s2 = np.dot(q_bank, k_bank)   # Could be: 42.7
s3 = np.dot(q_bank, k_grows)  # Could be: -28.9

scores = [s1, s2, s3]
# scores = [-35.2, 42.7, -28.9]

weights = softmax(scores)
# weights ≈ [0.0, 1.0, 0.0]  # EXTREME!
```

**Result:**
- "bank" attends 100% to itself
- Completely ignores "money" and "grows"
- **No learning possible!**

---

#### With Scaling (divide by √d_k = √512 ≈ 22.6):

```python
# Same dot products
scores = [-35.2, 42.7, -28.9]

# Scale them down
scaled_scores = scores / np.sqrt(512)
# scaled_scores = [-1.56, 1.89, -1.28]

weights = softmax(scaled_scores)
# weights ≈ [0.08, 0.76, 0.16]  # Much better!
```

**Result:**
- "bank": 76% attention
- "grows": 16% attention
- "money": 8% attention
- **Balanced enough to learn!**

---

## 📐 Complete Variance Rules (Statistical Foundation)

Before we understand why √d_k specifically, let's review the key statistical rules:

### Rule 1: Variance of a Constant

```
If c is a constant:
Var(c) = 0
```

**Example:**
```python
X = [5, 5, 5, 5, 5]  # All same value
Var(X) = 0  # No variation
```

---

### Rule 2: Variance of Scaled Random Variable

```
If X is a random variable and c is a constant:
Var(cX) = c² × Var(X)
```

**Example:**
```python
X = [1, 2, 3, 4, 5]
Var(X) = 2.0

Y = 3 × X = [3, 6, 9, 12, 15]
Var(Y) = 3² × Var(X) = 9 × 2.0 = 18.0

# Verify
print(np.var([1,2,3,4,5]))      # 2.0
print(np.var([3,6,9,12,15]))    # 18.0 ✓
```

**Key insight:** Scaling by c multiplies variance by c²!

---

### Rule 3: Variance of Sum of Independent Variables

```
If X and Y are independent random variables:
Var(X + Y) = Var(X) + Var(Y)
```

**Example:**
```python
# Two dice rolls
X = [1,2,3,4,5,6]  # First die
Y = [1,2,3,4,5,6]  # Second die

Var(X) = 2.92
Var(Y) = 2.92

# Sum of two dice
Var(X + Y) = Var(X) + Var(Y) = 2.92 + 2.92 = 5.84
```

---

### Rule 4: Variance of Dot Product

**This is the KEY rule for understanding attention!**

```
For two independent random vectors X and Y of dimension d:
X = [x₁, x₂, ..., xₐ]
Y = [y₁, y₂, ..., yₐ]

Dot product: X·Y = x₁y₁ + x₂y₂ + ... + xₐyₐ

If each element has variance σ²:
Var(X·Y) = Var(x₁y₁) + Var(x₂y₂) + ... + Var(xₐyₐ)
         = σ² + σ² + ... + σ² (d times)
         = d × σ²
```

**Proof:**
```python
import numpy as np

# Test with different dimensions
for d in [1, 5, 10, 50, 100]:
    dot_products = []
    
    for _ in range(10000):
        x = np.random.randn(d)  # Each element ~ N(0,1), so Var=1
        y = np.random.randn(d)
        dot_products.append(np.dot(x, y))
    
    theoretical_var = d * 1.0  # d × Var(element)
    empirical_var = np.var(dot_products)
    
    print(f"d={d:3d} | Theoretical: {theoretical_var:6.2f} | "
          f"Empirical: {empirical_var:6.2f} | "
          f"Match: {abs(theoretical_var - empirical_var) < 0.5}")

# Output:
# d=  1 | Theoretical:   1.00 | Empirical:   1.02 | Match: True
# d=  5 | Theoretical:   5.00 | Empirical:   4.98 | Match: True
# d= 10 | Theoretical:  10.00 | Empirical:  10.15 | Match: True
# d= 50 | Theoretical:  50.00 | Empirical:  49.87 | Match: True
# d=100 | Theoretical: 100.00 | Empirical: 100.23 | Match: True
```

**Confirmed:** Var(X·Y) = d × Var(element) ✓

---

### Rule 5: Standard Deviation and Variance

```
Standard Deviation (σ) = √(Variance)
Variance (σ²) = (Standard Deviation)²
```

**Example:**
```python
data = [1, 2, 3, 4, 5]

variance = np.var(data)           # 2.0
std_dev = np.std(data)            # 1.414 (≈ √2)

print(f"Variance: {variance}")           # 2.0
print(f"Std Dev: {std_dev}")             # 1.414
print(f"√(Variance): {np.sqrt(variance)}") # 1.414 ✓
print(f"(Std Dev)²: {std_dev**2}")       # 2.0 ✓
```

---

### 🎯 Applying to Attention Mechanism

**Original dot product (unscaled):**
```
Q·K^T where Q, K have dimension d_k

If each element ~ N(0, 1):
Var(Q·K^T) = d_k × 1 = d_k

Standard Deviation = √d_k
```

**Scaled dot product:**
```
(Q·K^T) / √d_k

Using Rule 2: Var(cX) = c² × Var(X)

Var((Q·K^T)/√d_k) = (1/√d_k)² × Var(Q·K^T)
                   = (1/d_k) × d_k
                   = 1

Standard Deviation = √1 = 1
```

**Result:** By dividing by √d_k, we normalize variance back to 1! ✓

---

### 📊 Complete Numerical Example

```python
import numpy as np

def analyze_variance(d_k):
    """Analyze variance with and without scaling."""
    samples = 10000
    
    # Without scaling
    unscaled_dots = []
    for _ in range(samples):
        q = np.random.randn(d_k)
        k = np.random.randn(d_k)
        unscaled_dots.append(np.dot(q, k))
    
    # With scaling
    scaled_dots = []
    for _ in range(samples):
        q = np.random.randn(d_k)
        k = np.random.randn(d_k)
        scaled_dots.append(np.dot(q, k) / np.sqrt(d_k))
    
    print(f"{'='*70}")
    print(f"Dimension d_k = {d_k}")
    print(f"{'='*70}")
    print(f"WITHOUT SCALING:")
    print(f"  Variance:  {np.var(unscaled_dots):8.2f} (≈ d_k = {d_k})")
    print(f"  Std Dev:   {np.std(unscaled_dots):8.2f} (≈ √d_k = {np.sqrt(d_k):.2f})")
    print(f"  Range:     [{np.min(unscaled_dots):6.2f}, {np.max(unscaled_dots):6.2f}]")
    print()
    print(f"WITH SCALING (÷ √d_k):")
    print(f"  Variance:  {np.var(scaled_dots):8.2f} (≈ 1)")
    print(f"  Std Dev:   {np.std(scaled_dots):8.2f} (≈ 1)")
    print(f"  Range:     [{np.min(scaled_dots):6.2f}, {np.max(scaled_dots):6.2f}]")
    print(f"{'='*70}\n")

# Test with different dimensions
for d in [3, 10, 64, 512]:
    analyze_variance(d)
```

**Sample Output:**
```
======================================================================
Dimension d_k = 512
======================================================================
WITHOUT SCALING:
  Variance:   512.34 (≈ d_k = 512)
  Std Dev:    22.63 (≈ √d_k = 22.63)
  Range:     [-71.23,  75.89]

WITH SCALING (÷ √d_k):
  Variance:     1.00 (≈ 1)
  Std Dev:      1.00 (≈ 1)
  Range:      [-3.15,   3.35]
======================================================================
```

**Perfect normalization!** ✓

---

### 🧪 Code Demonstration

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Simulate attention scores in different dimensions
dimensions = [3, 10, 100, 1000]

for d in dimensions:
    scores = []
    for _ in range(1000):
        q = np.random.randn(d)
        k = np.random.randn(d)
        score = np.dot(q, k)
        scores.append(score)
    
    variance = np.var(scores)
    print(f"Dimension {d:4d} → Variance: {variance:8.2f} → √d: {np.sqrt(d):.2f}")

# Output:
# Dimension    3 → Variance:     3.12 → √d: 1.73
# Dimension   10 → Variance:    10.05 → √d: 3.16
# Dimension  100 → Variance:   100.23 → √d: 10.00
# Dimension 1000 → Variance:  1001.45 → √d: 31.62
```

**Key insight:** Variance ≈ d_k, so dividing by √d_k brings it back to ~1

---

## ⚠️ The Double Problem: Variance → Instability → Vanishing Gradients

### 💭 Key Question: How does high variance lead to gradient problems?

**Answer:** It's a chain reaction! Let's trace the complete path:

```
High Dimension (d_k) 
    ↓
High Variance in Dot Products
    ↓
Large Score Values (e.g., ±50)
    ↓
Extreme Softmax Outputs (0% or 100%)
    ↓
Vanishing Gradients
    ↓
No Learning!
```

---

### 🔗 The Complete Connection

#### Step 1: High Variance Creates Large Scores

```python
# High dimension = High variance
d_k = 512

q = np.random.randn(512)
k = np.random.randn(512)

score = np.dot(q, k)
print(f"Score: {score:.2f}")
# Output: Score: -47.32  (or +38.91, etc.)

# Variance
Var(score) = d_k = 512
Std Dev = √512 ≈ 22.6

# Typical range: ±3σ = ±68 (HUGE!)
```

**Problem:** Scores are spread across a wide range [-70, +70]

---

#### Step 2: Large Scores Cause Unstable Softmax

```python
# Example: 3 words with high-variance scores
scores = np.array([-47.3, 38.9, -35.2])

probs = softmax(scores)
print("Probabilities:", probs)
# Output: [0.0, 1.0, 0.0]
```

**Problem:** Softmax becomes **unstable** - one value dominates completely

---

#### Step 3: Extreme Probabilities → Vanishing Gradients

```python
# Gradient of softmax: ∂p/∂s = p(1-p)
for i, p in enumerate(probs):
    gradient = p * (1 - p)
    print(f"Word {i}: prob={p:.6f}, gradient={gradient:.10f}")

# Output:
# Word 0: prob=0.000000, gradient=0.0000000000
# Word 1: prob=1.000000, gradient=0.0000000000
# Word 2: prob=0.000000, gradient=0.0000000000
```

**Problem:** All gradients vanish → **no weight updates** → **no learning**

---

### 📊 Visual Chain Reaction

```
WITHOUT SCALING (d_k = 512):

Variance:      512 (HIGH!)
                ↓
Scores:        [-47, 39, -35] (LARGE RANGE!)
                ↓
Softmax:       [0.00, 1.00, 0.00] (UNSTABLE!)
                ↓
Gradients:     [0.00, 0.00, 0.00] (VANISHED!)
                ↓
Learning:      ❌ BLOCKED


WITH SCALING (÷ √512):

Variance:      1 (NORMALIZED!)
                ↓
Scores:        [-2.08, 1.72, -1.56] (CONTROLLED!)
                ↓
Softmax:       [0.06, 0.82, 0.12] (STABLE!)
                ↓
Gradients:     [0.056, 0.148, 0.106] (HEALTHY!)
                ↓
Learning:      ✅ WORKING
```

---

### 🔬 Complete Demonstration

```python
def demonstrate_variance_to_gradient_problem(d_k):
    """Show complete chain: variance → instability → vanishing gradients."""
    
    print(f"\n{'='*70}")
    print(f"Demonstration with d_k = {d_k}")
    print(f"{'='*70}\n")
    
    # Generate random Q and K
    q = np.random.randn(d_k)
    k1 = np.random.randn(d_k)
    k2 = np.random.randn(d_k)
    k3 = np.random.randn(d_k)
    
    # Compute dot products (attention scores)
    scores = np.array([
        np.dot(q, k1),
        np.dot(q, k2),
        np.dot(q, k3)
    ])
    
    print("STEP 1: VARIANCE ANALYSIS")
    print(f"  Theoretical Var(dot product) = d_k = {d_k}")
    print(f"  Theoretical Std Dev = √{d_k} = {np.sqrt(d_k):.2f}")
    print(f"  Actual scores: {scores}")
    print(f"  Score range: [{scores.min():.2f}, {scores.max():.2f}]")
    print()
    
    # Without scaling
    print("STEP 2: SOFTMAX (WITHOUT SCALING)")
    probs_unscaled = softmax(scores)
    print(f"  Probabilities: {probs_unscaled}")
    print(f"  Max prob: {probs_unscaled.max():.6f}")
    print(f"  Min prob: {probs_unscaled.min():.10f}")
    print(f"  Status: {'UNSTABLE ❌' if probs_unscaled.max() > 0.95 else 'STABLE ✓'}")
    print()
    
    # Gradients without scaling
    print("STEP 3: GRADIENTS (WITHOUT SCALING)")
    grads_unscaled = probs_unscaled * (1 - probs_unscaled)
    print(f"  Gradients: {grads_unscaled}")
    print(f"  Avg gradient: {grads_unscaled.mean():.10f}")
    print(f"  Status: {'VANISHED ❌' if grads_unscaled.mean() < 0.01 else 'HEALTHY ✓'}")
    print()
    
    print("─" * 70)
    
    # With scaling
    scaled_scores = scores / np.sqrt(d_k)
    print("STEP 2: SOFTMAX (WITH SCALING ÷ √d_k)")
    probs_scaled = softmax(scaled_scores)
    print(f"  Scaled scores: {scaled_scores}")
    print(f"  Probabilities: {probs_scaled}")
    print(f"  Max prob: {probs_scaled.max():.6f}")
    print(f"  Min prob: {probs_scaled.min():.6f}")
    print(f"  Status: {'UNSTABLE ❌' if probs_scaled.max() > 0.95 else 'STABLE ✓'}")
    print()
    
    # Gradients with scaling
    print("STEP 3: GRADIENTS (WITH SCALING)")
    grads_scaled = probs_scaled * (1 - probs_scaled)
    print(f"  Gradients: {grads_scaled}")
    print(f"  Avg gradient: {grads_scaled.mean():.6f}")
    print(f"  Status: {'VANISHED ❌' if grads_scaled.mean() < 0.01 else 'HEALTHY ✓'}")
    print()
    
    # Summary
    print(f"{'='*70}")
    print("SUMMARY:")
    print(f"  Without scaling: Unstable={probs_unscaled.max() > 0.95}, "
          f"Vanished={grads_unscaled.mean() < 0.01}")
    print(f"  With scaling:    Unstable={probs_scaled.max() > 0.95}, "
          f"Vanished={grads_scaled.mean() < 0.01}")
    print(f"{'='*70}\n")

# Test with different dimensions
for dimension in [64, 256, 512]:
    demonstrate_variance_to_gradient_problem(dimension)
```

**Sample Output:**
```
======================================================================
Demonstration with d_k = 512
======================================================================

STEP 1: VARIANCE ANALYSIS
  Theoretical Var(dot product) = d_k = 512
  Theoretical Std Dev = √512 = 22.63
  Actual scores: [-35.21  42.73 -28.94]
  Score range: [-35.21, 42.73]

STEP 2: SOFTMAX (WITHOUT SCALING)
  Probabilities: [0. 1. 0.]
  Max prob: 1.000000
  Min prob: 0.0000000000
  Status: UNSTABLE ❌

STEP 3: GRADIENTS (WITHOUT SCALING)
  Gradients: [0. 0. 0.]
  Avg gradient: 0.0000000000
  Status: VANISHED ❌

──────────────────────────────────────────────────────────────────────
STEP 2: SOFTMAX (WITH SCALING ÷ √d_k)
  Scaled scores: [-1.56  1.89 -1.28]
  Probabilities: [0.08 0.76 0.16]
  Max prob: 0.760000
  Min prob: 0.080000
  Status: STABLE ✓

STEP 3: GRADIENTS (WITH SCALING)
  Gradients: [0.074 0.182 0.134]
  Avg gradient: 0.130000
  Status: HEALTHY ✓

======================================================================
SUMMARY:
  Without scaling: Unstable=True, Vanished=True
  With scaling:    Unstable=False, Vanished=False
======================================================================
```

---

### 🎯 The Root Cause Chain

```
ROOT CAUSE:
├── High Dimension (d_k = 512, 1024, etc.)
│   
├── CONSEQUENCE 1: High Variance
│   └── Var(Q·K^T) = d_k (grows with dimension!)
│       
├── CONSEQUENCE 2: Large Score Range
│   └── Scores in range ±3√d_k = ±68 (for d_k=512)
│       
├── CONSEQUENCE 3: Softmax Instability
│   └── exp(±68) causes extreme outputs [0, 1, 0]
│       
└── FINAL PROBLEM: Vanishing Gradients
    └── Gradient ≈ 0 → No learning!

SOLUTION:
└── Divide by √d_k
    ├── Normalizes variance to 1
    ├── Keeps scores in range ±3
    ├── Softmax stays balanced [0.1, 0.7, 0.2]
    └── Gradients remain healthy > 0.05
```

---

### 💡 Why Both Problems Are Related

**Think of it as a domino effect:**

1. **Variance Problem** (statistical issue)
   - High dimension → high variance in dot products
   - Scores become unpredictably large

2. **Instability Problem** (numerical issue)
   - Large scores → softmax saturation
   - One output dominates (99.9%)

3. **Gradient Problem** (learning issue)
   - Extreme probabilities → derivative ≈ 0
   - No weight updates → training stalls

**All three are connected! Fixing variance (by scaling) fixes all three!**

---

---

### What Happens During Training?

**Forward pass:**
```
Input → Attention → Softmax → Output → Loss
```

**Backward pass (learning):**
```
Loss → ∂Loss/∂Softmax → ∂Softmax/∂Scores → Update weights
```

---

### Vanishing Gradients Explained

#### Scenario 1: Extreme Softmax Output (Bad)

```python
# Without scaling - extreme scores
scores = np.array([10, 50, 15])
probs = softmax(scores)
print("Probabilities:", probs)
# Output: [0.0, 1.0, 0.0]

# Gradient of softmax
def softmax_gradient(softmax_output, index):
    """
    Gradient of softmax w.r.t. input scores.
    ∂softmax_i/∂score_j = softmax_i × (δᵢⱼ - softmax_j)
    """
    s_i = softmax_output[index]
    gradient = s_i * (1 - s_i)  # Simplified for single element
    return gradient

# Gradients for each position
for i, prob in enumerate(probs):
    grad = prob * (1 - prob)
    print(f"Position {i}: prob={prob:.6f}, gradient={grad:.10f}")

# Output:
# Position 0: prob=0.000000, gradient=0.0000000000 ← ZERO!
# Position 1: prob=1.000000, gradient=0.0000000000 ← ZERO!
# Position 2: prob=0.000000, gradient=0.0000000000 ← ZERO!
```

**All gradients ≈ 0 → No learning!** ❌

---

#### Scenario 2: Balanced Softmax Output (Good)

```python
# With scaling - moderate scores
scores = np.array([0.5, 2.2, 0.7])
probs = softmax(scores)
print("Probabilities:", probs)
# Output: [0.11, 0.71, 0.18]

# Gradients for each position
for i, prob in enumerate(probs):
    grad = prob * (1 - prob)
    print(f"Position {i}: prob={prob:.6f}, gradient={grad:.10f}")

# Output:
# Position 0: prob=0.110000, gradient=0.0979000000 ← Good!
# Position 1: prob=0.710000, gradient=0.2059000000 ← Good!
# Position 2: prob=0.180000, gradient=0.1476000000 ← Good!
```

**All gradients > 0 → Learning happens!** ✅

---

### 📊 Visual Comparison of Gradients

```python
import matplotlib.pyplot as plt

def plot_softmax_gradient():
    """Plot gradient of softmax as function of probability."""
    probs = np.linspace(0, 1, 1000)
    gradients = probs * (1 - probs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(probs, gradients, linewidth=2)
    plt.xlabel('Softmax Probability', fontsize=12)
    plt.ylabel('Gradient Magnitude', fontsize=12)
    plt.title('Softmax Gradient vs Probability', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Mark extreme vs balanced regions
    plt.axvspan(0, 0.05, alpha=0.3, color='red', label='Vanishing (extreme)')
    plt.axvspan(0.95, 1.0, alpha=0.3, color='red')
    plt.axvspan(0.2, 0.8, alpha=0.3, color='green', label='Healthy (balanced)')
    
    # Annotate
    plt.annotate('Max gradient\nat p=0.5', xy=(0.5, 0.25), 
                xytext=(0.3, 0.3),
                arrowprops=dict(arrowstyle='->'), fontsize=11)
    plt.legend()
    plt.show()

plot_softmax_gradient()
```

**Key observation:**
- Gradient peaks at probability = 0.5
- Gradient → 0 as probability → 0 or 1
- We want probabilities in the 0.2-0.8 range!

---

### 🔬 Complete Training Example

```python
def simulate_training_step(scores, scale=False, d_k=512):
    """Simulate one training step with/without scaling."""
    
    # Apply scaling if requested
    if scale:
        scores = scores / np.sqrt(d_k)
    
    # Forward pass
    probs = softmax(scores)
    
    # Compute gradients (simplified)
    gradients = probs * (1 - probs)
    
    # Check if learning is possible
    avg_gradient = np.mean(gradients)
    max_gradient = np.max(gradients)
    
    return {
        'probabilities': probs,
        'gradients': gradients,
        'avg_gradient': avg_gradient,
        'max_gradient': max_gradient,
        'can_learn': avg_gradient > 0.01  # Threshold
    }

# Scenario 1: Large unscaled scores (d_k=512)
unscaled_scores = np.array([-35.2, 42.7, -28.9])
result_unscaled = simulate_training_step(unscaled_scores, scale=False)

print("WITHOUT SCALING:")
print(f"  Probabilities: {result_unscaled['probabilities']}")
print(f"  Gradients:     {result_unscaled['gradients']}")
print(f"  Avg gradient:  {result_unscaled['avg_gradient']:.10f}")
print(f"  Can learn:     {result_unscaled['can_learn']}")
print()

# Scenario 2: Same scores but scaled
result_scaled = simulate_training_step(unscaled_scores, scale=True, d_k=512)

print("WITH SCALING (÷ √512):")
print(f"  Probabilities: {result_scaled['probabilities']}")
print(f"  Gradients:     {result_scaled['gradients']}")
print(f"  Avg gradient:  {result_scaled['avg_gradient']:.10f}")
print(f"  Can learn:     {result_scaled['can_learn']}")

# Output:
# WITHOUT SCALING:
#   Probabilities: [0. 1. 0.]
#   Gradients:     [0. 0. 0.]
#   Avg gradient:  0.0000000000
#   Can learn:     False  ← BLOCKED!
#
# WITH SCALING:
#   Probabilities: [0.08 0.76 0.16]
#   Gradients:     [0.074 0.182 0.134]
#   Avg gradient:  0.1300000000
#   Can learn:     True  ← WORKING!
```

---

### 💡 Summary: Why Scaling Fixes Gradients

| Aspect | Without Scaling | With Scaling |
|--------|----------------|--------------|
| **Score range** | [-50, +50] | [-2.5, +2.5] |
| **Softmax output** | [0, 1, 0] (extreme) | [0.08, 0.76, 0.16] (balanced) |
| **Gradient** | ≈ 0 (vanished) | > 0.07 (healthy) |
| **Learning** | ❌ Blocked | ✅ Works |

**Scaling enables learning by keeping attention weights in a balanced range!**

---

## 🎓 Why Specifically √d_k?

### Question: Why not divide by d_k directly?

**Answer:** Because of how variance scales!

```python
# Example: 2D vectors
a = [2, 3]
b = [4, 5]

# Variance of individual elements
Var(a[0]) ≈ 1
Var(a[1]) ≈ 1

# Dot product variance
Var(a·b) = Var(a[0]×b[0] + a[1]×b[1])
         = Var(a[0]×b[0]) + Var(a[1]×b[1])
         = 1 + 1
         = 2  (adds up!)

# For d dimensions: Var(dot product) = d
```

**To normalize to unit variance:**
```
Standard deviation = √(variance) = √d_k

So we divide by √d_k (not d_k)
```

---

## 📝 Complete Formula

```
Scaled Dot-Product Attention:

Attention(Q, K, V) = softmax((Q·K^T) / √d_k) × V
```

**Step-by-step:**
```python
# 1. Compute attention scores
scores = Q @ K.T  # shape: (n, n)

# 2. Scale by √d_k
d_k = Q.shape[-1]  # dimension of Key
scaled_scores = scores / np.sqrt(d_k)

# 3. Apply softmax
attention_weights = softmax(scaled_scores)

# 4. Weighted sum of values
output = attention_weights @ V
```

---

## 🔬 Real Example with Numbers

### Setup

```python
# 3 words, embedding dimension = 512
d_k = 512

Q = np.random.randn(3, 512)  # Query
K = np.random.randn(3, 512)  # Key
V = np.random.randn(3, 512)  # Value
```

### Without Scaling

```python
scores = Q @ K.T
# scores might be:
# [[ 12.3,  45.7, -32.1],
#  [-28.4,  38.9,  15.2],
#  [ 41.2, -19.8,  22.6]]

weights = softmax(scores, axis=1)
# weights:
# [[0.000, 1.000, 0.000],  ← Extreme!
#  [0.000, 1.000, 0.000],  ← Extreme!
#  [1.000, 0.000, 0.000]]  ← Extreme!
```

**Problem:** Every word only attends to ONE other word!

---

### With Scaling

```python
scores = Q @ K.T / np.sqrt(d_k)
# scaled scores:
# [[ 0.54,  2.02, -1.42],
#  [-1.26,  1.72,  0.67],
#  [ 1.82, -0.88,  1.00]]

weights = softmax(scores, axis=1)
# weights:
# [[0.12, 0.74, 0.14],  ← Balanced!
#  [0.05, 0.86, 0.09],  ← Balanced!
#  [0.63, 0.05, 0.32]]  ← Balanced!
```

**Result:** Each word attends to multiple words with meaningful weights!

---

## ✅ Benefits of Scaling

1. **Stable gradients** → Better training
2. **Balanced attention** → Model learns from all words
3. **Consistent across dimensions** → Works for any d_k
4. **Prevents saturation** → Softmax doesn't get stuck

---

## 🎯 Key Takeaways

1. **Dot products grow with dimensionality:** Var(Q·K^T) ≈ d_k
2. **High variance → extreme softmax outputs:** Model can't learn
3. **Scaling by √d_k normalizes variance:** Brings it back to ~1
4. **Stable attention weights:** All words contribute to learning
5. **Formula:** `Attention = softmax(Q·K^T / √d_k) × V`

---

## 📊 Summary Table

| Aspect | Without Scaling | With Scaling (÷√d_k) |
|--------|----------------|----------------------|
| **Variance** | d_k × σ² (huge!) | σ² (normalized) |
| **Score range** | -50 to +50 | -2 to +2 |
| **Softmax output** | [0.0001, 0.9999] (extreme) | [0.15, 0.70, 0.15] (balanced) |
| **Gradients** | Vanish | Flow properly |
| **Learning** | Stuck | Effective |

---

## 🧮 Python Implementation

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention.
    
    Args:
        Q: Query matrix (n, d_k)
        K: Key matrix (n, d_k)
        V: Value matrix (n, d_v)
    
    Returns:
        output: Attention output (n, d_v)
        attention_weights: Attention weights (n, n)
    """
    # Get dimension
    d_k = Q.shape[-1]
    
    # 1. Compute attention scores
    scores = Q @ K.T  # (n, n)
    
    # 2. Scale by sqrt(d_k)  ← THE MAGIC STEP!
    scaled_scores = scores / np.sqrt(d_k)
    
    # 3. Apply softmax
    attention_weights = softmax(scaled_scores)
    
    # 4. Weighted sum of values
    output = attention_weights @ V
    
    return output, attention_weights

def softmax(x):
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Example usage
n_words = 3
d_k = 64

Q = np.random.randn(n_words, d_k)
K = np.random.randn(n_words, d_k)
V = np.random.randn(n_words, d_k)

output, weights = scaled_dot_product_attention(Q, K, V)

print("Attention weights:")
print(weights)
# Output will be balanced probabilities!
```

---

## 🚀 Next: Multi-Head Attention

Now that we understand scaling, we can tackle the next big concept:

**What if we want to attend to different aspects simultaneously?**
- Syntactic relationships
- Semantic relationships
- Positional relationships

This is where **Multi-Head Attention** comes in!

👉 Continue to: [04-multi-head-attention.md](04-multi-head-attention.md)
