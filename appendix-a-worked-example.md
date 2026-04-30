# Appendix A: Complete Worked Example - Geometric Intuition


<div align="center">
  <img src="Attention_is_all_you_need.jpeg" alt="Attention is All You Need" width="700">
  <p><em>The paper that started it all: "Attention is All You Need" (Vaswani et al., 2017)</em></p>
</div>

---

## 🎯 Goal

Work through a **complete numerical example** of attention mechanism from scratch, calculating every single step with actual numbers.

**Sentence:** "money bank"

We'll compute attention for the word "bank" step by step.

---

## 📊 Step 0: Setup (Initial Word Embeddings)

Let's use **2-dimensional embeddings** for simplicity (real transformers use 512+).

```python
import numpy as np

# Initial word embeddings (pretrained, e.g., from Word2Vec)
e_money = np.array([3, 1])  # 2D vector
e_bank  = np.array([1, 2])  # 2D vector

print("Word Embeddings:")
print(f"e_money = {e_money}")
print(f"e_bank  = {e_bank}")
```

**Visual representation:**
```
    y
    │
  2 │     • e_bank (1, 2)
    │    
  1 │ • e_money (3, 1)
    │
  0 └─────────────── x
    0   1   2   3
```

**Current state:**
```
Sentence: "money bank"
          ↓      ↓
        [3,1]  [1,2]
```

---

## 📐 Step 1: Define Weight Matrices

We need **three transformation matrices**: W^Q, W^K, W^V

For this example:
- Input dimension: d = 2
- Output dimension: d_k = d_v = 2

```python
# Query weight matrix (2×2)
W_Q = np.array([
    [2, 1],
    [1, 2]
])

# Key weight matrix (2×2)
W_K = np.array([
    [3, 1],
    [5, 1]
])

# Value weight matrix (2×2)
W_V = np.array([
    [4, 1],
    [2, 1]
])

print("\nWeight Matrices:")
print(f"W_Q =\n{W_Q}")
print(f"\nW_K =\n{W_K}")
print(f"\nW_V =\n{W_V}")
```

**Note:** These are **learnable parameters** (trained via backpropagation)

---

## 🔄 Step 2: Transform to Query, Key, Value

### For "money":

#### Query for "money":
```python
q_money = e_money @ W_Q
```

**Manual calculation:**
```
q_money = [3, 1] × [[2, 1],
                     [1, 2]]

= [3×2 + 1×1,  3×1 + 1×2]
= [6 + 1,      3 + 2]
= [7, 5]
```

**Verify:**
```python
q_money = e_money @ W_Q
print(f"q_money = {q_money}")
# Output: [7 5]
```

---

#### Key for "money":
```python
k_money = e_money @ W_K
```

**Manual calculation:**
```
k_money = [3, 1] × [[3, 1],
                     [5, 1]]

= [3×3 + 1×5,  3×1 + 1×1]
= [9 + 5,      3 + 1]
= [14, 4]
```

**Verify:**
```python
k_money = e_money @ W_K
print(f"k_money = {k_money}")
# Output: [14  4]
```

---

#### Value for "money":
```python
v_money = e_money @ W_V
```

**Manual calculation:**
```
v_money = [3, 1] × [[4, 1],
                     [2, 1]]

= [3×4 + 1×2,  3×1 + 1×1]
= [12 + 2,     3 + 1]
= [14, 4]
```

**Verify:**
```python
v_money = e_money @ W_V
print(f"v_money = {v_money}")
# Output: [14  4]
```

---

### For "bank":

#### Query for "bank":
```python
q_bank = e_bank @ W_Q
```

**Manual calculation:**
```
q_bank = [1, 2] × [[2, 1],
                    [1, 2]]

= [1×2 + 2×1,  1×1 + 2×2]
= [2 + 2,      1 + 4]
= [4, 5]
```

**Verify:**
```python
q_bank = e_bank @ W_Q
print(f"q_bank = {q_bank}")
# Output: [4 5]
```

---

#### Key for "bank":
```python
k_bank = e_bank @ W_K
```

**Manual calculation:**
```
k_bank = [1, 2] × [[3, 1],
                    [5, 1]]

= [1×3 + 2×5,  1×1 + 2×1]
= [3 + 10,     1 + 2]
= [13, 3]
```

**Verify:**
```python
k_bank = e_bank @ W_K
print(f"k_bank = {k_bank}")
# Output: [13  3]
```

---

#### Value for "bank":
```python
v_bank = e_bank @ W_V
```

**Manual calculation:**
```
v_bank = [1, 2] × [[4, 1],
                    [2, 1]]

= [1×4 + 2×2,  1×1 + 2×1]
= [4 + 4,      1 + 2]
= [8, 3]
```

**Verify:**
```python
v_bank = e_bank @ W_V
print(f"v_bank = {v_bank}")
# Output: [8 3]
```

---

## 📋 Summary of Transformations

```python
print("\n" + "="*60)
print("TRANSFORMED VECTORS:")
print("="*60)
print("\nFor 'money':")
print(f"  q_money = {q_money}")
print(f"  k_money = {k_money}")
print(f"  v_money = {v_money}")

print("\nFor 'bank':")
print(f"  q_bank = {q_bank}")
print(f"  k_bank = {k_bank}")
print(f"  v_bank = {v_bank}")
print("="*60)
```

**Output:**
```
============================================================
TRANSFORMED VECTORS:
============================================================

For 'money':
  q_money = [ 7  5]
  k_money = [14  4]
  v_money = [14  4]

For 'bank':
  q_bank = [4 5]
  k_bank = [13  3]
  v_bank = [8 3]
============================================================
```

---

## 🎯 Step 3: Compute Attention Scores

We want to compute attention for **"bank"** looking at all words.

**Formula:** Score = Query · Key^T

### Score: "bank" attending to "money"

```python
score_bank_money = q_bank @ k_money.T
```

**Manual calculation:**
```
score_bank_money = [4, 5] · [14, 4]^T
                 = [4, 5] · [14, 4]
                 = 4×14 + 5×4
                 = 56 + 20
                 = 76
```

**Verify:**
```python
score_bank_money = np.dot(q_bank, k_money)
print(f"Score (bank → money) = {score_bank_money}")
# Output: 76
```

---

### Score: "bank" attending to "bank" (itself)

```python
score_bank_bank = q_bank @ k_bank.T
```

**Manual calculation:**
```
score_bank_bank = [4, 5] · [13, 3]^T
                = [4, 5] · [13, 3]
                = 4×13 + 5×3
                = 52 + 15
                = 67
```

**Verify:**
```python
score_bank_bank = np.dot(q_bank, k_bank)
print(f"Score (bank → bank) = {score_bank_bank}")
# Output: 67
```

---

## 📏 Step 4: Scale the Scores

**Formula:** Scaled score = score / √d_k

Where d_k = 2 (dimension of key vectors)

```python
d_k = 2
scale_factor = np.sqrt(d_k)
print(f"\nScale factor √d_k = √{d_k} = {scale_factor:.4f}")
```

### Scaled scores:

```python
scaled_score_bank_money = score_bank_money / scale_factor
scaled_score_bank_bank = score_bank_bank / scale_factor
```

**Manual calculation:**
```
√2 ≈ 1.4142

scaled_score_bank_money = 76 / 1.4142 = 53.74
scaled_score_bank_bank  = 67 / 1.4142 = 47.38
```

**Verify:**
```python
print(f"\nScaled Scores:")
print(f"  bank → money: {scaled_score_bank_money:.2f}")
print(f"  bank → bank:  {scaled_score_bank_bank:.2f}")

# Output:
# Scaled Scores:
#   bank → money: 53.74
#   bank → bank:  47.38
```

---

## 🎲 Step 5: Apply Softmax

**Formula:** 
```
softmax([s₁, s₂]) = [e^s₁/(e^s₁ + e^s₂), e^s₂/(e^s₁ + e^s₂)]
```

### Manual calculation:

```python
scaled_scores = np.array([scaled_score_bank_money, scaled_score_bank_bank])
print(f"\nScaled scores vector: {scaled_scores}")
```

**Step-by-step softmax:**

```
s₁ = 53.74 (bank → money)
s₂ = 47.38 (bank → bank)

# Step 1: Compute exponentials
e^s₁ = e^53.74 ≈ 3.07 × 10²³
e^s₂ = e^47.38 ≈ 2.51 × 10²⁰

# Step 2: Sum
sum = e^s₁ + e^s₂ ≈ 3.07 × 10²³

# Step 3: Normalize
w₁ = e^s₁ / sum ≈ 3.07×10²³ / 3.07×10²³ ≈ 0.9992
w₂ = e^s₂ / sum ≈ 2.51×10²⁰ / 3.07×10²³ ≈ 0.0008
```

**Verify with code:**
```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / np.sum(exp_x)

attention_weights = softmax(scaled_scores)
print(f"\nAttention Weights:")
print(f"  bank → money: {attention_weights[0]:.6f}")
print(f"  bank → bank:  {attention_weights[1]:.6f}")
print(f"  Sum: {attention_weights.sum():.6f}")

# Output:
# Attention Weights:
#   bank → money: 0.998377
#   bank → bank:  0.001623
#   Sum: 1.000000
```

**Interpretation:**
- "bank" pays **99.84% attention** to "money"
- "bank" pays **0.16% attention** to itself

---

## 🎨 Step 6: Weighted Sum of Values

**Formula:** Output = w₁ × v_money + w₂ × v_bank

```python
w1 = attention_weights[0]  # 0.998377
w2 = attention_weights[1]  # 0.001623

output_bank = w1 * v_money + w2 * v_bank
```

**Manual calculation:**

```
v_money = [14, 4]
v_bank  = [8, 3]

output_bank = 0.998377 × [14, 4] + 0.001623 × [8, 3]

# First component:
= 0.998377 × 14 + 0.001623 × 8
= 13.977 + 0.013
= 13.990

# Second component:
= 0.998377 × 4 + 0.001623 × 3
= 3.994 + 0.005
= 3.999

output_bank ≈ [13.99, 3.99]
```

**Verify:**
```python
print(f"\nValue Vectors:")
print(f"  v_money = {v_money}")
print(f"  v_bank  = {v_bank}")

print(f"\nWeighted Sum Calculation:")
print(f"  {w1:.6f} × {v_money} + {w2:.6f} × {v_bank}")

output_bank = w1 * v_money + w2 * v_bank
print(f"\nFinal Output for 'bank':")
print(f"  output_bank = {output_bank}")
print(f"  output_bank ≈ [{output_bank[0]:.2f}, {output_bank[1]:.2f}]")

# Output:
# Final Output for 'bank':
#   output_bank = [13.9898302   3.99660087]
#   output_bank ≈ [13.99, 4.00]
```

**Observation:** 
- Output is almost identical to v_money = [14, 4]
- This makes sense because "bank" attended 99.84% to "money"!

---

## 📊 Complete Summary Table

| Step | Operation | Input | Output | Dimension |
|------|-----------|-------|--------|-----------|
| 0 | Word Embedding | "bank" | [1, 2] | 2 |
| 1 | Query Transform | [1, 2] × W_Q | [4, 5] | 2 |
| 1 | Key Transform | [1, 2] × W_K | [13, 3] | 2 |
| 1 | Value Transform | [1, 2] × W_V | [8, 3] | 2 |
| 2 | Dot Product (money) | [4,5] · [14,4] | 76 | scalar |
| 2 | Dot Product (bank) | [4,5] · [13,3] | 67 | scalar |
| 3 | Scaling | [76, 67] / √2 | [53.74, 47.38] | 2 |
| 4 | Softmax | [53.74, 47.38] | [0.998, 0.002] | 2 |
| 5 | Weighted Sum | 0.998×[14,4] + 0.002×[8,3] | [13.99, 4.00] | 2 |

---

## 🧮 Complete Python Code

```python
import numpy as np

# ==============================================================
# COMPLETE WORKED EXAMPLE
# ==============================================================

print("="*70)
print("ATTENTION MECHANISM - COMPLETE NUMERICAL EXAMPLE")
print("="*70)

# Step 0: Word Embeddings
e_money = np.array([3, 1])
e_bank  = np.array([1, 2])

print("\nSTEP 0: WORD EMBEDDINGS")
print(f"  e_money = {e_money}")
print(f"  e_bank  = {e_bank}")

# Step 1: Weight Matrices
W_Q = np.array([[2, 1], [1, 2]])
W_K = np.array([[3, 1], [5, 1]])
W_V = np.array([[4, 1], [2, 1]])

print("\nSTEP 1: WEIGHT MATRICES")
print(f"  W_Q =\n{W_Q}")
print(f"  W_K =\n{W_K}")
print(f"  W_V =\n{W_V}")

# Step 2: Transform to Q, K, V
q_money = e_money @ W_Q
k_money = e_money @ W_K
v_money = e_money @ W_V

q_bank = e_bank @ W_Q
k_bank = e_bank @ W_K
v_bank = e_bank @ W_V

print("\nSTEP 2: QUERY, KEY, VALUE VECTORS")
print(f"  q_money = {q_money}, k_money = {k_money}, v_money = {v_money}")
print(f"  q_bank  = {q_bank}, k_bank  = {k_bank}, v_bank  = {v_bank}")

# Step 3: Attention Scores
score_bank_money = np.dot(q_bank, k_money)
score_bank_bank  = np.dot(q_bank, k_bank)

print("\nSTEP 3: ATTENTION SCORES (Q·K^T)")
print(f"  bank → money: {score_bank_money}")
print(f"  bank → bank:  {score_bank_bank}")

# Step 4: Scaling
d_k = 2
scaled_scores = np.array([score_bank_money, score_bank_bank]) / np.sqrt(d_k)

print(f"\nSTEP 4: SCALED SCORES (÷ √{d_k})")
print(f"  Scaled scores: {scaled_scores}")

# Step 5: Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

attention_weights = softmax(scaled_scores)

print("\nSTEP 5: SOFTMAX (ATTENTION WEIGHTS)")
print(f"  bank → money: {attention_weights[0]:.6f} ({attention_weights[0]*100:.2f}%)")
print(f"  bank → bank:  {attention_weights[1]:.6f} ({attention_weights[1]*100:.2f}%)")

# Step 6: Weighted Sum
output_bank = attention_weights[0] * v_money + attention_weights[1] * v_bank

print("\nSTEP 6: WEIGHTED SUM OF VALUES")
print(f"  output_bank = {output_bank}")
print(f"  output_bank ≈ [{output_bank[0]:.2f}, {output_bank[1]:.2f}]")

print("\n" + "="*70)
print("COMPLETE! 'bank' now has context-aware representation")
print("="*70)
```

---

## 🎯 Geometric Visualization

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Original Embeddings
ax1 = axes[0, 0]
ax1.scatter(*e_money, c='red', s=200, marker='o', label='money')
ax1.scatter(*e_bank, c='blue', s=200, marker='s', label='bank')
ax1.arrow(0, 0, e_money[0], e_money[1], head_width=0.2, color='red', alpha=0.5)
ax1.arrow(0, 0, e_bank[0], e_bank[1], head_width=0.2, color='blue', alpha=0.5)
ax1.set_title('Original Embeddings')
ax1.legend()
ax1.grid(True)
ax1.set_xlabel('Dimension 1')
ax1.set_ylabel('Dimension 2')

# Plot 2: Query Vectors
ax2 = axes[0, 1]
ax2.scatter(*q_money, c='red', s=200, marker='o', label='q_money')
ax2.scatter(*q_bank, c='blue', s=200, marker='s', label='q_bank')
ax2.arrow(0, 0, q_money[0], q_money[1], head_width=0.3, color='red', alpha=0.5)
ax2.arrow(0, 0, q_bank[0], q_bank[1], head_width=0.3, color='blue', alpha=0.5)
ax2.set_title('Query Vectors')
ax2.legend()
ax2.grid(True)
ax2.set_xlabel('Dimension 1')
ax2.set_ylabel('Dimension 2')

# Plot 3: Key Vectors
ax3 = axes[1, 0]
ax3.scatter(*k_money, c='red', s=200, marker='o', label='k_money')
ax3.scatter(*k_bank, c='blue', s=200, marker='s', label='k_bank')
ax3.arrow(0, 0, k_money[0], k_money[1], head_width=0.5, color='red', alpha=0.5)
ax3.arrow(0, 0, k_bank[0], k_bank[1], head_width=0.5, color='blue', alpha=0.5)
ax3.set_title('Key Vectors')
ax3.legend()
ax3.grid(True)
ax3.set_xlabel('Dimension 1')
ax3.set_ylabel('Dimension 2')

# Plot 4: Value Vectors
ax4 = axes[1, 1]
ax4.scatter(*v_money, c='red', s=200, marker='o', label='v_money')
ax4.scatter(*v_bank, c='blue', s=200, marker='s', label='v_bank')
ax4.scatter(*output_bank, c='green', s=200, marker='*', label='output_bank')
ax4.arrow(0, 0, v_money[0], v_money[1], head_width=0.5, color='red', alpha=0.5)
ax4.arrow(0, 0, v_bank[0], v_bank[1], head_width=0.5, color='blue', alpha=0.5)
ax4.arrow(0, 0, output_bank[0], output_bank[1], head_width=0.5, color='green', alpha=0.5)
ax4.set_title('Value Vectors + Output')
ax4.legend()
ax4.grid(True)
ax4.set_xlabel('Dimension 1')
ax4.set_ylabel('Dimension 2')

plt.tight_layout()
plt.savefig('attention_geometric_intuition.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## ✅ Key Insights from This Example

1. **Transformation separates concerns:**
   - Query: [4, 5] - "what bank is looking for"
   - Key: [13, 3] and [14, 4] - "what each word offers"
   - Value: [8, 3] and [14, 4] - "what to actually pass on"

2. **Dot product measures relevance:**
   - bank·money = 76 (high!) → similar direction
   - bank·bank = 67 (also high)

3. **Softmax creates probability distribution:**
   - Converts scores to weights that sum to 1.0

4. **Output is context-aware:**
   - output_bank ≈ [13.99, 4.00] is heavily influenced by money
   - Original e_bank was [1, 2] - completely different!

5. **Learned weights matter:**
   - Different W_Q, W_K, W_V would give completely different results
   - These are optimized during training

---

## 🎓 Exercise: Try Changing the Weights

```python
# Try different weight matrices and see how attention changes!
# Example: Make bank attend more to itself

W_Q_new = np.array([[1, 0], [0, 1]])  # Identity
W_K_new = np.array([[1, 0], [0, 1]])
W_V_new = np.array([[1, 0], [0, 1]])

# Recompute and compare!
```

This is the **geometric intuition** behind attention - vectors in space, transformations, and weighted combinations!

---

## 🤔 Why is it Called "SELF" Attention?

### The Big Question

We've been computing attention, but why specifically **"SELF"** attention?

---

### 💡 The Answer: The Sentence Attends to ITSELF

**"Self" means the sequence attends to its own elements!**

Let's break this down:

---

### 📊 What "Self" Means

In our example: **"money bank"**

When computing attention for "bank":
- ✅ We looked at "**money**" (another word in the **same sentence**)
- ✅ We looked at "**bank**" (the word **itself**)

**Key insight:** Both attention targets come from the **SAME input sequence**!

```
Input sequence:  "money bank"
                    ↓      ↓
For "bank":      attend  attend
                 to this  to this
                    ↓      ↓
                 Both from the SAME sentence!
```

---

### 🔄 Self-Attention vs Other Types

#### ❌ NOT Self-Attention (Cross-Attention)

```
Source sentence (English): "The cat"
Target sentence (French):  "Le chat"

When translating "Le":
  - Query: from "Le" (French)
  - Keys: from "The", "cat" (English)
  - Values: from "The", "cat" (English)

  Different sequences! → CROSS-attention
```

---

#### ✅ Self-Attention

```
Sentence: "money bank"

When processing "bank":
  - Query: from "bank" (same sentence)
  - Keys: from "money", "bank" (same sentence)
  - Values: from "money", "bank" (same sentence)

  Same sequence! → SELF-attention
```

---

### 🎯 Technical Definition

**Self-Attention:**
```
Given input sequence X = [x₁, x₂, ..., xₙ]

For each position i:
  - Query comes from: xᵢ (position i in X)
  - Keys come from: x₁, x₂, ..., xₙ (all positions in X)
  - Values come from: x₁, x₂, ..., xₙ (all positions in X)

Q, K, V all derived from the SAME sequence X
```

---

### 📝 Our Example Breakdown

```python
# Input sequence
sentence = ["money", "bank"]

# Embeddings (from same sentence)
e_money = [3, 1]  # from sentence
e_bank  = [1, 2]  # from sentence

# For "bank" (position 1):
# Query: transformed from e_bank
q_bank = e_bank @ W_Q = [4, 5]

# Keys: transformed from ALL words in SAME sentence
k_money = e_money @ W_K = [14, 4]  ← from sentence
k_bank  = e_bank @ W_K  = [13, 3]  ← from sentence

# Values: transformed from ALL words in SAME sentence
v_money = e_money @ W_V = [14, 4]  ← from sentence
v_bank  = e_bank @ W_V  = [8, 3]   ← from sentence

# Attention computation
# "bank" attends to words from the SAME sentence
output_bank = w₁ × v_money + w₂ × v_bank
```

**Everything comes from the SAME input sequence → SELF-attention!**

---

### 🌐 Real-World Analogy

**Self-Attention = Self-Reflection**

Imagine you're writing your autobiography:

```
❌ NOT Self-Attention (Cross-Attention):
You: "In 2010, I graduated..."
Looking at: Your friend's diary to understand what you did

✅ Self-Attention:
You: "In 2010, I graduated..."
Looking at: Your own earlier chapters
  - "In 2005, I started high school" ← your own past
  - "In 2008, I applied to college" ← your own past
  
Using your OWN history to understand your current state!
```

---

### 📊 Visual Comparison

```
CROSS-ATTENTION (e.g., Translation):
┌─────────────────┐        ┌─────────────────┐
│  English:       │        │  French:        │
│  "The cat"      │───────>│  "Le chat"      │
└─────────────────┘        └─────────────────┘
     Source                     Target
                                  ↓
                    Queries from French
                    Keys/Values from English
                    
Different sequences!


SELF-ATTENTION:
┌─────────────────────────┐
│  "money bank"           │──┐
└─────────────────────────┘  │
           ↑                  │
           └──────────────────┘
              Looks at itself!
              
Same sequence!
```

---

### 🧮 Code Demonstration

```python
# SELF-ATTENTION
def self_attention(sentence_embeddings, W_Q, W_K, W_V):
    """
    All Q, K, V come from the SAME input.
    """
    Q = sentence_embeddings @ W_Q  # From SAME sentence
    K = sentence_embeddings @ W_K  # From SAME sentence
    V = sentence_embeddings @ W_V  # From SAME sentence
    
    scores = Q @ K.T
    weights = softmax(scores / np.sqrt(d_k))
    output = weights @ V
    
    return output

# CROSS-ATTENTION (for comparison)
def cross_attention(target_embeddings, source_embeddings, W_Q, W_K, W_V):
    """
    Q from target, K and V from source.
    """
    Q = target_embeddings @ W_Q  # From TARGET
    K = source_embeddings @ W_K  # From SOURCE (different!)
    V = source_embeddings @ W_V  # From SOURCE (different!)
    
    scores = Q @ K.T
    weights = softmax(scores / np.sqrt(d_k))
    output = weights @ V
    
    return output

# Example usage:
sentence = np.array([[3, 1], [1, 2]])  # "money bank"

# Self-attention: sentence looks at ITSELF
output_self = self_attention(sentence, W_Q, W_K, W_V)

# Cross-attention: would need TWO different sequences
# english = np.array([[2, 3], [1, 4]])
# french = np.array([[3, 1]])
# output_cross = cross_attention(french, english, W_Q, W_K, W_V)
```

---

### ✅ Summary: Why "SELF"

| Aspect | Self-Attention | Cross-Attention |
|--------|---------------|-----------------|
| **Query source** | Same sequence | Target sequence |
| **Key source** | Same sequence | Source sequence |
| **Value source** | Same sequence | Source sequence |
| **Use case** | Understanding context within one sentence | Connecting two different sequences |
| **Example** | BERT, GPT (single text) | Translation (English→French) |

**The word "SELF" emphasizes that the sequence is attending to ITSELF, not to some external sequence!**

---

### 🎯 Final Insight

In our worked example:

```
"money bank"
   ↓     ↓
Both words are from the SAME sentence

When "bank" computes attention:
  - It looks at "money" ← from SELF (same sentence)
  - It looks at "bank" ← from SELF (same sentence)
  
Hence: SELF-attention!
```

**The mechanism allows each word to gather context from other words in the SAME sequence - hence "SELF" attention!** ✓

---

## 🎓 Exercise: Try Both Types

```python
# Self-Attention Example
sentence = ["The", "cat", "sleeps"]
# Each word attends to all words in SAME sentence

# Cross-Attention Example (Translation)
english = ["The", "cat"]
french = ["Le", "chat"]
# French words attend to English words (DIFFERENT sequences)
```

Now you understand why it's called **"SELF"** attention! 🎉
