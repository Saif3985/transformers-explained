# 04. Multi-Head Attention

## 🤔 Opening Questions

Before we dive in, think about these:

1. **Why isn't one attention head enough?**
2. **What happens with ambiguous sentences like "The man saw the astronomer with a telescope"?**
3. **How can we capture multiple relationships simultaneously?**
4. **What does it mean to have 8 different "perspectives" on the same sentence?**
5. **How do we combine outputs from multiple heads efficiently?**

Let's answer all of these step by step!

---

## 🎯 The Problem: Single-Head Attention is Limited

### The Limitation We've Discovered

So far, we've built a powerful self-attention mechanism:
- ✅ Context-aware representations
- ✅ Captures word relationships
- ✅ Handles long-range dependencies

**But there's a critical limitation...**

---

### 💭 Question: What's Wrong with One Attention Head?

**Problem:** A single self-attention block produces **only ONE set of attention weights**.

This means:
- ❌ Only **one way** to look at relationships
- ❌ Only **one perspective** on the sentence
- ❌ Can't capture **multiple types** of relationships simultaneously

---

### 📊 Real Example: Ambiguous Sentence

**Sentence:** "The man saw the astronomer with a telescope"

**Question:** Who has the telescope?
- Option 1: The **man** is using a telescope to see the astronomer
- Option 2: The **astronomer** has a telescope

**With single-head attention:**
```
Attention weights for "telescope":
  - "man": 0.3
  - "saw": 0.1
  - "astronomer": 0.5
  - "with": 0.1

Only ONE interpretation captured! ❌
```

**What we actually need:**
```
Head 1 (syntactic): "telescope" relates to "saw" (instrument of seeing)
Head 2 (possessive): "telescope" belongs to "astronomer"
Head 3 (location): "telescope" is being held by "man"

Multiple perspectives! ✓
```

---

### 🔍 More Examples of Multi-Faceted Relationships

#### Example 1: "The bank by the river"

**Single head** might focus on:
- Only syntactic structure (bank ← river)

**Multiple heads** can capture:
- **Head 1:** Syntactic (bank modified by "river")
- **Head 2:** Semantic (financial bank vs river bank disambiguation)
- **Head 3:** Spatial (location relationship)

---

#### Example 2: Document Summarization

**Task:** Summarize a 1000-word article

**Single head problems:**
- Can only focus on one aspect (e.g., main topic)
- Misses supporting details, arguments, conclusions

**Multi-head solution:**
- **Head 1:** Main topic identification
- **Head 2:** Supporting evidence
- **Head 3:** Conclusions and implications
- **Head 4:** Temporal flow
- **Head 5:** Cause-effect relationships

---

## 💡 The Solution: Multi-Head Attention

### Core Idea

**Instead of one attention mechanism, use MULTIPLE attention mechanisms in parallel!**

```
Single-Head:
Input → [Self-Attention] → Output

Multi-Head:
        ┌─ [Self-Attention Head 1] ─┐
Input ──┼─ [Self-Attention Head 2] ─┼→ Combine → Output
        ├─ [Self-Attention Head 3] ─┤
        └─ [Self-Attention Head 4] ─┘
```

Each head can learn to focus on **different patterns**!

---

## 🏗️ Architecture: How Multi-Head Attention Works

### Step-by-Step Process

#### Step 1: Create Multiple Q, K, V Projections

Instead of one set of weight matrices (W^Q, W^K, W^V), we have **h sets** (where h = number of heads).

**For 8 heads:**
```
Head 1: W^Q₁, W^K₁, W^V₁
Head 2: W^Q₂, W^K₂, W^V₂
Head 3: W^Q₃, W^K₃, W^V₃
...
Head 8: W^Q₈, W^K₈, W^V₈
```

**Each head has its OWN learnable parameters!**

---

#### Step 2: Run Attention in Parallel

**For each head independently:**

```python
# Head 1
Q₁ = X @ W^Q₁
K₁ = X @ W^K₁
V₁ = X @ W^V₁
head₁ = Attention(Q₁, K₁, V₁)

# Head 2
Q₂ = X @ W^Q₂
K₂ = X @ W^K₂
V₂ = X @ W^V₂
head₂ = Attention(Q₂, K₂, V₂)

# ... and so on for all 8 heads
```

**All heads process in parallel (simultaneously)!**

---

#### Step 3: Concatenate Outputs

```python
# Combine all head outputs
multi_head_output = Concat(head₁, head₂, head₃, ..., head₈)
```

**Visual:**
```
head₁ = [0.2, 0.3, 0.5, ...]  (64 dims)
head₂ = [0.1, 0.8, 0.2, ...]  (64 dims)
head₃ = [0.5, 0.1, 0.4, ...]  (64 dims)
...
head₈ = [0.3, 0.6, 0.1, ...]  (64 dims)
         ↓ Concatenate ↓
output = [0.2,0.3,0.5,...,0.1,0.8,0.2,...,...,0.3,0.6,0.1,...]
         (8 × 64 = 512 dims)
```

---

#### Step 4: Final Linear Transformation

```python
# Combine perspectives with another learned matrix
final_output = multi_head_output @ W^O
```

Where W^O is a **learned output weight matrix**.

**Purpose:** Mix information from all heads into a unified representation.

---

## 📐 Dimension Management (The Clever Trick!)

### 💭 Question: Won't 8 heads be 8× more expensive?

**Answer: No! Here's the trick:**

---

### The Original Transformer Setup

**Model dimension:** d_model = 512

**For single-head attention:**
- Q, K, V dimensions: 512 each
- Computation: O(n² × 512)

**For 8-head attention (naive):**
- 8 heads × 512 dimensions = 8× computation ❌ TOO EXPENSIVE!

---

### The Smart Solution: Dimension Reduction

**Instead of 8 × 512, use 8 × 64!**

```
Total model dimension: 512
Number of heads: 8
Dimension per head: 512 / 8 = 64

Each head:
  - d_k = 64 (Query/Key dimension)
  - d_v = 64 (Value dimension)
  
Total across all heads: 8 × 64 = 512
```

**Same total computation as single head!** ✓

---

### Detailed Breakdown

```python
# Original dimensions
d_model = 512  # Model dimension
h = 8          # Number of heads
d_k = d_model // h = 64  # Dimension per head

# Single head (old):
W^Q: (512, 512)  → Parameters: 262,144
W^K: (512, 512)  → Parameters: 262,144
W^V: (512, 512)  → Parameters: 262,144
Total: 786,432 parameters

# Multi-head (8 heads):
For each head:
  W^Qᵢ: (512, 64)  → Parameters: 32,768
  W^Kᵢ: (512, 64)  → Parameters: 32,768
  W^Vᵢ: (512, 64)  → Parameters: 32,768
  
8 heads × 98,304 = 786,432 parameters
Plus W^O: (512, 512) = 262,144 parameters

Total: 1,048,576 parameters (comparable!)
```

**Computation cost remains similar while gaining 8× representational power!**

---

## 🧮 Complete Multi-Head Attention Algorithm

### Mathematical Formula

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₕ) W^O

where headᵢ = Attention(Q W^Qᵢ, K W^Kᵢ, V W^Vᵢ)

Attention(Q, K, V) = softmax((Q K^T) / √d_k) V
```

---

### Python Implementation

```python
import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model=512, num_heads=8):
        """
        d_model: Model dimension (e.g., 512)
        num_heads: Number of attention heads (e.g., 8)
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 512 / 8 = 64
        
        # Weight matrices for each head
        self.W_Q = [np.random.randn(d_model, self.d_k) for _ in range(num_heads)]
        self.W_K = [np.random.randn(d_model, self.d_k) for _ in range(num_heads)]
        self.W_V = [np.random.randn(d_model, self.d_k) for _ in range(num_heads)]
        
        # Output projection
        self.W_O = np.random.randn(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V):
        """Single attention head computation."""
        d_k = Q.shape[-1]
        
        # Compute attention scores
        scores = Q @ K.T / np.sqrt(d_k)
        
        # Apply softmax
        attention_weights = self.softmax(scores)
        
        # Weighted sum of values
        output = attention_weights @ V
        
        return output, attention_weights
    
    def softmax(self, x):
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, X):
        """
        X: Input sequence (n_words, d_model)
        Returns: Output (n_words, d_model)
        """
        n_words = X.shape[0]
        
        # Store outputs from all heads
        head_outputs = []
        
        # Process each head
        for i in range(self.num_heads):
            # Project to Q, K, V for this head
            Q = X @ self.W_Q[i]  # (n_words, d_k)
            K = X @ self.W_K[i]  # (n_words, d_k)
            V = X @ self.W_V[i]  # (n_words, d_k)
            
            # Compute attention for this head
            head_output, _ = self.scaled_dot_product_attention(Q, K, V)
            head_outputs.append(head_output)
        
        # Concatenate all heads
        multi_head = np.concatenate(head_outputs, axis=-1)  # (n_words, d_model)
        
        # Final linear transformation
        output = multi_head @ self.W_O
        
        return output

# Example usage
d_model = 512
num_heads = 8
seq_length = 10  # Number of words

# Create random input
X = np.random.randn(seq_length, d_model)

# Initialize multi-head attention
mha = MultiHeadAttention(d_model, num_heads)

# Forward pass
output = mha.forward(X)

print(f"Input shape: {X.shape}")
print(f"Output shape: {output.shape}")
print(f"Number of heads: {num_heads}")
print(f"Dimension per head: {d_model // num_heads}")
```

---

## 🎨 Worked Example: 2 Heads

Let's work through a simple example with **2 heads** and **2 words**.

### Setup

```python
# Sentence: "money bank"
# Embeddings (d_model = 4)
e_money = np.array([1, 0, 1, 0])
e_bank  = np.array([0, 1, 0, 1])

X = np.array([e_money, e_bank])  # (2, 4)

# Parameters
d_model = 4
num_heads = 2
d_k = d_model // num_heads = 2
```

---

### Head 1 Computation

```python
# Weight matrices for Head 1 (4 × 2)
W_Q1 = np.array([[1, 0],
                  [0, 1],
                  [1, 1],
                  [0, 0]])

W_K1 = np.array([[0, 1],
                  [1, 0],
                  [0, 1],
                  [1, 0]])

W_V1 = np.array([[1, 1],
                  [1, 0],
                  [0, 1],
                  [1, 1]])

# Project to Q, K, V
Q1 = X @ W_Q1  # (2, 2)
K1 = X @ W_K1
V1 = X @ W_V1

# Compute attention
scores1 = Q1 @ K1.T / np.sqrt(2)
weights1 = softmax(scores1)
head1_output = weights1 @ V1  # (2, 2)
```

---

### Head 2 Computation

```python
# Different weight matrices for Head 2
W_Q2 = np.array([[0, 1],
                  [1, 0],
                  [0, 0],
                  [1, 1]])

W_K2 = np.array([[1, 1],
                  [0, 1],
                  [1, 0],
                  [0, 0]])

W_V2 = np.array([[0, 1],
                  [1, 1],
                  [1, 0],
                  [0, 1]])

# Project to Q, K, V
Q2 = X @ W_Q2
K2 = X @ W_K2
V2 = X @ W_V2

# Compute attention
scores2 = Q2 @ K2.T / np.sqrt(2)
weights2 = softmax(scores2)
head2_output = weights2 @ V2  # (2, 2)
```

---

### Concatenation and Output

```python
# Concatenate both heads
multi_head = np.concatenate([head1_output, head2_output], axis=-1)
# Shape: (2, 4)  [2 words, 2 dims from head1 + 2 dims from head2]

# Output projection (4 × 4)
W_O = np.array([[1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [1, 1, 0, 0],
                 [0, 0, 1, 1]])

# Final output
output = multi_head @ W_O  # (2, 4)

print("Head 1 output shape:", head1_output.shape)  # (2, 2)
print("Head 2 output shape:", head2_output.shape)  # (2, 2)
print("Concatenated shape:", multi_head.shape)     # (2, 4)
print("Final output shape:", output.shape)         # (2, 4)
```

**Result:** Each word now has a 4-dimensional output combining insights from both heads!

---

## 🎯 What Do Different Heads Learn?

### Empirical Observations from Research

Different heads in transformers tend to specialize:

#### Head 1: Syntactic Relations
```
Sentence: "The cat sat on the mat"

Attention pattern:
  - "sat" → "cat" (subject)
  - "sat" → "on" (preposition)
  - "on" → "mat" (object)
```

**Captures grammatical structure!**

---

#### Head 2: Semantic Similarity
```
Sentence: "The bank by the river has money"

Attention pattern:
  - "bank" → "money" (semantic link)
  - "river" → ignored (different meaning)
```

**Disambiguates word meanings!**

---

#### Head 3: Positional/Sequential
```
Sentence: "First, do this. Then, do that"

Attention pattern:
  - "Then" → "First" (temporal order)
  - "that" → "this" (reference resolution)
```

**Tracks sequence and order!**

---

#### Head 4: Long-Range Dependencies
```
Sentence: "The keys, which I thought I lost yesterday, were in my pocket"

Attention pattern:
  - "were" → "keys" (across 9 words!)
```

**Connects distant related words!**

---

## 📊 Visualization: 8 Heads in Action

**Sentence:** "The man saw the astronomer with a telescope"

```
HEAD 1 (Syntactic):
  man → saw (subject-verb)
  saw → astronomer (verb-object)

HEAD 2 (Instrument):
  saw → telescope (instrument of seeing)
  with → telescope

HEAD 3 (Possession):
  astronomer → telescope (possessive)
  
HEAD 4 (Articles):
  The → man
  the → astronomer
  a → telescope

HEAD 5 (Verb Relations):
  saw → with (verb-preposition)

HEAD 6 (Noun Phrases):
  astronomer → with telescope (phrase)

HEAD 7 (Subject Focus):
  All words → man (main subject)

HEAD 8 (Action Focus):
  All words → saw (main action)
```

**8 different perspectives on the SAME sentence!**

---

## ✅ Benefits of Multi-Head Attention

### 1. **Richer Representations**
- Captures multiple aspects simultaneously
- No need to choose one interpretation

### 2. **Better Disambiguation**
- Can handle polysemy ("bank" = financial vs river)
- Resolves structural ambiguity

### 3. **Robust Learning**
- If one head fails, others compensate
- Redundancy improves performance

### 4. **Computational Efficiency**
- Parallel processing (all heads at once)
- Similar cost to single head (via dimension reduction)

### 5. **Interpretability**
- Can visualize what each head learns
- Helps understand model decisions

---

## 🔢 The Numbers: Original Transformer (2017)

**"Attention is All You Need" paper configuration:**

```
Model dimension (d_model): 512
Number of heads (h): 8
Dimension per head (d_k): 512 / 8 = 64
Number of layers: 6 (encoder) + 6 (decoder)

Parameters per layer:
  - Multi-head attention: ~1M parameters
  - Feed-forward network: ~2M parameters
  
Total model: ~65M parameters (Base model)
```

**Why 8 heads?**
- Sweet spot found through experimentation
- Enough diversity, not too complex
- Balances performance vs computational cost

---

## 🎓 Key Takeaways

1. **Single-head limitation:** Only one perspective on relationships

2. **Multi-head solution:** h parallel attention mechanisms

3. **Dimension trick:** Reduce per-head dimension to maintain efficiency
   ```
   8 heads × 64 dims = 512 total (same as 1 head × 512)
   ```

4. **Process:**
   ```
   Input → [h parallel attention heads] → Concatenate → Linear → Output
   ```

5. **Each head learns different patterns:** syntax, semantics, position, etc.

6. **Original Transformer:** 8 heads with d_k = 64

---

## 🚀 Next: Positional Encoding

We have attention working beautifully, but there's still a problem:

**Our attention mechanism has no notion of word ORDER!**

```
"dog bites man" vs "man bites dog"
```

Both would get the same attention patterns! We need **Positional Encoding**.

👉 Continue to: [05-positional-encoding.md](05-positional-encoding.md)

---

## 📝 Complete Code Summary

```python
# Multi-Head Attention (simplified)
def multi_head_attention(X, W_Q_list, W_K_list, W_V_list, W_O, num_heads):
    """
    X: (n, d_model) - input sequence
    W_Q_list, W_K_list, W_V_list: h matrices of shape (d_model, d_k)
    W_O: (d_model, d_model) - output projection
    num_heads: number of attention heads
    """
    d_k = X.shape[-1] // num_heads
    heads = []
    
    # Compute each head
    for i in range(num_heads):
        Q = X @ W_Q_list[i]
        K = X @ W_K_list[i]
        V = X @ W_V_list[i]
        
        # Scaled dot-product attention
        scores = (Q @ K.T) / np.sqrt(d_k)
        weights = softmax(scores)
        head_output = weights @ V
        
        heads.append(head_output)
    
    # Concatenate and project
    multi_head = np.concatenate(heads, axis=-1)
    output = multi_head @ W_O
    
    return output
```

**That's Multi-Head Attention in full detail!** 🎉

---

## 📓 Interactive Notebooks

Want to see this in action with visualizations and interactive code?

### Notebook 1: Multi-Head Attention Implementation

🔗 **[Multi-Head Attention Interactive Notebook](https://colab.research.google.com/drive/1rPk3ohrmVclqhH7uQ7qys4oznDdAhpzF)**

This notebook includes:
- ✅ Complete multi-head attention implementation
- ✅ Visualization of attention patterns for each head
- ✅ Step-by-step computation examples
- ✅ Interactive demos with real sentences
- ✅ Dimension management examples
- ✅ Comparison between single-head vs multi-head

### Notebook 2: Transformer Architecture Deep Dive

🔗 **[Complete Transformer Tutorial](https://colab.research.google.com/drive/1hXIQ77A4TYS4y3UthWF-Ci7V7vVUoxmQ#scrollTo=YLAhBxDSScmV)**

This notebook covers:
- ✅ Full transformer architecture from scratch
- ✅ Multi-head attention with detailed examples
- ✅ Positional encoding
- ✅ Encoder-decoder structure
- ✅ Training and inference examples
- ✅ Real-world applications

**Run them yourself and experiment with different configurations!**
