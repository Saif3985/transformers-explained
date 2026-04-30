# 01. Self-Attention Mechanism

## 🎯 What is Self-Attention?

Self-attention is a mechanism that allows each word in a sentence to **look at all other words** and decide how much attention to pay to each one when creating its own representation.

**Key Idea:** Instead of treating all words equally, self-attention computes **dynamic weights** that determine how much each word should contribute to understanding another word.

---

## 💡 Intuition: A Simple Example

Let's understand this with a sentence:

**Sentence:** "money bank grows"

We have 3 words, each with its own embedding vector:
- **money** → embedding vector
- **bank** → embedding vector  
- **grows** → embedding vector

---

## 🔄 The Self-Attention Process

### Step 1: Initial Embeddings

Each word starts with a static embedding (from Word2Vec, GloVe, etc.):

```
S1: money = [vector of n dimensions]
S1: bank  = [vector of n dimensions]
S1: grows = [vector of n dimensions]
```

**Problem:** These are **context-independent** (same vector regardless of surrounding words)

---

### Step 2: Computing Similarity Scores

For each word, we compute how **similar** it is to every other word (including itself).

**Example: Computing new representation for "bank"**

```
For word "bank", we compute:
- How similar is "bank" to "money"?
- How similar is "bank" to "bank" (itself)?
- How similar is "bank" to "grows"?
```

**Similarity Computation:**
```
S₂₁ = similarity(e_bank, e_money)
S₂₂ = similarity(e_bank, e_bank)
S₂₃ = similarity(e_bank, e_grows)
```

Where similarity = **dot product** of embeddings:
```
S₂₁ = e_bank · e_money^T
```

**Visual representation:**
```
           money    bank    grows
bank:     [S₂₁]    [S₂₂]    [S₂₃]
```

These scores tell us how relevant each word is to "bank".

---

### Step 3: Normalize with Softmax

Raw similarity scores can be any value. We normalize them to create **attention weights** (probabilities):

```
Softmax([S₂₁, S₂₂, S₂₃]) → [w₂₁, w₂₂, w₂₃]

where:
w₂₁ = e^S₂₁ / (e^S₂₁ + e^S₂₂ + e^S₂₃)
w₂₂ = e^S₂₂ / (e^S₂₁ + e^S₂₂ + e^S₂₃)
w₂₃ = e^S₂₃ / (e^S₂₁ + e^S₂₂ + e^S₂₃)
```

**Key Properties:**
- All weights sum to 1: `w₂₁ + w₂₂ + w₂₃ = 1.0`
- Each weight is between 0 and 1
- Higher similarity → higher weight

**Example values:**
```
w₂₁ = 0.25  (25% attention to "money")
w₂₂ = 0.70  (70% attention to "bank" itself)
w₂₃ = 0.05  (5% attention to "grows")
```

---

### Step 4: Weighted Sum (Context-Aware Representation)

Now we create a **new representation** for "bank" by taking a weighted combination of all word embeddings:

```
e_bank^(new) = w₂₁ × e_money + w₂₂ × e_bank + w₂₃ × e_grows
```

Substituting our example weights:
```
e_bank^(new) = 0.25 × e_money + 0.70 × e_bank + 0.05 × e_grows
```

**What does this mean?**
- The new "bank" embedding is **70% from "bank" itself**
- But also **25% influenced by "money"** (high relevance!)
- And **5% from "grows"** (low relevance)

This creates a **context-aware** representation!

---

## 🔄 Complete Process for All Words

We repeat this for **every word** in the sentence:

### For "money":
```
1. Compute similarities: [S₁₁, S₁₂, S₁₃]
2. Apply softmax: [w₁₁, w₁₂, w₁₃]
3. Weighted sum: e_money^(new) = w₁₁×e_money + w₁₂×e_bank + w₁₃×e_grows
```

### For "bank":
```
1. Compute similarities: [S₂₁, S₂₂, S₂₃]
2. Apply softmax: [w₂₁, w₂₂, w₂₃]
3. Weighted sum: e_bank^(new) = w₂₁×e_money + w₂₂×e_bank + w₂₃×e_grows
```

### For "grows":
```
1. Compute similarities: [S₃₁, S₃₂, S₃₃]
2. Apply softmax: [w₃₁, w₃₂, w₃₃]
3. Weighted sum: e_grows^(new) = w₃₁×e_money + w₃₂×e_bank + w₃₃×e_grows
```

---

## 📊 Visual Summary

```
Original Embeddings (S1):
┌─────────┐
│  money  │ ──────┐
└─────────┘       │
┌─────────┐       │    Similarity    ┌─────────┐
│  bank   │ ──────┼───── Scores ────>│ Softmax │
└─────────┘       │                  └─────────┘
┌─────────┐       │                       │
│  grows  │ ──────┘                       │
└─────────┘                               │
                                          ▼
                                     Weights (w)
                                          │
                                          ▼
                              ┌────────────────────┐
                              │   Weighted Sum     │
                              │  (New Embeddings)  │
                              └────────────────────┘
                                          │
                                          ▼
                              Context-Aware Embeddings (S2)
                              ┌─────────────┐
                              │ money (new) │
                              │ bank (new)  │
                              │ grows (new) │
                              └─────────────┘
```

---

## ✅ What Self-Attention Achieves

### 1. **Context-Aware Embeddings**
- "bank" in "money bank grows" gets influenced by "money"
- "bank" in "river bank flows" would get influenced by "river"
- ✅ Solves the **polysemy problem**

### 2. **Different Attention for Each Word**
```
For "money":
  - High attention to "bank" (related concept)
  - Low attention to "grows"

For "bank":
  - High attention to "money" (related concept)
  - Medium attention to itself
  
For "grows":
  - High attention to "money" (subject growing)
  - Low attention to "bank"
```

### 3. **Importance Weighting**
- Important words get higher weights
- Filler words get lower weights
- ✅ Solves the **all words equal** problem

### 4. **Long-Range Dependencies**
- Every word can attend to **any other word**
- Distance doesn't matter
- ✅ Solves the **long-range dependency** problem

---

## 🔑 Key Takeaways

1. **Self-attention computes similarity between all word pairs**
2. **Softmax converts similarities to normalized weights**
3. **Weighted sum creates context-aware representations**
4. **Process repeats for every word in the sentence**
5. **Result: Dynamic, context-dependent embeddings**

---

## ⚠️ Limitations of This Approach

This is a **simplified, first-principles explanation**. It has limitations:

1. ❌ **General context, not task-specific**
   - Weights are based purely on embedding similarity
   - Not optimized for specific tasks (translation, summarization, etc.)

2. ❌ **No learned parameters**
   - Just using dot product of existing embeddings
   - Cannot adapt or improve with training

3. ❌ **Single representation space**
   - All words operate in the same embedding space
   - Cannot capture multiple types of relationships

---

## 🚀 Next: Query-Key-Value (QKV) Mechanism

To make self-attention **learnable and task-specific**, we introduce:
- **Query (Q):** What am I looking for?
- **Key (K):** What information do I have?
- **Value (V):** What information should I pass on?

This transforms basic self-attention into the powerful mechanism used in Transformers.

👉 Continue to: [02-query-key-value.md](02-query-key-value.md)

---

## 📝 Summary in Code (Simplified)

```python
import numpy as np

# Step 1: Word embeddings
embeddings = {
    'money': np.array([0.7, 0.2, 0.1]),
    'bank':  np.array([0.25, 0.7, 0.05]),
    'grows': np.array([0.1, 0.2, 0.7])
}

# Step 2: Compute similarities (dot products)
def compute_similarity(word1, word2):
    return np.dot(embeddings[word1], embeddings[word2])

# For "bank":
S_21 = compute_similarity('bank', 'money')
S_22 = compute_similarity('bank', 'bank')
S_23 = compute_similarity('bank', 'grows')

# Step 3: Softmax normalization
def softmax(scores):
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)

weights = softmax([S_21, S_22, S_23])
# weights = [w_21, w_22, w_23]

# Step 4: Weighted sum
bank_new = (weights[0] * embeddings['money'] + 
            weights[1] * embeddings['bank'] + 
            weights[2] * embeddings['grows'])

print(f"New bank embedding: {bank_new}")
```

This is the **essence of self-attention**!
