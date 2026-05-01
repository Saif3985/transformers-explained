# 02. Query, Key, and Value (QKV) Mechanism


<div align="center">
  <img src="Attention_is_all_you_need.jpeg" alt="Attention is All You Need" width="700">
  <p><em>The paper that started it all: "Attention is All You Need" (Vaswani et al., 2017)</em></p>
</div>

---

## 🤔 Quick Recap: The Problem with Basic Self-Attention

In the previous chapter, we used the **same embedding** for everything:

```
For word "bank":
- Similarity calculation: Used e_bank
- Weighting: Used e_bank  
- Final representation: Used e_bank
```

----

Python dictionary structure : {key:value} ----> d = {a:2,b :3}

query-->d[a] == 2 (value)

---

**This is like posting your entire autobiography on a  website instead of creating a proper profile!**

---

## 🎯 Real-World Analogy: The  Website Problem

### ❌ The Wrong Approach

Imagine a 40-year-old writer who wants to get married. He joins **marriage.com** and does this:

```
Profile Page:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"I was born in 1986 in Lahore. When I was 5, I moved to 
Karachi. In school, I loved mathematics. I won a science 
competition in 1998. I graduated from FAST in 2008. I got 
my first job as a software engineer in 2009. I wrote my 
first novel in 2015. My favorite food is biryani. I 
currently work as an ML engineer..."

[Entire 10,000-word autobiography posted here]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Searching/Matching:**
When someone searches for "Software Engineer, Age 40, Lahore", the website has to read his ENTIRE autobiography to find matches!

---

### ✅ The Right Approach

Instead, he should create **three separate things**:

#### 1. **Profile (What I'm Looking For)** = QUERY
```
- Age: 35-45
- Profession: Teacher, Engineer, Doctor
- Location: Lahore, Islamabad
- Interests: Reading, Travel
```

#### 2. **Search Index (What I Offer)** = KEY
```
Searchable tags:
- Age: 40
- Profession: ML Engineer, Writer
- Location: Lahore
- Interests: Books, AI, Coding
```

#### 3. **Detailed Information (What I Share)** = VALUE
```
Actual content to show:
- Full name: [Name]
- Career highlights: [Summary]
- Hobbies: Writing, ML research
- Bio: "Passionate about AI and literature..."
```

**How it works:**
1. His **Query** (what he's looking for) is matched against others' **Keys** (their searchable attributes)
2. If match score is high → retrieve their **Value** (actual profile to show)
3. Much more efficient than reading entire autobiographies!

---

## 💡 Dictionary Analogy: Understanding QKV

Think of a **Python dictionary**:

```python
author_data = {
    "jequahsanthi": {
        "profile": "Author, 33 years old",
        "search_tags": ["writing", "poetry", "age_30s"],
        "full_bio": "Born in Delhi, published 5 books..."
    }
}
```

**How you use it:**

1. **Query:** "Find me authors interested in poetry"
2. **Key:** Search through `search_tags` of all people
3. **Value:** When match found, return `full_bio`

**You DON'T search through the entire `full_bio` every time!**

---

## 🔑 The Core Idea: Three Different Representations

Instead of using the **same embedding** for everything, we create **three specialized vectors** from each word embedding:

```
Original embedding: e_bank

      ↓ Transform ↓

Query:  q_bank  (What am I looking for?)
Key:    k_bank  (What do I represent for searching?)
Value:  v_bank  (What information should I pass on?)
```

---

## ❓ Critical Questions

Before we dive into the math, let's address key questions:

### Q1: Why do we need THREE different vectors?

**Answer:** Different purposes require different representations:
- **Query:** Optimized for asking/searching (what patterns to look for)
- **Key:** Optimized for being searched/matched (what patterns I have)
- **Value:** Optimized for information transfer (what to actually communicate)

**Real example:**
- Google Search Query: "best restaurants near me"
- Website Key (metadata): "restaurant, location: Lahore, rating: 4.5"
- Website Value (content): Full restaurant details, menu, reviews

### Q2: How do we create these vectors?

**Answer:** Using **learnable weight matrices** (W^Q, W^K, W^V)

### Q3: Where do these weight matrices come from?

**Answer:** They are **learned during training** through backpropagation, just like any neural network weights!

---

## 🔢 The Mathematics: Creating Q, K, V

### Step 1: Start with Word Embeddings

Our sentence: **"money bank grows"**

```
Initial embeddings (dimension d = 3):
e_money = [0.7, 0.2, 0.1]
e_bank  = [0.25, 0.7, 0.05]
e_grows = [0.1, 0.2, 0.7]
```

We can stack them as a matrix:
```
X = [ e_money ]   =  [ 0.7   0.2   0.1  ]
    [ e_bank  ]      [ 0.25  0.7   0.05 ]
    [ e_grows ]      [ 0.1   0.2   0.7  ]

Shape: (3, 3) = (num_words, embedding_dim)
```

---

### Step 2: Define Weight Matrices (Learnable Parameters)

We create **three weight matrices** (these are learned during training):

```
W^Q = Query weight matrix (d × d_k)
W^K = Key weight matrix (d × d_k)
W^V = Value weight matrix (d × d_v)
```

**Example** (assuming d_k = d_v = d = 3 for simplicity):

```python
# Query transformation matrix
W_Q = [
    [0.5, 0.3, 0.2],
    [0.4, 0.6, 0.1],
    [0.2, 0.1, 0.7]
]

# Key transformation matrix
W_K = [
    [0.6, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.1, 0.3, 0.6]
]

# Value transformation matrix
W_V = [
    [0.4, 0.4, 0.3],
    [0.3, 0.5, 0.3],
    [0.3, 0.1, 0.6]
]
```

**These matrices are trainable parameters** (like weights in any neural network)!

---

### Step 3: Transform Embeddings to Q, K, V

Now we multiply each embedding with the weight matrices:

#### For Query:
```
Q = X × W^Q

q_money = e_money × W^Q
q_bank  = e_bank × W^Q
q_grows = e_grows × W^Q
```

**Detailed calculation for q_bank:**
```
q_bank = [0.25, 0.7, 0.05] × W^Q

= [0.25, 0.7, 0.05] × [[0.5, 0.3, 0.2],
                        [0.4, 0.6, 0.1],
                        [0.2, 0.1, 0.7]]

= [0.25×0.5 + 0.7×0.4 + 0.05×0.2,
   0.25×0.3 + 0.7×0.6 + 0.05×0.1,
   0.25×0.2 + 0.7×0.1 + 0.05×0.7]

= [0.415, 0.500, 0.155]
```

Similarly:
```
q_money = [0.51, 0.44, 0.28]
q_grows = [0.24, 0.20, 0.51]
```

#### For Key:
```
K = X × W^K

k_money = e_money × W^K = [0.57, 0.37, 0.39]
k_bank  = e_bank × W^K  = [0.375, 0.390, 0.185]
k_grows = e_grows × W^K = [0.22, 0.32, 0.47]
```

#### For Value:
```
V = X × W^V

v_money = e_money × W^V = [0.51, 0.51, 0.39]
v_bank  = e_bank × W^V  = [0.325, 0.415, 0.180]
v_grows = e_grows × W^V = [0.31, 0.27, 0.45]
```

---

### 📊 Visual Summary

```
Original Embedding (e_bank):
    [0.25, 0.7, 0.05]
           |
           | Transform with learned matrices
           |
    ┌──────┴──────┬──────────┐
    ↓             ↓          ↓
  × W^Q         × W^K      × W^V
    ↓             ↓          ↓
Query (q_bank)  Key (k_bank)  Value (v_bank)
[0.415, 0.500, 0.155]  [0.375, 0.390, 0.185]  [0.325, 0.415, 0.180]
```

**Key Insight:** 
- Same original embedding (e_bank)
- Three DIFFERENT representations
- Each optimized for its specific role

---

## 🎯 Complete Self-Attention with QKV

Now we can perform attention using these specialized vectors:

### Step 1: Compute Attention Scores (Q · K^T)

For "bank" attending to all words:

```
Score with "money": q_bank · k_money^T
  = [0.415, 0.500, 0.155] · [0.57, 0.37, 0.39]^T
  = 0.415×0.57 + 0.500×0.37 + 0.155×0.39
  = 0.481

Score with "bank": q_bank · k_bank^T = 0.389
Score with "grows": q_bank · k_grows^T = 0.267
```

Scores: `[S_21, S_22, S_23] = [0.481, 0.389, 0.267]`

---

### Step 2: Apply Softmax

```python
import numpy as np

scores = [0.481, 0.389, 0.267]
weights = np.exp(scores) / np.sum(np.exp(scores))

# weights = [0.38, 0.35, 0.27]
```

---

### Step 3: Weighted Sum of VALUES (Not Embeddings!)

**This is crucial:** We use **Value vectors**, not original embeddings!

```
bank^(new) = w_21 × v_money + w_22 × v_bank + w_23 × v_grows

= 0.38 × [0.51, 0.51, 0.39] + 
  0.35 × [0.325, 0.415, 0.180] +
  0.27 × [0.31, 0.27, 0.45]

= [0.391, 0.412, 0.292]
```

---

## 🔄 Complete Process Diagram

```
Input: "money bank grows"
         ↓
   Word Embeddings
   [e_money, e_bank, e_grows]
         ↓
    ┌────┴────┬────────┬────────┐
    ↓         ↓        ↓        ↓
  × W^Q     × W^K    × W^V
    ↓         ↓        ↓
   Query     Key     Value
  [q₁,q₂,q₃] [k₁,k₂,k₃] [v₁,v₂,v₃]
    ↓         ↓
    └─────────┴────→ Q · K^T → Scores
                         ↓
                     Softmax
                         ↓
                     Weights
                         ↓
                  Weights × V
                         ↓
                Context-Aware Output
```

---

## ✅ Why This is Better

### ❌ Old Approach (Same Embedding for Everything):
```
Similarity: e_bank · e_money
Weighting: Use same scores
Output: Weighted sum of e_money, e_bank, e_grows
```
**Problem:** Same vector doing multiple jobs → suboptimal

### ✅ New Approach (Specialized Q, K, V):
```
Matching: q_bank · k_money (specialized for comparison)
Weighting: Based on Q·K scores
Output: Weighted sum of v_money, v_bank, v_grows (specialized for output)
```
**Benefit:** Each vector optimized for its specific purpose!

---

## 🎓 Key Takeaways

1. **Three transformations:** Each embedding → Query, Key, Value
2. **Learned matrices:** W^Q, W^K, W^V are trainable (learned during training)
3. **Specialized roles:**
   - Q: What am I looking for?
   - K: What do I offer for matching?
   - V: What information do I share?
4. **Attention scores:** Computed using Q · K^T
5. **Output:** Weighted sum of V vectors (not original embeddings!)

---

## 🧮 Python Implementation

```python
import numpy as np

# Input embeddings
X = np.array([
    [0.7, 0.2, 0.1],    # money
    [0.25, 0.7, 0.05],  # bank
    [0.1, 0.2, 0.7]     # grows
])

# Weight matrices (randomly initialized, then learned)
W_Q = np.array([
    [0.5, 0.3, 0.2],
    [0.4, 0.6, 0.1],
    [0.2, 0.1, 0.7]
])

W_K = np.array([
    [0.6, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.1, 0.3, 0.6]
])

W_V = np.array([
    [0.4, 0.4, 0.3],
    [0.3, 0.5, 0.3],
    [0.3, 0.1, 0.6]
])

# Transform to Q, K, V
Q = X @ W_Q  # (3, 3) @ (3, 3) = (3, 3)
K = X @ W_K
V = X @ W_V

print("Query matrix Q:")
print(Q)
print("\nKey matrix K:")
print(K)
print("\nValue matrix V:")
print(V)

# Compute attention scores
scores = Q @ K.T  # (3, 3) @ (3, 3) = (3, 3)
print("\nAttention scores (Q·K^T):")
print(scores)

# Apply softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / exp_x.sum(axis=1, keepdims=True)

attention_weights = softmax(scores)
print("\nAttention weights (after softmax):")
print(attention_weights)

# Weighted sum of values
output = attention_weights @ V
print("\nOutput (context-aware embeddings):")
print(output)
```

Output:

Query matrix Q:
[[0.45  0.34  0.23 ]
 [0.415 0.5   0.155]
 [0.27  0.22  0.53 ]]

Key matrix K:
[[0.49  0.27  0.31 ]
 [0.365 0.415 0.245]
 [0.19  0.33  0.49 ]]

Value matrix V:
[[0.37  0.39  0.33 ]
 [0.325 0.455 0.315]
 [0.31  0.21  0.51 ]]

Attention scores (Q·K^T):
[[0.3836  0.3617  0.3104 ]
 [0.3864  0.39695 0.3198 ]
 [0.356   0.3197  0.3836 ]]

Attention weights (after softmax):
[[0.34390817 0.33645845 0.31963338]
 [0.33942313 0.343023   0.31755388]
 [0.33418714 0.32227369 0.34353917]]

Output (context-aware embeddings):
[[0.33568137 0.35433579 0.38248713]
 [0.33551073 0.3551368  0.38201435]
 [0.33488533 0.34911074 0.38700295]]

---

## 🚀 Next: Scaling and Multi-Head Attention

We've covered the basics of QKV, but there's more:
- Why do we need **scaling** (dividing by √d_k)?
- What is **Multi-Head Attention**?
- How do multiple attention heads help?

👉 Continue to: [03-scaled-dot-product-attention.md](03-scaled-dot-product-attention.md)

---

## 📝 Quick Reference

**Formula:**
```
Q = X × W^Q
K = X × W^K  
V = X × W^V

Attention(Q, K, V) = softmax(Q·K^T) × V
```

**Where:**
- X: Input embeddings (n × d)
- W^Q, W^K, W^V: Learnable weight matrices (d × d_k)
- Q, K, V: Transformed representations
- Output: Context-aware embeddings (n × d_v)

---


## 🔗 Interactive Resources

🔗 **[Transformer Explainer (Interactive Visualization)](https://poloclub.github.io/transformer-explainer/)**

🔗 **[Single vs Multi-Head Attention (ByHand.ai)](https://www.byhand.ai/p/library-models-attention-single-vs-multi-head)**

🔗 **[Self-Attention vs Cross-Attention (ByHand.ai)](https://www.byhand.ai/p/library-models-attention-self-vs-cross)**

---
