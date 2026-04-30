# 08. Masked Multi-Head Attention

<div align="center">
  <img src="Attention_is_all_you_need.jpeg" alt="Attention is All You Need" width="700">
  <p><em>The paper that started it all: "Attention is All You Need" (Vaswani et al., 2017)</em></p>
</div>

---

## 🎯 The Core Challenge

**Key Insight:** Transformer decoder is:
- **Autoregressive** during inference (generation)
- **Non-autoregressive** during training (parallel processing)

**How do we train it efficiently while maintaining sequential generation capability?**

---

## 🔄 What is Autoregressive Generation?

### Definition

**Autoregressive models generate data sequentially, where each new output depends on all previously generated outputs.**

```
Generate word 1 → Use word 1 to generate word 2 
              → Use words 1,2 to generate word 3
              → Use words 1,2,3 to generate word 4
              ...
```

---

### Example: Translation

**Task:** Translate "How are you" (English) → Hindi

**Autoregressive Generation at Inference:**

```
Step 1: Generate آپ
  Input: <START>
  Output: آپ (you)

Step 2: Generate کیسے
  Input: <START>, آپ
  Output: کیسے (how)

Step 3: Generate ہیں
  Input: <START>, آپ, کیسے
  Output: ہیں (are)

Step 4: Generate <END>
  Input: <START>, آپ, کیسے, ہیں
  Output: <END>

Final: "آپ کیسے ہیں"
```

**Each word depends on ALL previous words.**

---

### Visual: Autoregressive Flow

```
Encoder Output (from "How are you")
        ↓
        ├──────────────────┐
        ↓                  ↓
┌─────────────┐    ┌─────────────┐
│  <START>    │───→│  Decoder    │
└─────────────┘    └─────────────┘
                          ↓
                      آپ (you)
                          ↓
┌─────────────────┐  ┌─────────────┐
│ <START>, آپ  │─→│  Decoder    │
└─────────────────┘  └─────────────┘
                          ↓
                    کیسے (how)
                          ↓
┌──────────────────────┐ ┌─────────────┐
│<START>, آپ, کیسے│→│  Decoder    │
└──────────────────────┘ └─────────────┘
                              ↓
                          ہیں (are)
```

**Sequential, one word at a time.**

---

## 🐌 The Problem: Training is SLOW

### If We Train Autoregressively

**Example sentence:** "آپ کیسے ہیں" (4 tokens including <END>)

**Training requires 4 separate forward passes:**

```
Pass 1: Input <START>           → Predict آپ
Pass 2: Input <START>, آپ    → Predict کیسے  
Pass 3: Input <START>, آپ, کیسے → Predict ہیں
Pass 4: Input <START>, آپ, کیسے, ہیں → Predict <END>
```

**For a dataset with:**
- 300 words average per sentence
- 10,000 sentences

**Total forward passes:** 300 × 10,000 = **3 million passes!**

**Training time:** Weeks or months ❌

---

## 💡 Naive Solution: Parallel Training

### What If We Process All Positions at Once?

```
Input ALL positions simultaneously:
  Position 1: <START>
  Position 2: آپ
  Position 3: کیسے
  Position 4: ہیں

Run decoder ONCE → Get all predictions

Training time: Hours instead of weeks! ✓
```

**Sounds perfect! But there's a fatal flaw...**

---

## 🚨 The Data Leakage Problem

### What Happens Without Masking

**During training, decoder sees the ENTIRE target sentence:**

```
Target: "آپ کیسے ہیں"

When predicting کیسے:
  Should only see: <START>, آپ
  Actually sees: <START>, آپ, کیسے, ہیں ❌
  
Model cheats! It knows the answer already!
```

**Self-Attention allows each word to attend to ALL words:**

```
Self-Attention Scores (WITHOUT masking):

         آپ    کیسے    ہیں
آپ   [  0.3    0.5    0.2  ]  ← آپ attends to future!
کیسے [  0.2    0.6    0.2  ]  ← کیسے sees ہیں!
ہیں  [  0.3    0.4    0.3  ]
```

**Result:**
- Model memorizes instead of learning patterns
- Fails completely during inference (no future tokens available)

---

### Example: The Cheating Problem

**Training:**
```python
# Model sees entire target
Input: [آپ, کیسے, ہیں]

Predicting کیسے:
  Attention weights:
    آپ:  0.15
    کیسے: 0.70  ← Cheating! Looking at itself
    ہیں:  0.15  ← Cheating! Looking ahead
  
  Prediction: کیسے ✓ (easy when you cheat!)
```

**Inference:**
```python
# Model only has previous tokens
Input: [آپ]

Predicting next word:
  Attention weights:
    آپ: 1.0
    (no کیسے or ہیں available)
  
  Prediction: ??? ❌ (model is lost!)
```

**Training ≠ Inference = Disaster!**

---

## ✅ The Solution: Masked Self-Attention

### Core Idea

**During training, prevent each position from attending to future positions.**

```
Block future tokens by setting their attention scores to -∞
  ↓
After softmax, -∞ becomes 0
  ↓
Future positions contribute nothing
```

---

### The Mask Matrix

**For sequence length 4:**

```
Mask (1 = allowed, 0 = blocked):

         pos1  pos2  pos3  pos4
pos1  [   1     0     0     0  ]  ← pos1 only sees itself
pos2  [   1     1     0     0  ]  ← pos2 sees pos1,2
pos3  [   1     1     1     0  ]  ← pos3 sees pos1,2,3
pos4  [   1     1     1     1  ]  ← pos4 sees all

Lower triangular matrix!
```

**Positions:**
- Position 1: <START>
- Position 2: آپ
- Position 3: کیسے
- Position 4: ہیں

---

### How Masking Works

**Step 1: Compute Attention Scores (Q·K^T)**

```
Scores (before masking):

         آپ    کیسے    ہیں
آپ   [  12     15     10  ]
کیسے [  8      20     14  ]
ہیں  [  11     16     18  ]
```

---

**Step 2: Apply Mask**

```
Mask:
         آپ    کیسے    ہیں
آپ   [  1      0      0   ]
کیسے [  1      1      0   ]
ہیں  [  1      1      1   ]

Operation: scores + (mask - 1) × ∞

Result (masked scores):
         آپ    کیسے    ہیں
آپ   [  12    -∞     -∞   ]
کیسے [  8      20    -∞   ]
ہیں  [  11     16     18  ]
```

**Future positions → -∞**

---

**Step 3: Softmax**

```
Softmax of masked scores:

         آپ    کیسے    ہیں
آپ   [  1.0    0.0    0.0 ]  ← Only attends to itself
کیسے [  0.15   0.85   0.0 ]  ← Attends to آپ, کیسے
ہیں  [  0.10   0.20   0.70]  ← Attends to all previous

Future weights = 0!
```

---

**Step 4: Weighted Sum**

```
For کیسے:
  output = 0.15×V_آپ + 0.85×V_کیسے + 0.0×V_ہیں
  
  No information leakage from ہیں! ✓
```

---

### Complete Example: Translation Training

**Task:** English "How are you" → Hindi "آپ کیسے ہیں"

**Training Data:**
```
Source (Encoder input): "How are you"
Target (Decoder input): <START> آپ کیسے ہیں
Target (Labels):        آپ کیسے ہیں <END>
```

**Decoder processes all positions in parallel:**

```
┌────────────────────────────────────────┐
│     Masked Multi-Head Attention        │
│                                        │
│  Position 1 (<START>):                │
│    Can attend to: <START> only        │
│    Predicts: آپ                     │
│                                        │
│  Position 2 (آپ):                   │
│    Can attend to: <START>, آپ       │
│    Predicts: کیسے                    │
│                                        │
│  Position 3 (کیسے):                  │
│    Can attend to: <START>, آپ, کیسے │
│    Predicts: ہیں                     │
│                                        │
│  Position 4 (ہیں):                   │
│    Can attend to: All previous        │
│    Predicts: <END>                    │
└────────────────────────────────────────┘

All processed in ONE forward pass! ✓
No data leakage! ✓
```

---

### Attention Patterns Visualization

**Without Masking (WRONG):**
```
Self-Attention Matrix:

              <START>  آپ   کیسے   ہیں
<START>    [   0.4    0.2   0.2   0.2  ]  ← Sees future!
آپ         [   0.2    0.3   0.3   0.2  ]  ← Sees future!
کیسے       [   0.1    0.2   0.4   0.3  ]  ← Sees future!
ہیں        [   0.2    0.2   0.3   0.3  ]
```

**With Masking (CORRECT):**
```
Masked Self-Attention Matrix:

              <START>  آپ   کیسے   ہیں
<START>    [   1.0    0.0   0.0   0.0  ]  ✓
آپ         [   0.4    0.6   0.0   0.0  ]  ✓
کیسے       [   0.2    0.3   0.5   0.0  ]  ✓
ہیں        [   0.2    0.2   0.3   0.3  ]  ✓

Lower triangular pattern!
```

---

## 🔢 Mathematical Formulation

### Standard Self-Attention (Encoder)

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) × V
```

### Masked Self-Attention (Decoder)

```
scores = Q·K^T / √d_k

mask = lower_triangular_matrix(seq_len)

masked_scores = scores + (1 - mask) × (-∞)

Attention = softmax(masked_scores) × V
```

---

### Python Implementation

```python
import numpy as np

def masked_self_attention(Q, K, V, mask=None):
    """
    Compute masked self-attention.
    
    Args:
        Q, K, V: Query, Key, Value matrices (seq_len, d_k)
        mask: Optional mask (seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    
    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_k)
    
    # Apply mask
    if mask is not None:
        scores = scores + (1 - mask) * -1e9
    
    # Softmax
    weights = softmax(scores)
    
    # Weighted sum
    output = weights @ V
    
    return output, weights

def create_causal_mask(seq_len):
    """Create lower triangular mask."""
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask

# Example usage
seq_len = 4
Q = np.random.randn(seq_len, 64)
K = np.random.randn(seq_len, 64)
V = np.random.randn(seq_len, 64)

mask = create_causal_mask(seq_len)
print("Causal Mask:")
print(mask)

output, weights = masked_self_attention(Q, K, V, mask)
print("\nAttention Weights (with masking):")
print(weights)
```

**Output:**
```
Causal Mask:
[[1. 0. 0. 0.]
 [1. 1. 0. 0.]
 [1. 1. 1. 0.]
 [1. 1. 1. 1.]]

Attention Weights:
[[1.0  0.0  0.0  0.0]   ← Position 1 only sees itself
 [0.4  0.6  0.0  0.0]   ← Position 2 sees 1-2
 [0.2  0.3  0.5  0.0]   ← Position 3 sees 1-3
 [0.2  0.2  0.3  0.3]]  ← Position 4 sees all
```

---

## 📊 Training vs Inference Comparison

### Training (with Masking)

```
Input: <START> آپ کیسے ہیں
       ↓       ↓   ↓   ↓
   [Decoder with Masked Attention]
       ↓       ↓   ↓   ↓
Predict: آپ  کیسے ہیں <END>

All predictions in ONE pass!
Time: Fast ✓
```

**Loss Calculation:**
```python
predictions = [آپ, کیسے, ہیں, "<END>"]
targets =     [آپ, کیسے, ہیں, "<END>"]

loss = cross_entropy(predictions, targets)
```

---

### Inference (Autoregressive)

```
Step 1:
  Input: <START>
  Predict: آپ

Step 2:
  Input: <START>, آپ
  Predict: کیسے

Step 3:
  Input: <START>, آپ, کیسے
  Predict: ہیں

Step 4:
  Input: <START>, آپ, کیسے, ہیں
  Predict: <END>

Multiple passes (sequential)
Time: Slower (but necessary!)
```

---

## 🎯 Why This Works

### Key Properties

1. **Training = Parallel**
   - Process entire sequence at once
   - Fast training ✓

2. **Masking = Sequential Integrity**
   - Each position only sees past
   - No data leakage ✓

3. **Training ≈ Inference**
   - Both respect autoregressive property
   - Model generalizes well ✓

---

### The Best of Both Worlds

```
┌─────────────────────────────────────────┐
│  WITHOUT Masking (Fast but Wrong)      │
│  - Parallel training ✓                 │
│  - Data leakage ❌                     │
│  - Inference fails ❌                  │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  Sequential Training (Slow but Correct)│
│  - No data leakage ✓                   │
│  - Inference works ✓                   │
│  - Training too slow ❌                │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  WITH Masking (Best of Both!)          │
│  - Parallel training ✓                 │
│  - No data leakage ✓                   │
│  - Inference works ✓                   │
│  - Fast training ✓                     │
└─────────────────────────────────────────┘
```

---

## 🔑 Key Takeaways

1. **Autoregressive:** Each output depends on all previous outputs

2. **Training Dilemma:**
   - Sequential training: correct but slow
   - Parallel training: fast but leaks data

3. **Masked Attention Solution:**
   - Lower triangular mask blocks future positions
   - Enables parallel training without data leakage

4. **Mask Pattern:**
   ```
   [1 0 0 0]
   [1 1 0 0]
   [1 1 1 0]
   [1 1 1 1]
   ```

5. **Training = Inference:**
   - Both respect autoregressive constraints
   - Model learns correct dependencies

---

## 🔗 Interactive Resources

🔗 **[Transformer Explainer (Interactive Visualization)](https://poloclub.github.io/transformer-explainer/)**

🔗 **[Single vs Multi-Head Attention (ByHand.ai)](https://www.byhand.ai/p/library-models-attention-single-vs-multi-head)**

🔗 **[Self-Attention vs Cross-Attention (ByHand.ai)](https://www.byhand.ai/p/library-models-attention-self-vs-cross)**

---

## 📝 Complete Implementation

```python
import torch
import torch.nn as nn

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)
        
        # Project to Q, K, V
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention with mask
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        scores = scores + mask  # Apply causal mask
        
        weights = torch.softmax(scores, dim=-1)
        output = weights @ V
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, d_model)
        output = self.W_O(output)
        
        return output

# Usage
model = MaskedMultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)  # (batch, seq_len, d_model)
output = model(x)
print(output.shape)  # torch.Size([2, 10, 512])
```

**That's Masked Multi-Head Attention explained!** 🎉
