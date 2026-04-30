# 09. Cross-Attention (Encoder-Decoder Attention)

<div align="center">
  <img src="Attention_is_all_you_need.jpeg" alt="Attention is All You Need" width="700">
  <p><em>The paper that started it all: "Attention is All You Need" (Vaswani et al., 2017)</em></p>
</div>

---

## 🎯 What is Cross-Attention?

**Cross-attention allows the decoder to focus on relevant parts of the input sequence while generating the output.**

**Key Difference from Self-Attention:**
- **Self-Attention:** Q, K, V all from the SAME sequence
- **Cross-Attention:** Q from one sequence, K & V from ANOTHER sequence

---

## 🔄 The Need for Cross-Attention

### Translation Example

**Task:** "I like eating ice-cream" (English) → "مجھے آئسکریم کھانا پسند ہے" (Urdu)

**Problem:** How does the decoder know WHAT to translate?

```
Encoder Output: Processed "I like eating ice-cream"
                ↓
Decoder must attend to relevant English words
                ↓
When generating "آئسکریم" → Focus on "ice-cream"
When generating "کھانا" → Focus on "eating"
When generating "پسند" → Focus on "like"
```

**Cross-attention provides this connection!**

---

## 📊 Self-Attention vs Cross-Attention

### Self-Attention (What We've Learned)

```
┌─────────────────────────────┐
│     Self-Attention          │
│                             │
│  Input: "We are friends"    │
│     ↓      ↓      ↓         │
│    We    are   friends      │
│     ↓      ↓      ↓         │
│  Generate Q, K, V           │
│  from SAME sequence         │
│     ↓                       │
│  Attention(Q, K, V)         │
│     ↓                       │
│  Output: Context-aware      │
│          embeddings         │
└─────────────────────────────┘

All from ONE sequence!
```

---

### Cross-Attention (New!)

```
┌─────────────────────────────────────────┐
│         Cross-Attention                 │
│                                         │
│  Decoder Input: "ہم دوست ہیں"          │
│     ↓      ↓      ↓                     │
│  Generate Q (Query)                     │
│  from decoder sequence                  │
│     ↓                                   │
│     ├──────────────────────┐            │
│     ↓                      ↓            │
│  Encoder Output:    Generate K, V      │
│  "We are friends"   from encoder!      │
│     ↓      ↓      ↓                     │
│    We    are   friends                 │
│     ↓                                   │
│  Attention(Q_decoder, K_encoder, V_encoder) │
│     ↓                                   │
│  Output: Decoder attends to encoder    │
└─────────────────────────────────────────┘

Q from decoder, K & V from encoder!
```

---

## 🔑 Key Difference: Input Sources

### Self-Attention

```
Q = W_Q × X
K = W_K × X    ← Same X!
V = W_V × X
```

**Single source sequence.**

---

### Cross-Attention

```
Q = W_Q × X_decoder      ← From decoder
K = W_K × X_encoder      ← From encoder!
V = W_V × X_encoder      ← From encoder!
```

**Two different source sequences!**

---

## 📐 How Cross-Attention Works

### Step-by-Step Process

**Given:**
- Encoder output: Embeddings of "I like eating ice-cream"
- Decoder state: Currently generating Urdu translation

---

### Step 1: Generate Query (from Decoder)

```
Decoder current state: "مجھے"
                       ↓
                    W_Q matrix
                       ↓
Query vector: Q_مجھے = [q1, q2, ..., q64]

Q represents: "What am I looking for in the input?"
```

---

### Step 2: Generate Keys & Values (from Encoder)

```
Encoder outputs for each English word:
  "I"          → K_I,    V_I
  "like"       → K_like, V_like
  "eating"     → K_eating, V_eating
  "ice-cream"  → K_ice-cream, V_ice-cream

K represents: "What information do I have?"
V represents: "The actual information content"
```

---

### Step 3: Compute Attention Scores

```
Similarity between Query and each Key:

score_I = Q_مجھے · K_I = 45
score_like = Q_مجھے · K_like = 12
score_eating = Q_مجھے · K_eating = 8
score_ice-cream = Q_مجھے · K_ice-cream = 5
```

**"مجھے" (me/I) most similar to "I" in English! ✓**

---

### Step 4: Apply Softmax

```
Softmax([45, 12, 8, 5]) / √d_k

Attention weights:
  "I":         0.85  ← High attention!
  "like":      0.08
  "eating":    0.04
  "ice-cream": 0.03

Total = 1.0
```

---

### Step 5: Weighted Sum of Values

```
Output = Σ (weight_i × V_i)

Output = 0.85×V_I + 0.08×V_like + 0.04×V_eating + 0.03×V_ice-cream

Result: Context vector focusing on "I"
```

**This helps decoder generate "مجھے" correctly!**

---

## 🎨 Complete Example: Translation

**English:** "I like eating ice-cream"  
**Urdu:** "مجھے آئسکریم کھانا پسند ہے"

### Cross-Attention Matrix

```
Decoder → | Encoder Input →
          |  I    like  eating  ice-cream
──────────┼────────────────────────────────
مجھے      | 0.85  0.08   0.04    0.03      ← Focuses on "I"
آئسکریم   | 0.05  0.03   0.10    0.82      ← Focuses on "ice-cream"
کھانا     | 0.03  0.05   0.88    0.04      ← Focuses on "eating"
پسند      | 0.04  0.87   0.06    0.03      ← Focuses on "like"
ہے        | 0.70  0.15   0.08    0.07      ← Focuses on "I"/"like"
```

**Each Urdu word attends to relevant English word(s)!**

---

## 📊 Visual: Attention Pattern

```
English Input (Encoder):
┌───┬──────┬────────┬───────────┐
│ I │ like │ eating │ ice-cream │
└───┴──────┴────────┴───────────┘
  ↑    ↑      ↑         ↑
  │    │      │         │
  │    │      │         └─────┐
  │    │      └───────┐       │
  │    └──────┐       │       │
  └────┐      │       │       │
       │      │       │       │
Urdu Output (Decoder):
┌──────┬─────────┬──────┬──────┬────┐
│ مجھے │ آئسکریم │ کھانا │ پسند │ ہے │
└──────┴─────────┴──────┴──────┴────┘

Thick lines = high attention
Thin lines = low attention
```

---

## 🔍 Detailed Architecture

### In the Decoder Block

```
┌────────────────────────────────────┐
│      DECODER BLOCK                 │
│                                    │
│  1. Masked Self-Attention         │
│     (Decoder attends to itself)   │
│         ↓                          │
│  2. Add & Norm                     │
│         ↓                          │
│  3. CROSS-ATTENTION ← We are here!│
│     (Decoder attends to Encoder)  │
│         ↓                          │
│  4. Add & Norm                     │
│         ↓                          │
│  5. Feed-Forward                   │
│         ↓                          │
│  6. Add & Norm                     │
└────────────────────────────────────┘
```

---

### Information Flow

```
Encoder Output (Memory)
      ↓
  [K, V matrices]
      ↓
      ├─────────────────┐
      ↓                 ↓
┌──────────┐      ┌──────────┐
│ Decoder  │      │  Cross   │
│ Hidden   │─────→│Attention │
│ State    │  Q   │          │
└──────────┘      └──────────┘
                       ↓
                  Context Vector
```

**Decoder queries encoder's memory!**

---

## 💻 Mathematical Formulation

### Cross-Attention Formula

```
Given:
  X_decoder: Decoder hidden states
  X_encoder: Encoder output (memory)

Q = X_decoder × W_Q     (from decoder)
K = X_encoder × W_K     (from encoder)
V = X_encoder × W_V     (from encoder)

Attention(Q, K, V) = softmax(Q·K^T / √d_k) × V
```

**Key Insight:** Q asks, K & V answer from different sequence!

---

## 🔢 Dimension Tracking

**Example:**
- English sentence: 4 words
- Urdu sentence: 5 words
- d_model = 512

```
Encoder Output:
  Shape: (4, 512)  ← 4 English words
  
Decoder State:
  Shape: (5, 512)  ← 5 Urdu words

Cross-Attention:
  Q: (5, 512) × W_Q → (5, 64×8) = (5, 512)
  K: (4, 512) × W_K → (4, 64×8) = (4, 512)
  V: (4, 512) × W_V → (4, 64×8) = (4, 512)
  
Attention Scores:
  Q · K^T = (5, 512) × (512, 4) = (5, 4)
                                   ↑   ↑
                          Urdu  English
                          words  words

After Softmax & Weighted Sum:
  Output: (5, 512)  ← Same as decoder input
```

**Each Urdu word gets context from all English words!**

---

## 🎓 Relation to Earlier Attention Mechanisms

### Evolution of Attention

**1. Bahdanau Attention (2014)**
```
- First neural attention mechanism
- Used in seq2seq models
- Decoder attends to encoder states
- Additive attention: score = v^T tanh(W[h_decoder; h_encoder])
```

**2. Luong Attention (2015)**
```
- Improved Bahdanau
- Multiplicative attention: score = h_decoder · h_encoder
- Simpler, faster
```

**3. Transformer Cross-Attention (2017)**
```
- Multi-head attention
- Scaled dot-product
- Parallel processing
- Query-Key-Value framework
- What we use today! ✓
```

**Cross-attention is the modern, scalable version of these ideas!**

---

## 🌐 Use Cases

### 1. Machine Translation

```
Source: "The cat sat on the mat"
Target: "بلی چٹائی پر بیٹھی"

Cross-attention aligns source and target words.
```

---

### 2. Image Captioning

```
Image (Encoder): CNN features of a cat image
Caption (Decoder): "A fluffy cat sitting on a sofa"

Cross-attention:
  "fluffy" → Attends to cat texture features
  "sitting" → Attends to pose features
  "sofa" → Attends to furniture features
```

---

### 3. Text-to-Image Generation (DALL-E, Stable Diffusion)

```
Text: "A red apple on a wooden table"

Cross-attention:
  Image region 1 → Attends to "red apple"
  Image region 2 → Attends to "wooden table"
```

---

### 4. Text-to-Speech

```
Text: "Hello, how are you?"
Speech: Audio waveform

Cross-attention aligns text phonemes to audio frames.
```

---

### 5. Video Captioning

```
Video frames → Encoder
Caption → Decoder with cross-attention to frames
```

**Any task with TWO different modalities or sequences!**

---

## 📝 Implementation

```python
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Separate projections for Q, K, V
        self.W_Q = nn.Linear(d_model, d_model)  # For decoder
        self.W_K = nn.Linear(d_model, d_model)  # For encoder
        self.W_V = nn.Linear(d_model, d_model)  # For encoder
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, decoder_hidden, encoder_output):
        """
        Args:
            decoder_hidden: (batch, target_len, d_model)
            encoder_output: (batch, source_len, d_model)
        """
        batch_size = decoder_hidden.size(0)
        
        # Q from decoder, K & V from encoder
        Q = self.W_Q(decoder_hidden)
        K = self.W_K(encoder_output)
        V = self.W_V(encoder_output)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        output = weights @ V
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.num_heads * self.d_k)
        output = self.W_O(output)
        
        return output, weights

# Example usage
cross_attn = CrossAttention(d_model=512, num_heads=8)

decoder_hidden = torch.randn(2, 5, 512)   # 5 target words
encoder_output = torch.randn(2, 4, 512)   # 4 source words

output, weights = cross_attn(decoder_hidden, encoder_output)

print(f"Output shape: {output.shape}")      # (2, 5, 512)
print(f"Attention weights: {weights.shape}") # (2, 8, 5, 4)
#                                              batch, heads, target, source
```

---

## 🎯 Key Takeaways

1. **Cross-Attention connects two different sequences**
   - Decoder queries encoder's memory

2. **Input Sources:**
   - Q from decoder (what I'm looking for)
   - K, V from encoder (what information is available)

3. **Attention Matrix:**
   - Rows: Decoder positions
   - Columns: Encoder positions
   - Shows alignment between sequences

4. **Evolution:**
   - Modern version of Bahdanau/Luong attention
   - Multi-head, scalable, parallel

5. **Applications:**
   - Translation, captioning, text-to-image, TTS
   - Any multimodal or sequence-to-sequence task

6. **In Transformer Decoder:**
   - Comes after masked self-attention
   - Before feed-forward network
   - Critical for connecting encoder and decoder

---

## 🔗 Interactive Resources

🔗 **[Transformer Explainer (Interactive Visualization)](https://poloclub.github.io/transformer-explainer/)**

🔗 **[Single vs Multi-Head Attention (ByHand.ai)](https://www.byhand.ai/p/library-models-attention-single-vs-multi-head)**

🔗 **[Self-Attention vs Cross-Attention (ByHand.ai)](https://www.byhand.ai/p/library-models-attention-self-vs-cross)**

---

## 📊 Summary Comparison

| Aspect | Self-Attention | Cross-Attention |
|--------|---------------|-----------------|
| **Q source** | Same sequence | Decoder |
| **K source** | Same sequence | Encoder |
| **V source** | Same sequence | Encoder |
| **Purpose** | Relate words in same sequence | Relate words across sequences |
| **Used in** | Encoder, Decoder (masked) | Decoder only |
| **Attention shape** | (seq_len, seq_len) | (target_len, source_len) |

**That's Cross-Attention explained!** 🎉
