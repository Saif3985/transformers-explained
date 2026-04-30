# Transformers Explained: From Basics to Architecture

From Basics to Architecture

<div align="center">
  <img src="Attention_is_all_you_need.jpeg" alt="Attention is All You Need" width="700">
  <p><em>The paper that started it all: "Attention is All You Need" (Vaswani et al., 2017)</em></p>
</div>

---

## What is a Transformer?

Just like we have different neural network architectures for different types of data:
- **ANN** (Artificial Neural Networks) → Tabular data
- **CNN** (Convolutional Neural Networks) → Image data
- **RNN** (Recurrent Neural Networks) → Sequential data

**Transformers** are neural networks designed specifically for **textual/sequential data**, revolutionizing Natural Language Processing (NLP).

---

## The NLP Challenge: Computers Don't Understand Words

### The Core Problem

Computers work with numbers, not words. When we say "glacier" or "river", the computer sees gibberish. Our main goal in NLP is:

> **Convert text/words into numbers that computers can process**

### Evolution of Text Representation

#### ❌ Old Methods (The Problems)

**1. One-Hot Encoding**
```
glacier = [0, 0, 1, 0, 0]
river   = [0, 1, 0, 0, 0]
ice     = [1, 0, 0, 0, 0]
```

**Problems:**
- All words are equally distant (no meaning captured)
- Huge vectors (vocabulary size = vector length)
- No relationship between similar words

**2. Bag of Words / TF-IDF**
- Only counts word frequency
- Ignores word order completely
- No semantic meaning

---

## ✅ Better Approach: Word Embeddings

### What are Word Embeddings?

Instead of sparse one-hot vectors, we use **dense vectors that capture meaning**:

```
glacier = [0.2, 0.8, 0.5]
ice     = [0.25, 0.75, 0.55]
car     = [0.9, 0.1, 0.2]
```

**Key Insight:**
- Similar words → Similar vectors
- `glacier` ≈ `ice` (vectors are close)
- `glacier` ≠ `car` (vectors are far apart)

**How are they created?**
- Learned using models like **Word2Vec**, **GloVe**, **FastText**
- Trained on large text corpora
- Capture semantic relationships

---

## From Words to Sentences

### Representing a Sentence

**Input sentence:**
```
"The glacier is melting"
```

**As embeddings:**
```
X = [x₁, x₂, x₃, x₄]

where:
x₁ = embedding of "The"
x₂ = embedding of "glacier"
x₃ = embedding of "is"
x₄ = embedding of "melting"
```

Each `xᵢ` is a dense vector (e.g., 300 dimensions).

---

## 🔴 The Problems with Simple Approaches

Even with word embeddings, simple methods fail. Here's why:

### Problem 1: Averaging Loses Word Order

**Common approach:**
```
sentence_vector = (x₁ + x₂ + x₃ + x₄) / 4
```

**Why it fails:**

**Sentence 1:** "dog bites man"  
**Sentence 2:** "man bites dog"

👉 **Same average vector!** 😱

But completely different meanings!

**Issues:**
- ❌ No word order information
- ❌ No grammar understanding
- ❌ Wrong meaning captured

---

### Problem 2: No Context (Polysemy Problem)

In Word2Vec or GloVe:
```
bank = [0.5, 0.2, 0.1]  (always the same)
```

**Sentence 1:** "I went to the **bank** to deposit money"  
**Sentence 2:** "I sat near the river **bank**"

👉 **Same vector for both!** ❌

**The Issue:**
- Cannot understand context
- Same word → same embedding → wrong in different contexts
- This is called the **polysemy problem** (one word, multiple meanings)

---

### Problem 3: All Words Treated Equally

**Sentence:** "The glacier is melting"

In averaging, all words have equal importance:
- "**the**" ❌ (not important - just a determiner)
- "**glacier**" ✅ (important - main subject)
- "**melting**" ✅ (important - main action)

**The Issue:**
- Model cannot **focus** on important words
- No concept of word importance
- Filler words pollute the representation

---

### Problem 4: No Long-Range Dependencies

**Sentence:** "The glacier near the mountain is melting rapidly"

**What we need:**
- "melting" should strongly relate to "glacier"
- But they're far apart in the sentence

**With averaging:**
- Everything gets mixed together blindly ❌
- Long-range relationships lost
- Context from distant words ignored

---

### Problem 5: Fixed Representations (Static Embeddings)

Once trained:
```
glacier = [0.2, 0.8, 0.5]  (always the same)
```

**The Issue:**
- Representation is **static** - never changes
- No matter what sentence it appears in
- Cannot adapt to context

**Example:**
- "The **glacier** is melting" 
- "We studied the **glacier** in geography"
- "The **glacier** expedition was dangerous"

All three get the **exact same vector** for "glacier" ❌

---

## 💡 The Solution: We Need a Better Mechanism

To solve all these problems, we need a mechanism that can:

1. ✅ **Capture word order** (not just bag of words)
2. ✅ **Understand context** (different meanings in different sentences)
3. ✅ **Focus on important words** (give weights/importance)
4. ✅ **Handle long-range dependencies** (relate distant words)
5. ✅ **Create dynamic representations** (context-aware embeddings)

This is where **Self-Attention** comes in.

---

## Next: Understanding Self-Attention

Now that we understand the problems, let's see how **self-attention mechanism** solves them.

👉 Continue to: [01-self-attention.md](01-self-attention.md)
