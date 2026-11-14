# GPT-2 Architecture Notes (Cheat Sheet)

## 1. What is a Tensor? (Mental Model)
A **Tensor** is just a container of numbers with $N$ dimensions:
* **0D (Scalar):** Single number (e.g. `loss = 2.4`).
* **1D (Vector):** A list `[·, ·, ·]` (e.g. one word's embedding).
* **2D (Matrix):** A grid of rows & columns (e.g. `idx` with shape `(B, T)`).
* **3D Tensor (A STACK of 2D grids):** Like a notebook!
  * **Stack / Pages (`B`):** Each page is one sentence.
  * **Rows on page (`T`):** Each row is one word in that sentence.
  * **Columns (`C`):** The vector numbers describing that word.

---

## 2. Core Dimensions: `(B, T, C)`
* **`B` (Batch Size):** Number of independent sentences processed in parallel (GPU efficiency).
* **`T` (Time / Sequence Length):** Number of tokens in the current sentence ($T \le \text{block\_size}$).
* **`C` (Channels / `n_embd`):** Dimension of the vector representing a single token (e.g. 768 in GPT-2).

> **Important Real-World Detail: Ragged Sequences & Padding**
> * Tensors MUST be rigid rectangles/cubes; all rows in a batch must have the same length `T`.
> * If sentences differ in length, shorter ones are filled with `<PAD>` tokens up to the longest sentence in that batch.
> * Modern LLMs (LLaMA, FlashAttention) use **Sequence Packing**: concatenating sentences with `<EOS>` separators to avoid wasting GPU compute on `<PAD>`.

---

## 3. How Embedding Lookup Works
```
Input: idx (B, T)
  (Sentence b, Word t) -> Gives Token ID -> Points to Row in Table -> Returns Vector (C)
Result: tok_emb of shape (B, T, C)
```
* **`idx` (Indices):** Integer pointers to rows in the table.
* **`torch.randint(0, vocab_size, (B, T))`:**
  * **Range:** `0` to `vocab_size - 1` (valid token IDs).
  * **Shape:** `(B, T)` (sentences $\times$ words).

---

## 4. The Input Layer: `wte + wpe`
Raw text $\rightarrow$ Tokens $\rightarrow$ Token IDs (`idx`).

* **`wte` (Word Token Embedding):** Lookup table `(vocab_size, n_embd)`.
  * Answers: **WHO** is this token? (e.g. "dog" $\rightarrow$ vector).
* **`wpe` (Word Positional Embedding):** Lookup table `(block_size, n_embd)`.
  * Answers: **WHERE** is this token? (e.g. position 0 vs. position 5).
* **Combination:**
  ```python
  x = wte(idx) + wpe(pos)  # Shape: (B, T, C)
  ```
* **AHA! Insight:**
  * Without position: `"dog bites man"` looks identical to `"man bites dog"`.
  * `WHO + WHERE = Complete Meaning`.

---

## 5. Model Blueprint Parameters (`GPTConfig`)
* **`vocab_size` (50,257):** Total unique tokens in vocabulary.
* **`block_size` (1,024):** Max **context window** (longest memory span).
* **`n_embd` (768):** Vector size per token (`C`).
* **`n_layer` (12):** Number of Transformer blocks stacked vertically.
* **`n_head` (12):** Number of parallel attention heads ($768 / 12 = 64$ dims per head).

---

## 6. Next: What happens to `x`?
`x` enters the **Transformer Blocks**:
1. **LayerNorm 1** $\rightarrow$ Normalizes vector magnitudes.
2. **Causal Self-Attention** $\rightarrow$ Tokens look back and communicate ($Q, K, V$).
3. **LayerNorm 2** $\rightarrow$ Normalizes again.
4. **MLP** $\rightarrow$ Tokens process what they learned ($4 \times \text{dimension}$).
5. **LM Head** $\rightarrow$ Projects vectors back to vocabulary to predict next token.
