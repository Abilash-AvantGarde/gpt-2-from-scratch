import torch
import torch.nn as nn

# --- Our Tiny Config ---
B, T, C = 2, 5, 3  # Batch, Time (sequence length), Channels (n_embd)
vocab_size = 10
block_size = 8     # Our model's max sequence length is 8

# --- The Layers ---
# The same token embedding table from last section
token_embedding_table = nn.Embedding(vocab_size, C)

# The new position embedding table!
# Note its "vocabulary" is the block_size
position_embedding_table = nn.Embedding(block_size, C)

# --- The Input Data ---
idx = torch.randint(0, vocab_size, (B, T))  # Shape (2, 5)

# --- Step 1: Get Token Embeddings (as before) ---
tok_emb = token_embedding_table(idx)  # Shape (B, T, C) -> (2, 5, 3)

# --- Step 2: Get Positional Embeddings ---
# We just need the positions for our CURRENT sequence length, T
pos = torch.arange(0, T, dtype=torch.long)  # Shape (T) -> tensor([0, 1, 2, 3, 4])
pos_emb = position_embedding_table(pos)     # Shape (T, C) -> (5, 3)

# --- Step 3: Combine them via addition ---
x = tok_emb + pos_emb

print("Shape of token embeddings:     ", tok_emb.shape)
print("Shape of positional embeddings:", pos_emb.shape)
print("Shape of final combined:       ", x.shape)
