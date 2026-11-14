import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 1. Configuration (The Blueprint)
# -----------------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 50257  # number of tokens in vocabulary
    block_size: int = 1024   # max sequence length (context window)
    n_layer: int = 12        # number of transformer layers
    n_head: int = 12         # number of attention heads
    n_embd: int = 768        # embedding dimension
    dropout: float = 0.1     # dropout rate


# -----------------------------------------------------------------------------
# 2. GPT-2 Model (Only what we've learned so far: Embeddings!)
# -----------------------------------------------------------------------------
class GPT2(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # The two embedding tables:
        # wte: Word Token Embeddings (vocab_size -> n_embd)
        # wpe: Word Positional Embeddings (block_size -> n_embd)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
            )
        )

    def forward(self, idx):
        # idx shape: (batch_size, sequence_length) -> (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"

        # 1. Positions tensor: [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # 2. Look up token embeddings & position embeddings
        tok_emb = self.transformer.wte(idx)  # shape: (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # shape: (T, n_embd)

        # 3. AHA! Add them together: (WHO + WHERE)
        # Broadcasting adds (T, n_embd) across the batch dimension -> (B, T, n_embd)
        x = tok_emb + pos_emb
        return x


# -----------------------------------------------------------------------------
# Quick Sanity Check (Test running our model)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Use a tiny config to verify everything works
    tiny_config = GPTConfig(vocab_size=10, block_size=8, n_embd=3)
    model = GPT2(tiny_config)

    # Dummy input: 2 sentences (batch=2), 5 tokens each (time=5)
    dummy_input = torch.tensor([[1, 4, 2, 8, 5], 
                                [3, 0, 7, 9, 2]])

    output = model(dummy_input)
    print("Input shape: ", dummy_input.shape)
    print("Output shape (B, T, n_embd):", output.shape)
