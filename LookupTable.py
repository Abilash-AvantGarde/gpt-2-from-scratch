import torch
import torch.nn as nn

vocab_size = 10
n_embd = 3

token_embedding_table = nn.Embedding(vocab_size, n_embd)

print("Token embedding table weight matrix:")
print(token_embedding_table.weight)
print("\nShape:", token_embedding_table.weight.shape)



