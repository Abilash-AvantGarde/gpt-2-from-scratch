import torch
import torch.nn as nn


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        self.wpe = nn.Embedding(
            config.block_size, config.n_embd
        )  # word positional embedding

        # AHA! MOMENT:
        # Without position: "dog bites man" looks identical to "man bites dog".
        # Word (WHO) + Position (WHERE) = Complete Meaning.
        #
        # Visual Example (with tiny 2D numbers):
        # Suppose the word "dog" vector is at:      [1.0, 2.0]
        # Position 0 (start of sentence) is at:     [0.1, 0.0]
        # Position 5 (end of sentence) is at:       [0.0, 0.9]
        #
        # "dog" at pos 0 = [1.0, 2.0] + [0.1, 0.0] = [1.1, 2.0]  <- unique spot!
        # "dog" at pos 5 = [1.0, 2.0] + [0.0, 0.9] = [1.0, 2.9]  <- different spot!


