import torch
from torch import nn


# Inputs Embedding
class InputsEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * torch.sqrt(
            torch.tensor(self.d_model, dtype=torch.float32)
        )


# Positional Encoding
