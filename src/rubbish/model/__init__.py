from typing import override

import torch
import torch.nn as nn
from torch.types import Tensor

NUMBER_OF_EMBEDDING_DIMENSIONS = 32
CONTEXT_SIZE = 8


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table: nn.Embedding = nn.Embedding(
            vocab_size, NUMBER_OF_EMBEDDING_DIMENSIONS
        )
        self.positional_embedding_table: nn.Embedding = nn.Embedding(
            CONTEXT_SIZE, NUMBER_OF_EMBEDDING_DIMENSIONS
        )
        self.lm_head: nn.Linear = nn.Linear(NUMBER_OF_EMBEDDING_DIMENSIONS, vocab_size)

    @override
    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)
        # positional_embeddings = self.positional_embedding_table(
        #     torch.arange(T, device=idx.device)
        # )
        # embeddings = token_embeddings + positional_embeddings
        embeddings = token_embeddings
        logits = self.lm_head(embeddings)

        if targets is not None:
            # Optionally calculate loss and reduce dimension of logits

            # Logits have dimension (Batch, Time, Channel)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            # Targets has have dimension (Batch, Time)
            targets = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx: Tensor, max_new_tokens: int) -> Tensor:
        print(idx.shape)
        print(idx)
        for _ in range(max_new_tokens):
            logits, _loss = self(idx)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
