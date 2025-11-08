from typing import override

import torch
import torch.nn as nn
from torch.types import Tensor

NUMBER_OF_EMBEDDING_DIMENSIONS = 32
CONTEXT_SIZE = 8


class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(NUMBER_OF_EMBEDDING_DIMENSIONS, head_size, bias=False)
        self.query = nn.Linear(NUMBER_OF_EMBEDDING_DIMENSIONS, head_size, bias=False)
        self.value = nn.Linear(NUMBER_OF_EMBEDDING_DIMENSIONS, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(CONTEXT_SIZE, CONTEXT_SIZE)))

    def forward(self, embedding: Tensor) -> Tensor:
        B, T, C = embedding.shape

        keys = self.key(embedding)
        queries = self.query(embedding)

        attention_scores = queries @ keys.transpose(-2, -1) * (C**-0.5)
        attention_scores = attention_scores.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)

        values = self.value(embedding)
        out = attention_weights @ values

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(
            num_heads * head_size, NUMBER_OF_EMBEDDING_DIMENSIONS
        )

    def forward(self, x: Tensor) -> Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        head_size = embedding_dim // num_heads
        self.self_attention = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForward(embedding_dim)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table: nn.Embedding = nn.Embedding(
            vocab_size, NUMBER_OF_EMBEDDING_DIMENSIONS
        )
        self.positional_embedding_table: nn.Embedding = nn.Embedding(
            CONTEXT_SIZE, NUMBER_OF_EMBEDDING_DIMENSIONS
        )
        self.blocks = nn.Sequential(
            Block(NUMBER_OF_EMBEDDING_DIMENSIONS, num_heads=4),
            Block(NUMBER_OF_EMBEDDING_DIMENSIONS, num_heads=4),
            Block(NUMBER_OF_EMBEDDING_DIMENSIONS, num_heads=4),
            nn.LayerNorm(NUMBER_OF_EMBEDDING_DIMENSIONS),
        )
        self.lm_head: nn.Linear = nn.Linear(NUMBER_OF_EMBEDDING_DIMENSIONS, vocab_size)

    @override
    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)
        positional_embeddings = self.positional_embedding_table(
            torch.arange(T, device=idx.device)
        )
        x = token_embeddings + positional_embeddings
        x = self.blocks(x)
        logits = self.lm_head(x)

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
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -CONTEXT_SIZE:]
            logits, _loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
