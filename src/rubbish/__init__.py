from typing import override

import torch
import torch.nn as nn
from torch.types import Tensor

from .tokenizer import Tokenizer
from .data import Data


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table: nn.Embedding = nn.Embedding(vocab_size, vocab_size)

    @override
    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        logits = self.token_embedding_table(idx)

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
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def main() -> None:
    torch.manual_seed(1337)

    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = Tokenizer(text)
    context_size = 8
    batch_size = 32
    data = Data(text, tokenizer, context_size, batch_size)

    model = BigramLanguageModel(tokenizer.vocab_size())

    idx = torch.zeros((1, 1), dtype=torch.long)
    generated_batch = model.generate(idx=idx, max_new_tokens=100)
    print(tokenizer.decode(generated_batch[0].tolist()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for steps in range(10000):
        xb, yb = data.training_batch()
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(loss.item())

    idx = torch.zeros((1, 1), dtype=torch.long)
    generated_batch = model.generate(idx=idx, max_new_tokens=100)
    print(tokenizer.decode(generated_batch[0].tolist()))
