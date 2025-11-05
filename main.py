from typing import final

import torch.nn as nn
import torch


@final
class Tokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.s_to_i = {ch: i for i, ch in enumerate(chars)}
        self.i_to_s = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.s_to_i[c] for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join([self.i_to_s[i] for i in tokens])


def _batch(
    data: torch.Tensor, context_size: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(data) >= context_size + 1, "Data too small for the given context size."
    ix = torch.randint(high=len(data) - context_size, size=(batch_size,))
    context = torch.stack([data[i : i + context_size] for i in ix])
    target = torch.stack([data[i + 1 : i + context_size + 1] for i in ix])
    return context, target


@final
class Data:
    def __init__(
        self, text: str, tokenizer: Tokenizer, context_size: int, batch_size: int
    ):
        tokenized = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(0.9 * len(tokenized))
        self.training = tokenized[:n]
        self.validation = tokenized[n:]
        self.context_size = context_size
        self.batch_size = batch_size

    def training_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _batch(self.training, self.context_size, self.batch_size)

    def validation_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _batch(self.validation, self.context_size, self.batch_size)


def main():
    torch.manual_seed(1337)

    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = Tokenizer(text)
    context_size = 8
    batch_size = 4
    data = Data(text, tokenizer, context_size, batch_size)

    xb, yb = data.training_batch()

    print("inputs:")
    print(xb.shape)
    print(xb)

    print("targets:")
    print(yb.shape)
    print(yb)

    print("---")

    for b in range(batch_size):
        for t in range(context_size):
            context = xb[b, : t + 1]
            target = yb[b, t]
            print(f"When input is {context.tolist()}, the target is {target}")


if __name__ == "__main__":
    main()
