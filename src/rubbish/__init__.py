import torch

from .tokenizer import Tokenizer
from .data import Data
from .model import BigramLanguageModel


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
