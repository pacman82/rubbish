import torch
from torch.xpu import device

from .data import Data
from .model import BigramLanguageModel
from .tokenizer import Tokenizer
from .train import train

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

batch_size = 32


def print_generate_text(model: BigramLanguageModel, tokenizer: Tokenizer) -> None:
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_batch = model.generate(idx=context, max_new_tokens=100)
    print(tokenizer.decode(generated_batch[0].tolist()))


def main() -> None:
    torch.manual_seed(1337)

    print(f"Using device: {device}")

    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = Tokenizer(text)
    data = Data(text, tokenizer, batch_size, device)

    model = BigramLanguageModel(tokenizer.vocab_size())
    model = model.to(device)

    print_generate_text(model, tokenizer)

    train(model, data)

    print_generate_text(model, tokenizer)
