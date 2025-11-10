import torch

from .data import Data
from .model import BigramLanguageModel
from .tokenizer import Tokenizer
from .train import estimate_loss, train

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 64


def print_generate_text(
    model: BigramLanguageModel, tokenizer: Tokenizer, max_new_tokens=100
) -> None:
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_batch = model.generate(idx=context, max_new_tokens=max_new_tokens)
    print(tokenizer.decode(generated_batch[0].tolist()))


class Evaluator:
    def __init__(
        self, model: BigramLanguageModel, data: Data, device: str, tokenizer: Tokenizer
    ):
        self.data = data
        self.device = device
        self.tokenizer = tokenizer

    @torch.no_grad()
    def eval(self, model: BigramLanguageModel, step: int):
        losses = estimate_loss(model, self.data, self.device)
        print_generate_text(model, self.tokenizer, max_new_tokens=512)
        print(
            f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )


def main() -> None:
    torch.manual_seed(1337)

    print(f"Using device: {device}")

    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = Tokenizer(text)
    data = Data(text, tokenizer, batch_size)

    model = BigramLanguageModel(tokenizer.vocab_size())
    model = model.to(device)

    print(f"Number of model parameters {sum(p.numel() for p in model.parameters())}")

    evaluator = Evaluator(model, data, device, tokenizer)

    train(model, data, device, eval=evaluator.eval)

    print_generate_text(model, tokenizer, max_new_tokens=4096)
