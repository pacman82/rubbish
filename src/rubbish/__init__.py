import torch

from .data import Data
from .model import BigramLanguageModel
from .tokenizer import Tokenizer

context_size = 8
batch_size = 32
learning_rate = 1e-2
max_iterations = 3000
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iterations = 200
eval_interval = 300


@torch.no_grad()
def estimate_loss(model: BigramLanguageModel, data: Data):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations):
            xb, yb = (
                data.training_batch() if split == "train" else data.validation_batch()
            )
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


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
    data = Data(text, tokenizer, context_size, batch_size, device)

    model = BigramLanguageModel(tokenizer.vocab_size())
    model = model.to(device)

    print_generate_text(model, tokenizer)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iterations):
        if step % eval_interval == 0:
            losses = estimate_loss(model, data)
            print(
                f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            print_generate_text(model, tokenizer)

        xb, yb = data.training_batch()
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print_generate_text(model, tokenizer)
