import torch

from rubbish.data import Data
from rubbish.model import CONTEXT_SIZE, BigramLanguageModel

learning_rate = 1e-3
max_iterations = 50000
eval_iterations = 200
eval_interval = 500


@torch.no_grad()
def estimate_loss(model: BigramLanguageModel, data: Data, device: str):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations):
            xb, yb = (
                data.training_batch(CONTEXT_SIZE, device=device)
                if split == "train"
                else data.validation_batch(CONTEXT_SIZE, device=device)
            )
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train(model: BigramLanguageModel, data: Data, device: str):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iterations):
        if step % eval_interval == 0:
            losses = estimate_loss(model, data, device)
            print(
                f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        xb, yb = data.training_batch(CONTEXT_SIZE, device=device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
