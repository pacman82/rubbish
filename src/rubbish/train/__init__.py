from pathlib import Path
from typing import Callable

import torch
from torch.optim import AdamW

from rubbish.data import Data
from rubbish.model import CONTEXT_SIZE, BigramLanguageModel

learning_rate = 3e-4
max_iterations = 5000
eval_iterations = 200
eval_interval = 500


@torch.no_grad()
def estimate_loss(model: BigramLanguageModel, data: Data, device: str):
    out = {}
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
    return out


def train(
    model: BigramLanguageModel,
    data: Data,
    device: str,
    eval: Callable[[BigramLanguageModel, int], None] | None = None,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if Path("optimizer.pt").is_file():
        model.load_state_dict(torch.load("model.pt"))
        optimizer.load_state_dict(torch.load("optimizer.pt"))

    for step in range(max_iterations):
        if step % eval_interval == 0 and eval is not None:
            model.eval()
            eval(model, step)
            model.train()

            torch.save(model.state_dict(), "model.pt")
            torch.save(optimizer.state_dict(), "optimizer.pt")

        train_step(model, optimizer, data, device)

    if eval is not None:
        model.eval()
        eval(model, max_iterations)
        model.train()

    torch.save(model.state_dict(), "model.pt")
    torch.save(optimizer.state_dict(), "optimizer.pt")


def train_step(
    model: BigramLanguageModel, optimizer: AdamW, data: Data, device: str
) -> float:
    xb, yb = data.training_batch(CONTEXT_SIZE, device=device)
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()
