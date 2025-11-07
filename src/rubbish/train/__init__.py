import torch

from rubbish.data import Data
from rubbish.model import BigramLanguageModel
from rubbish.tokenizer import Tokenizer

learning_rate = 1e-2
max_iterations = 3000
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


def train(model: BigramLanguageModel, data: Data):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iterations):
        if step % eval_interval == 0:
            losses = estimate_loss(model, data)
            print(
                f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        xb, yb = data.training_batch()
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
