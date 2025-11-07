import torch
from torch.types import Tensor

from ..tokenizer import Tokenizer


def _batch(
    data: torch.Tensor, context_size: int, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(data) >= context_size + 1, "Data too small for the given context size."
    ix = torch.randint(high=len(data) - context_size, size=(batch_size,))
    context = torch.stack([data[i : i + context_size] for i in ix])
    target = torch.stack([data[i + 1 : i + context_size + 1] for i in ix])
    context, target = context.to(device), target.to(device)
    return context, target


class Data:
    def __init__(
        self,
        text: str,
        tokenizer: Tokenizer,
        context_size: int,
        batch_size: int,
        device: str,
    ):
        tokenized = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(0.9 * len(tokenized))
        self.training: Tensor = tokenized[:n]
        self.validation: Tensor = tokenized[n:]
        self.context_size: int = context_size
        self.batch_size: int = batch_size
        self.device: str = device

    def training_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _batch(self.training, self.context_size, self.batch_size, self.device)

    def validation_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _batch(self.validation, self.context_size, self.batch_size, self.device)
