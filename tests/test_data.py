import torch

from rubbish import Data, Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


def test_sample_training_batches():
    text = "Hello, World!"
    tokenizer = Tokenizer(text)

    data = Data(text, tokenizer, batch_size=4)

    xb, yb = data.training_batch(context_size=8, device=device)
    assert xb.shape == (4, 8), f"Expected input shape (4, 8), got {xb.shape}"
    assert yb.shape == (4, 8), f"Expected target shape (4, 8), got {yb.shape}"
