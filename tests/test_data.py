from main import Data, Tokenizer


def test_sample_training_batches():
    text = "Hello, World!"
    tokenizer = Tokenizer(text)

    data = Data(text, tokenizer, context_size=8, batch_size=4)

    xb, yb = data.training_batch()
    assert xb.shape == (4, 8), f"Expected input shape (4, 8), got {xb.shape}"
    assert yb.shape == (4, 8), f"Expected target shape (4, 8), got {yb.shape}"
