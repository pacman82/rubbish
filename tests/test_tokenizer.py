from rubbish import Tokenizer


def test_text_should_be_the_same_after_decoding_encoding():
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = Tokenizer(text)

    sample_text = "Hello, World!"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)

    assert decoded == sample_text, (
        f"Decoded text '{decoded}' does not match original '{sample_text}'"
    )
