class Tokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.s_to_i: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.i_to_s: dict[int, str] = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.s_to_i[c] for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join([self.i_to_s[i] for i in tokens])

    def vocab_size(self) -> int:
        return len(self.i_to_s)
