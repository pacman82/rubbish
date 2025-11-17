use std::collections::{HashMap, HashSet};

/// The tokenizer is responsible for converting text into a sequence of numbers and back.
///
/// This implementation uses a simple character-level tokenization approach, where each unique
/// character in the input text is assigned an integer.
pub struct Tokenizer {
    /// Spars lookup from character to integer.
    char_to_int: HashMap<char, u32>,
    /// Lookup from integer to character. We use a vector, because we know our token_ids to be
    /// contigiuous.
    int_to_char: Vec<char>,
}

impl Tokenizer {
    /// Creates a tokenizer from the given text.
    pub fn from_text(text: &str) -> Self {
        let set: HashSet<_> = text.chars().collect();
        let mut chars = Vec::from_iter(set.into_iter());
        chars.sort();
        let char_to_int = chars
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32))
            .collect::<std::collections::HashMap<_, _>>();
        let int_to_char = chars;
        eprintln!("Size of vocabulary: {}", int_to_char.len());
        Tokenizer {
            char_to_int,
            int_to_char,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|c| *self.char_to_int.get(&c).unwrap())
            .collect()
    }

    #[allow(unused)]
    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .map(|&i| self.int_to_char[i as usize])
            .collect()
    }

    pub fn decode_single(&self, token: u32) -> char {
        self.int_to_char[token as usize]
    }

    pub fn vocab_size(&self) -> usize {
        self.int_to_char.len()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn hello_world_tokenization_roundtrip() {
        let tokenizer = Tokenizer::from_text("Hello, World!");

        let roundtrip = |input: &str| {
            let encoded = tokenizer.encode(input);
            let decoded = tokenizer.decode(&encoded);
            decoded
        };

        assert_eq!("Hello, World!", roundtrip("Hello, World!"));
        assert_eq!("Hello", roundtrip("Hello"));
        assert_eq!("World", roundtrip("World"));
        assert_eq!("Hello!", roundtrip("Hello!"));
    }

    #[test]
    fn tokenize_single_characters() {
        let tokenizer = Tokenizer::from_text("abc \n");

        assert_eq!('\n', tokenizer.decode_single(0));
        assert_eq!(' ', tokenizer.decode_single(1));
        assert_eq!('a', tokenizer.decode_single(2));
        assert_eq!('b', tokenizer.decode_single(3));
        assert_eq!('c', tokenizer.decode_single(4));
    }
}
