mod data;
mod device;
mod tokenizer;

use std::fs;

use anyhow::Context;

use crate::device::choose_device;

use self::{data::Data, tokenizer::Tokenizer};

fn main() -> anyhow::Result<()> {
    let device = choose_device();

    let text = fs::read_to_string("../input.txt").context("Error opening input text.")?;
    let tokenizer = Tokenizer::from_text(&text);
    let tokenized = tokenizer.encode(&text);

    let data = Data::from_tokenized(tokenized, &device);

    Ok(())
}
