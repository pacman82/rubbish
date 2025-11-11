mod data;
mod tokenizer;

use std::fs;

use anyhow::Context;
use candle_core::{Device, Tensor};

use self::{data::Data, tokenizer::Tokenizer};

fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;
    eprintln!("Using device: {device:?}");

    let text = fs::read_to_string("../input.txt").context("Error opening input text.")?;
    let tokenizer = Tokenizer::from_text(&text);
    let tokenized = tokenizer.encode(&text);

    let data = Data::from_tokenized(tokenized, &device);

    Ok(())
}
