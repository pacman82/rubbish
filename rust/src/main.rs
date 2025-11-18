mod data;
mod device;
mod generate;
mod model;
mod tokenizer;
mod train;

use std::fs;

use anyhow::Context;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use rand::rng;

use crate::train::Training;

use self::{
    data::Data, device::choose_device, generate::GenerateIter, model::Model, tokenizer::Tokenizer,
};

const BATCH_SIZE: usize = 32;
const CONTEXT_SIZE: usize = 8;
const LEARNING_RATE: f64 = 1e-3;

fn main() -> anyhow::Result<()> {
    let device = choose_device();

    let text = fs::read_to_string("../input.txt").context("Error opening input text.")?;
    let tokenizer = Tokenizer::from_text(&text);
    let tokenized = tokenizer.encode(&text);

    let batch_sampler = Data::new(tokenized, &device, rng());

    let model = Model::new(device.clone(), tokenizer.vocab_size());
    eprintln!(
        "Number of trainable parameters: {}",
        model.number_of_parameters()
    );

    let mut training = Training::new(batch_sampler, &model);
    for _step in 1..=10_000 {
        training.step();
    }
    training.estimate_loss();

    print_generated_text(&model, &tokenizer, 100);

    Ok(())
}

fn print_generated_text(model: &Model, tokenizer: &Tokenizer, num_tokens: usize) {
    let sampler = LogitsProcessor::from_sampling(42, Sampling::All { temperature: 1.0 });
    let prompt = tokenizer.encode("\n");
    let generator =
        GenerateIter::new(model, sampler, prompt).map(|token| tokenizer.decode_single(token));
    for token in generator.take(num_tokens) {
        print!("{token}");
    }
}
