mod data;
mod device;
mod tokenizer;

use std::fs;

use anyhow::Context;
use rand::rng;

use crate::device::choose_device;

use self::{data::BatchSampler, tokenizer::Tokenizer};

const BATCH_SIZE: usize = 4;
const CONTEXT_SIZE: usize = 8;

fn main() -> anyhow::Result<()> {
    let device = choose_device();

    let text = fs::read_to_string("../input.txt").context("Error opening input text.")?;
    let tokenizer = Tokenizer::from_text(&text);
    let tokenized = tokenizer.encode(&text);

    let mut batch_sampler = BatchSampler::from_tokenized(tokenized, &device);

    // Sample a training and a validation batch and print their decoded versions
    let rng = &mut rng();
    let training_batch = batch_sampler.sample_training_batch(BATCH_SIZE, CONTEXT_SIZE, rng);
    let validation_batch = batch_sampler.sample_validation_batch(BATCH_SIZE, CONTEXT_SIZE, rng);

    println!("Training batch:");
    for batch_index in 0..BATCH_SIZE {
        let data = training_batch
            .context
            .get(batch_index)
            .unwrap()
            .to_vec1()
            .unwrap();
        let context = tokenizer.decode(&data);
        let data = training_batch
            .target
            .get(batch_index)
            .unwrap()
            .to_vec1()
            .unwrap();
        let target = tokenizer.decode(&data);
        println!("context: {context}, target: {target}");
    }

    println!("Validation batch:");
    for batch_index in 0..BATCH_SIZE {
        let data = validation_batch
            .context
            .get(batch_index)
            .unwrap()
            .to_vec1()
            .unwrap();
        let context = tokenizer.decode(&data);
        let data = validation_batch
            .target
            .get(batch_index)
            .unwrap()
            .to_vec1()
            .unwrap();
        let target = tokenizer.decode(&data);
        println!("context: {context}, target: {target}");
    }

    Ok(())
}
