mod data;
mod device;
mod generate;
mod model;
mod tokenizer;

use std::fs;

use anyhow::Context;
use candle_core::Tensor;
use candle_nn::{ModuleT, loss::cross_entropy};
use rand::rng;

use self::{
    data::BatchSampler, device::choose_device, generate::generate, model::Model,
    tokenizer::Tokenizer,
};

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

    let model = Model::new(&device, tokenizer.vocab_size());
    let logits = model.forward_t(&training_batch.context, false).unwrap();
    let loss = loss(&logits, &training_batch.target);
    println!("{:?}", logits.shape());
    println!("loss: {loss}");

    let mut text = vec![0];
    generate(&model, &mut text, &device, 100);

    let generated = tokenizer.decode(&text);
    println!("Generated text: {generated}");

    Ok(())
}

fn loss(logits: &Tensor, targets: &Tensor) -> f32 {
    // B, T, C => B * T, C
    let (b, t, c) = logits.dims3().unwrap();
    let logits = logits.reshape((b * t, c)).unwrap();
    let targets = targets.reshape((b * t,)).unwrap();
    let loss = cross_entropy(&logits, &targets).unwrap();
    loss.to_scalar().unwrap()
}
