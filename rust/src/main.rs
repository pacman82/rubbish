mod data;
mod device;
mod generate;
mod model;
mod tokenizer;
mod train;

use std::fs;

use anyhow::Context;
use candle_nn::{AdamW, ModuleT, Optimizer};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use rand::rng;

use self::{
    data::BatchSampler, device::choose_device, generate::GenerateIter, model::Model,
    tokenizer::Tokenizer, train::calculate_loss,
};

const BATCH_SIZE: usize = 32;
const CONTEXT_SIZE: usize = 8;

fn main() -> anyhow::Result<()> {
    let device = choose_device();

    let text = fs::read_to_string("../input.txt").context("Error opening input text.")?;
    let tokenizer = Tokenizer::from_text(&text);
    let tokenized = tokenizer.encode(&text);

    let mut batch_sampler = BatchSampler::from_tokenized(tokenized, &device);

    // Sample a training and a validation batch and print their decoded versions
    let rng = &mut rng();

    let model = Model::new(device.clone(), tokenizer.vocab_size());
    eprintln!(
        "Number of trainable parameters: {}",
        model.number_of_parameters()
    );

    let learning_rate = 1e-3;
    let trainable_params = model.all_vars();
    let mut optimizer = AdamW::new_lr(trainable_params, learning_rate).unwrap();

    for step in 1..=1_000 {
        let batch = batch_sampler.sample_training_batch(BATCH_SIZE, CONTEXT_SIZE, rng);
        let logits = model.forward_t(&batch.context, true).unwrap();
        let loss = calculate_loss(&logits, &batch.target);
        if step % 100 == 0 {
            println!("step {step}, loss: {}", loss.to_scalar::<f32>().unwrap());
        }
        optimizer.backward_step(&loss).unwrap();
    }

    println!("Generated text:");
    let sampler = LogitsProcessor::from_sampling(42, Sampling::All { temperature: 1.0 });
    let prompt = tokenizer.encode("\n");
    let generator =
        GenerateIter::new(&model, sampler, prompt).map(|token| tokenizer.decode(&[token]));
    for token in generator.take(100) {
        print!("{token}");
    }

    Ok(())
}
