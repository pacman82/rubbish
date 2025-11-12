use anyhow::Context;
use candle_core::{Device, Tensor};
use rand::{
    Rng,
    distr::{Distribution, Uniform},
};

/// Data ready for training and validation.
pub struct Data {
    training_data: Tensor,
    validation_data: Tensor,
}

impl Data {
    pub fn from_tokenized(tokenized: Vec<u32>, device: &Device) -> anyhow::Result<Self> {
        let num_tokens = tokenized.len();
        eprintln!("Number of tokens in input data: {num_tokens}");
        let tokenized = Tensor::from_vec(tokenized, num_tokens, device)
            .context("Unable to create tokenized data on device")?;
        let fraction_of_input_used_for_training = 0.9;
        eprintln!("Fraction of input used for training: {fraction_of_input_used_for_training}");
        let num_training_tokens =
            (fraction_of_input_used_for_training * num_tokens as f64).trunc() as usize;
        let num_validation_tokens = num_tokens - num_training_tokens;
        let training_data = tokenized.narrow(0, 0, num_training_tokens)?;
        let validation_data = tokenized.narrow(0, num_training_tokens, num_tokens)?;
        eprintln!("Number of training tokens: {num_training_tokens}");
        eprintln!("Number of validation tokens: {num_validation_tokens}");
        Ok(Data {
            training_data,
            validation_data,
        })
    }

    pub fn sample_training_batch(
        &self,
        batch_size: usize,
        context_size: usize,
        rng: &mut impl Rng,
    ) -> SampleBatch {
        sample_batch(&self.training_data, batch_size, context_size, rng)
    }

    pub fn sample_validation_batch(
        &self,
        batch_size: usize,
        context_size: usize,
        rng: &mut impl Rng,
    ) -> SampleBatch {
        sample_batch(&self.validation_data, batch_size, context_size, rng)
    }
}

fn sample_batch(
    data: &Tensor,
    batch_size: usize,
    context_size: usize,
    rng: &mut impl Rng,
) -> SampleBatch {
    // We need to be able to sample context size tokens and one additional target tokens
    assert!(data.dim(0).unwrap() >= context_size + 1);

    let up = data.dim(0).unwrap() - context_size;
    let random_start_indices = Uniform::new(0, up).unwrap();
    let mut context_batch = Vec::with_capacity(batch_size);
    let mut target_batch = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        let start_index = random_start_indices.sample(rng);
        context_batch.push(data.narrow(0, start_index, context_size).unwrap());
        target_batch.push(data.narrow(0, start_index + context_size, 1).unwrap());
    }
    let context = Tensor::stack(context_batch.as_slice(), 0).unwrap();
    let target = Tensor::stack(target_batch.as_slice(), 0).unwrap();
    SampleBatch { context, target }
}

pub struct SampleBatch {
    context: Tensor,
    target: Tensor,
}

#[cfg(test)]
mod tests {}
