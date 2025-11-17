use candle_core::{Device, Tensor};
use rand::{
    Rng,
    distr::{Distribution, Uniform},
};

/// Samples training and validation batches
pub struct Data<R> {
    training_data: Tensor,
    validation_data: Tensor,
    // We reuse these buffers for composing batches to avoid repeated allocations
    context_batch_buffer: Vec<Tensor>,
    target_batch_buffer: Vec<Tensor>,
    rng: R,
}

impl<R> Data<R> {
    pub fn new(tokenized: Vec<u32>, device: &Device, rng: R) -> Self {
        let num_tokens = tokenized.len();
        eprintln!("Number of tokens in input data: {num_tokens}");
        let tokenized = Tensor::from_vec(tokenized, num_tokens, device).unwrap();
        let fraction_of_input_used_for_training = 0.9;
        eprintln!("Fraction of input used for training: {fraction_of_input_used_for_training}");
        let num_training_tokens =
            (fraction_of_input_used_for_training * num_tokens as f64).trunc() as usize;
        let num_validation_tokens = num_tokens - num_training_tokens;
        let training_data = tokenized.narrow(0, 0, num_training_tokens).unwrap();
        let validation_data = tokenized
            .narrow(0, num_training_tokens, num_validation_tokens)
            .unwrap();
        eprintln!("Number of training tokens: {num_training_tokens}");
        eprintln!("Number of validation tokens: {num_validation_tokens}");
        Data {
            training_data,
            validation_data,
            context_batch_buffer: Vec::new(),
            target_batch_buffer: Vec::new(),
            rng,
        }
    }
}

/// Sampling of training and validation batches
pub trait BatchSampler {
    fn sample_training_batch(&mut self, batch_size: usize, context_size: usize) -> SampleBatch;
    fn sample_validation_batch(&mut self, batch_size: usize, context_size: usize) -> SampleBatch;
}

impl<R> BatchSampler for Data<R>
where
    R: BatchStartIndices,
{
    fn sample_training_batch(&mut self, batch_size: usize, context_size: usize) -> SampleBatch {
        sample_batch(
            &self.training_data,
            batch_size,
            context_size,
            &mut self.rng,
            &mut self.context_batch_buffer,
            &mut self.target_batch_buffer,
        )
    }

    fn sample_validation_batch(&mut self, batch_size: usize, context_size: usize) -> SampleBatch {
        sample_batch(
            &self.validation_data,
            batch_size,
            context_size,
            &mut self.rng,
            &mut self.context_batch_buffer,
            &mut self.target_batch_buffer,
        )
    }
}

fn sample_batch(
    data: &Tensor,
    batch_size: usize,
    context_size: usize,
    rng: &mut impl BatchStartIndices,
    context_batch_buffer: &mut Vec<Tensor>,
    target_batch_buffer: &mut Vec<Tensor>,
) -> SampleBatch {
    // We need to be able to sample context size tokens and one additional target tokens
    assert!(
        data.dim(0).unwrap() >= context_size + 1,
        "Not enough data is available to suppor requested batch size"
    );

    context_batch_buffer.clear();
    target_batch_buffer.clear();

    let up = data.dim(0).unwrap() - context_size;
    for _ in 0..batch_size {
        let start_index = rng.next_batch_start_index(up);
        context_batch_buffer.push(data.narrow(0, start_index, context_size).unwrap());
        target_batch_buffer.push(data.narrow(0, start_index + 1, context_size).unwrap());
    }
    let context = Tensor::stack(context_batch_buffer.as_slice(), 0).unwrap();
    let target = Tensor::stack(target_batch_buffer.as_slice(), 0).unwrap();
    SampleBatch { context, target }
}

/// A batch sampled for training
pub struct SampleBatch {
    /// A tensor of shape [batch_size, context_size]. Each line in the batch dimension contains
    /// context_size contiguous tokens from the training data.
    pub context: Tensor,
    /// A tensor of shape [batch_size, context_size]. Each line in the batch dimension contains the
    /// token immediately following the context tokens, of the batch with the same index of context.
    pub target: Tensor,
}

/// Provide start indices for training or validation batches. This trait is auto implemented for any
/// `Rng` as we intend to sample the batches randomly in training.
pub trait BatchStartIndices {
    /// # Parameters
    ///
    /// * `up`: Exclusive upper bound for the generated index.
    fn next_batch_start_index(&mut self, up: usize) -> usize;
}

impl<T> BatchStartIndices for T
where
    T: Rng,
{
    fn next_batch_start_index(&mut self, up: usize) -> usize {
        Uniform::new(0, up).unwrap().sample(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::device::choose_device;

    use super::*;

    #[test]
    fn sample_training_batch() {
        let device = choose_device();
        let batch_size = 3;
        let context_size = 2;
        let rng = CountingStartIndices::new();

        let mut data = Data::new((0..10).collect(), &device, rng);
        let batch = data.sample_training_batch(batch_size, context_size);
        assert_eq!(batch.context.dims(), &[3, 2]);
        assert_eq!(batch.target.dims(), &[3, 2]);

        assert_eq!(
            vec![vec![0, 1], vec![1, 2], vec![2, 3]],
            batch.context.to_vec2::<u32>().unwrap()
        );

        assert_eq!(
            vec![vec![1, 2], vec![2, 3], vec![3, 4]],
            batch.target.to_vec2::<u32>().unwrap()
        );
    }

    #[test]
    fn sample_validation_batch() {
        let device = choose_device();
        let batch_size = 3;
        let context_size = 2;
        let rng = CountingStartIndices::new();

        let mut data = Data::new((0..30).collect(), &device, rng);
        let batch = data.sample_validation_batch(batch_size, context_size);
        assert_eq!(batch.context.dims(), &[3, 2]);
        assert_eq!(batch.target.dims(), &[3, 2]);

        assert_eq!(
            vec![vec![27, 28], vec![27, 28], vec![27, 28]],
            batch.context.to_vec2::<u32>().unwrap()
        );

        assert_eq!(
            vec![vec![28, 29], vec![28, 29], vec![28, 29]],
            batch.target.to_vec2::<u32>().unwrap()
        );
    }

    /// Fake random start indices for testing purposes. Counts batch indices starting from 0.
    struct CountingStartIndices(usize);

    impl CountingStartIndices {
        fn new() -> Self {
            CountingStartIndices(0)
        }
    }

    impl BatchStartIndices for CountingStartIndices {
        fn next_batch_start_index(&mut self, up: usize) -> usize {
            let value = self.0 % up;
            self.0 += 1;
            value
        }
    }
}
