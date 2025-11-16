use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;

use crate::model::PredictLogits;

pub struct GenerateIter<'a, M, S> {
    model: &'a M,
    input_ids: Vec<u32>,
    sampler: S,
    device: Device,
}

impl<'a, M: PredictLogits, S: Sampler> GenerateIter<'a, M, S> {
    pub fn new(model: &'a M, sampler: S, input_ids: Vec<u32>) -> Self {
        Self {
            model,
            input_ids,
            sampler,
            device: model.device().clone(),
        }
    }
}

impl<M, S> Iterator for GenerateIter<'_, M, S>
where
    M: PredictLogits,
    S: Sampler,
{
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        let input_tensor =
            Tensor::from_slice(&self.input_ids, &[1, self.input_ids.len()], &self.device).unwrap();
        let logits = self.model.predict_logits(&input_tensor);
        let next = self.sampler.sample_single(&logits);
        self.input_ids.push(next);
        Some(next)
    }
}

pub trait Sampler {
    /// Samples a single token form a logits tensor with shape (vocab_size)
    fn sample_single(&mut self, logits: &Tensor) -> u32;
}

impl Sampler for LogitsProcessor {
    fn sample_single(&mut self, logits: &Tensor) -> u32 {
        self.sample(logits).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Tensor;
    use candle_transformers::generation::{LogitsProcessor, Sampling};

    use crate::{device::choose_device, generate::Sampler};

    #[test]
    fn sampler() {
        let device = choose_device();
        // Given a tensor of three logits, with the one at index 1 being the largest
        let logits = Tensor::from_slice(&[-1.5f32, 4., 0.0], 3, &device).unwrap();
        let sampling = Sampling::All { temperature: 1.0 };
        let mut sampler = LogitsProcessor::from_sampling(42, sampling);

        // When we sample from it
        let sampled = sampler.sample_single(&logits);

        // Then we should get sample the index 1 most of the time. We fix the seed here in the test.
        assert_eq!(sampled, 1);
    }
}
