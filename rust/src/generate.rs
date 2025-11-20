use candle_core::{Device, Tensor};
use candle_nn::ModuleT;
use candle_transformers::generation::LogitsProcessor;

use crate::{CONTEXT_SIZE, model::DeviceAffine};

/// Predict logits continuing the input sequence. Used for text generation. This is the step before
/// sampling.
pub trait PredictLogits: DeviceAffine {
    /// # Parameters
    ///
    /// Use the model to predict the logits for the next token for each batch in a sequence of
    /// inputs.
    ///
    /// * `xs`: Input tensor of shape (batch_size, sequence_length). The sequence length must not
    ///   exceed the context size of the model.
    ///
    /// # Returns
    ///
    /// Tensor of shape (batch_size, vocab_size) representing the logits for the next token.
    fn predict_logits(&self, xs: &Tensor) -> Tensor;

    /// Context size of the model. Used to limit the `sequence_length` passed into
    /// [`PredictLogits::predict_logits`]
    fn context_size(&self) -> usize;
}

impl<T> PredictLogits for T
where
    T: ModuleT + DeviceAffine,
{
    fn predict_logits(&self, xs: &Tensor) -> Tensor {
        // Forward pass through the model
        let logits = self.forward_t(xs, false).unwrap();

        // Get the last time step's embeddings
        let (_batch, time_size, _vocab) = logits.dims3().unwrap();
        let last_logits = logits.get_on_dim(1, time_size - 1).unwrap();
        let last_logits = last_logits.squeeze(0).unwrap();

        // For simplicity, we will just return the last embeddings as logits
        // In a real model, you would have a linear layer here to project to vocab size
        last_logits
    }

    fn context_size(&self) -> usize {
        CONTEXT_SIZE
    }
}

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
        let context_size = self.model.context_size();
        let input_tensor =
            Tensor::from_slice(&self.input_ids, &[1, self.input_ids.len()], &self.device).unwrap();
        let logits = self.model.predict_logits(&input_tensor);
        let next = self.sampler.sample_single(&logits);
        if self.input_ids.len() == context_size {
            // Input ids already have the length of context size, so we drop the first tokens from
            // the start and add the new token at the end.
            self.input_ids.rotate_left(1);
            *self.input_ids.last_mut().unwrap() = next;
        } else {
            // The input ids have not reached the length of the context size yet, so we just append.
            self.input_ids.push(next)
        }
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
    use std::{mem::replace, sync::Mutex};

    use candle_core::Tensor;
    use candle_nn::ModuleT;
    use candle_transformers::generation::{LogitsProcessor, Sampling};

    use crate::{
        device::choose_device,
        generate::{GenerateIter, PredictLogits, Sampler},
        model::DeviceAffine,
    };

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

    #[test]
    fn models_sees_only_inputs_which_are_smaller_or_equal_to_context_size() {
        struct ModelSpy {
            predict_logits_record: Mutex<Vec<Tensor>>,
        }

        impl ModelSpy {
            pub fn new() -> Self {
                Self {
                    predict_logits_record: Mutex::new(Vec::new()),
                }
            }

            pub fn take_recorded_predict_logits(&self) -> Vec<Tensor> {
                replace(&mut *self.predict_logits_record.lock().unwrap(), Vec::new())
            }
        }

        impl DeviceAffine for ModelSpy {
            fn device(&self) -> candle_core::Device {
                choose_device()
            }
        }

        impl PredictLogits for ModelSpy {
            fn predict_logits(&self, xs: &Tensor) -> Tensor {
                self.predict_logits_record.lock().unwrap().push(xs.clone());
                Tensor::new(&[[0f32; 5]], &self.device()).unwrap()
            }

            fn context_size(&self) -> usize {
                3
            }
        }

        let model_spy = ModelSpy::new();
        let input_ids = vec![];
        let mut it = GenerateIter::new(&model_spy, CountingSampler::new(5), input_ids);
        it.next().unwrap();
        it.next().unwrap();
        it.next().unwrap();
        it.next().unwrap();
        it.next().unwrap();
        it.next().unwrap();

        let record = model_spy.take_recorded_predict_logits();

        let token_ids = |i: usize| record[i].squeeze(0).unwrap().to_vec1::<u32>().unwrap();

        assert_eq!(&token_ids(0), &[0u32; 0]);
        assert_eq!(&token_ids(1), &[0u32]);
        assert_eq!(&token_ids(2), &[0u32, 1]);
        assert_eq!(&token_ids(3), &[0u32, 1, 2]);
        // We specified a context size of three, so we only see the last three input tokens
        assert_eq!(&token_ids(4), &[1u32, 2, 3]);
        assert_eq!(&token_ids(5), &[2u32, 3, 4]);
    }

    /// Fake sampler which samples tokens round robind iterating through the indices
    struct CountingSampler {
        current: u32,
        vocab_size: u32,
    }

    impl CountingSampler {
        pub fn new(vocab_size: u32) -> Self {
            Self {
                current: 0,
                vocab_size,
            }
        }
    }

    impl Sampler for CountingSampler {
        fn sample_single(&mut self, _logits: &Tensor) -> u32 {
            let next = self.current % self.vocab_size;
            self.current += 1;
            next
        }
    }

    #[test]
    fn predict_logits_returns_last_time_dimension_of_forward_pass() {
        // Given
        struct ModelStub;

        impl ModuleT for ModelStub {
            /// Returns
            /// [[0 1 2], [3 4 5], [6 7 8], [9 10 11]],
            /// [[12 13 14], [15 16 17], [18 19 20], [21 22 23]]
            fn forward_t(&self, _xs: &Tensor, _train: bool) -> candle_core::Result<Tensor> {
                // Batch size 2, time size 4, vocab size 3
                let vec = (0..(2 * 4 * 3)).map(|x| x as f32).collect::<Vec<f32>>();
                Tensor::from_slice(&vec, (2, 4, 3), &self.device())
            }
        }

        impl DeviceAffine for ModelStub {
            fn device(&self) -> candle_core::Device {
                choose_device()
            }
        }

        // When
        let dummy_input = &Tensor::from_vec(vec![0u32, 0u32], 1, &choose_device()).unwrap();
        let logits = ModelStub.predict_logits(dummy_input);

        // Then
        let logits_first_batch = logits.get_on_dim(0, 0).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!([9f32, 10., 11.].as_slice(), logits_first_batch);

        let logits_second_batch = logits.get_on_dim(0, 1).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!([21f32, 22., 23.].as_slice(), logits_second_batch);
    }
}
