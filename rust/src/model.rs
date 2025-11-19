use candle_core::{Device, Tensor, Var};
use candle_nn::{Embedding, Linear, ModuleT, VarBuilder, VarMap};

use crate::{CONTEXT_SIZE, train::Trainable};

const EMBEDDING_DIMENSION: usize = 32;

pub struct Model {
    /// Maps token ids to embedding vector
    token_embedding: Embedding,
    /// Maps token position to embedding vector
    positional_embedding: Embedding,
    /// Maps back to vocab size
    lm_head: Linear,
    var_map: VarMap,
}

impl Model {
    pub fn new(device: Device, vocab_size: usize) -> Self {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);
        let token_embedding = candle_nn::embedding(
            vocab_size,
            EMBEDDING_DIMENSION,
            vb.pp("token_embedding_table"),
        )
        .unwrap();
        let positional_embedding = candle_nn::embedding(
            CONTEXT_SIZE,
            EMBEDDING_DIMENSION,
            vb.pp("positional_embedding_table"),
        )
        .unwrap();
        let lm_head = candle_nn::linear(EMBEDDING_DIMENSION, vocab_size, vb.pp("lm_head")).unwrap();
        Self {
            token_embedding,
            positional_embedding,
            lm_head,
            var_map,
        }
    }

    pub fn number_of_parameters(&self) -> usize {
        self.var_map
            .all_vars()
            .iter()
            .map(|var| var.elem_count())
            .sum::<usize>()
    }
}

impl ModuleT for Model {
    fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        // xs in batch, time (context) and its elements range from 0 to vocab size
        let (_b, t) = xs.dims2().unwrap();
        // returns batch, time, embedding_dimension
        let token_embeddings = self.token_embedding.forward_t(xs, train).unwrap();
        // Position would be a 1 dimensional vector in the time dimension
        // let position = Tensor::arange::<u32>(0, t as u32, xs.device()).unwrap();
        // let positional_embeddings = self
        //     .positional_embedding
        //     .forward_t(&position, train)
        //     .unwrap();
        // let x = (token_embeddings + positional_embeddings).unwrap();
        // We use a linear layer to map the channel back to the vocabulary size.
        let logits = self.lm_head.forward_t(&token_embeddings, train).unwrap();
        Ok(logits)
    }
}

/// Models are loaded to a device (likely either Cuda or CPU). This trait allows us to learn the
/// device and declare appropriate tensors.
pub trait DeviceAffine {
    fn device(&self) -> Device;
}

impl DeviceAffine for Model {
    fn device(&self) -> Device {
        self.token_embedding.embeddings().device().clone()
    }
}

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

impl Trainable for Model {
    fn all_vars(&self) -> Vec<Var> {
        self.var_map.all_vars()
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Tensor;
    use candle_nn::ModuleT;

    use crate::{
        device::choose_device,
        model::{DeviceAffine, PredictLogits},
    };

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
