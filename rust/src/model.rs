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
        let (b, t) = xs.dims2().unwrap();
        // returns batch, time, embedding_dimension
        let token_embeddings = self.token_embedding.forward_t(xs, train).unwrap();
        // Position would be a 1 dimensional vector in the time dimension
        let position = Tensor::arange::<u32>(0, t as u32, xs.device()).unwrap();
        // Postional embeddings have dimension (time (context), channel (embedding size))
        let positional_embeddings = self
            .positional_embedding
            .forward_t(&position, train)
            .unwrap();
        // We need to add the batch dimension to the positional embeddings
        let positional_embeddings = positional_embeddings
            .broadcast_as((b, t, EMBEDDING_DIMENSION))
            .unwrap();
        let x = (token_embeddings + positional_embeddings).unwrap();
        // We use a linear layer to map the channel back to the vocabulary size.
        // I.e Batch, Time, Embedding size -> Batch, Time, Vocab Size
        let logits = self.lm_head.forward_t(&x, train).unwrap();
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

impl Trainable for Model {
    fn all_vars(&self) -> Vec<Var> {
        self.var_map.all_vars()
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::ModuleT;

    use crate::{device::choose_device, model::Model};

    // This tests drives the linear layer, mapping back to vocab size
    #[test]
    fn logits_have_dim_batch_time_vocab_size() {
        let device = choose_device();
        let vocab_size = 5;
        let time = 1;
        let batch_size = 2;
        let tensor = Tensor::zeros((batch_size, time), DType::U32, &device).unwrap();
        let model = Model::new(device, vocab_size);

        let logit = model.forward_t(&tensor, false).unwrap();

        assert_eq!((batch_size, time, vocab_size), logit.dims3().unwrap());
    }

    #[test]
    fn learning_test_broadcast() {
        let x = Tensor::new(&[1u32, 2, 3], &Device::Cpu).unwrap();
        let x = x.broadcast_as((2, 3)).unwrap();

        // [[1, 2, 3], [1, 2, 3]]
        eprintln!("{:?}", x.to_vec2::<u32>().unwrap());
    }
}
