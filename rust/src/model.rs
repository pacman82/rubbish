use candle_core::{Device, Tensor};
use candle_nn::{Embedding, ModuleT, VarBuilder, VarMap};

pub struct Model {
    embedding: Embedding,
}

impl Model {
    pub fn new(device: &Device, vocab_size: usize) -> Self {
        let embedding_dimension = vocab_size;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, candle_core::DType::F32, &device);

        let embedding = candle_nn::embedding(vocab_size, embedding_dimension, vb).unwrap();
        Self { embedding }
    }
}

impl ModuleT for Model {
    fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        // xs in batch, time (context), channel (vocab size)
        // returns batch, time, embedding_dimension
        self.embedding.forward_t(xs, train)
    }
}
