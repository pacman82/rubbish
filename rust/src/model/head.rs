use candle_core::{D, Tensor};
use candle_nn::{Linear, ModuleT, VarBuilder, linear};

/// A self attention head. We use this so the net can learn what things mean in context
pub struct Head {
    /// Used to emit a key vector for each embeddings indicating what kind of information it is
    /// interessted in.
    key: Linear,
    /// Used to emit a query vector for each embedding indicating what kind of information it
    /// contains.
    query: Linear,
    values: Linear,
}

impl Head {
    pub fn new(embedding_dim: usize, head_size: usize, vb: VarBuilder) -> Self {
        let key = linear(embedding_dim, head_size, vb.pp("key")).unwrap();
        let query = linear(embedding_dim, head_size, vb.pp("query")).unwrap();
        let values = linear(embedding_dim, head_size, vb.pp("values")).unwrap();
        Head { key, query, values }
    }
}

impl ModuleT for Head {
    fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let key = self.key.forward_t(xs, train).unwrap(); // Dimension: Batch, Time, Head Size
        let query = self.query.forward_t(xs, train).unwrap();

        // We want to calculate the affinity. I.e. how much is key interessted in query. This is the
        // dot product between the key and the query vector. Of course here the vectors are part of
        // a matrix and also we need to take the batch dimension into account.

        // In order to multiply key with query we want to transpose the last two dimensions
        let key = key.transpose(D::Minus2, D::Minus1).unwrap(); // Key is now Batch, Head Size, Time
        let affinity = query.matmul(&key).unwrap(); // Dimension: Batch, Time, Time

        self.values.forward_t(xs, train)
    }
}
