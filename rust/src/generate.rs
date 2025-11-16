use candle_core::{Device, Tensor};
use candle_nn::ModuleT;
use candle_transformers::generation::{LogitsProcessor, Sampling};

pub fn generate(
    model: &impl ModuleT,
    input_ids: &mut Vec<u32>,
    device: &Device,
    max_new_tokens: usize,
) {
    let seed = 42;
    let sampling = Sampling::All { temperature: 1.0 };
    let mut sampling = LogitsProcessor::from_sampling(seed, sampling);
    for _ in 0..max_new_tokens {
        let input_tensor = Tensor::from_slice(input_ids, &[1, input_ids.len()], device).unwrap();
        // logits are Batch, Time, Vocab_size
        let logits = model.forward_t(&input_tensor, false).unwrap();
        // We are only interessted in the predicted tokens of the last time step
        let (_batch, time_size, _vocab) = logits.dims3().unwrap();
        let last_logits = logits.get_on_dim(1, time_size - 1).unwrap();
        let last_logits = last_logits.squeeze(0).unwrap();

        let next = sampling.sample(&last_logits).unwrap();
        input_ids.push(next);
    }
}
