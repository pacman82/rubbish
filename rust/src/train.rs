use candle_core::Tensor;
use candle_nn::{AdamW, ModuleT, Optimizer, loss::cross_entropy};

use crate::{BATCH_SIZE, CONTEXT_SIZE, LEARNING_RATE, data::BatchSampler, model::Model};

pub struct Training<'a, B> {
    data: B,
    optimizer: AdamW,
    model: &'a Model,
}

impl<'a, B> Training<'a, B> {
    pub fn new(data: B, model: &'a Model) -> Self {
        let trainable_params = model.all_vars();
        let optimizer = AdamW::new_lr(trainable_params, LEARNING_RATE).unwrap();
        Training {
            data,
            optimizer,
            model,
        }
    }

    pub fn step(&mut self)
    where
        B: BatchSampler,
    {
        let batch = self.data.sample_training_batch(BATCH_SIZE, CONTEXT_SIZE);
        let logits = self.model.forward_t(&batch.context, true).unwrap();
        let loss = calculate_loss(&logits, &batch.target);
        self.optimizer.backward_step(&loss).unwrap();
    }
}

fn calculate_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    // B, T, C => B * T, C
    let (b, t, c) = logits.dims3().unwrap();
    let logits = logits.reshape((b * t, c)).unwrap();
    let targets = targets.reshape((b * t,)).unwrap();
    let loss = cross_entropy(&logits, &targets).unwrap();
    loss
}
