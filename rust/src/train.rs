use candle_core::{Tensor, Var};
use candle_nn::{AdamW, ModuleT, Optimizer, loss::cross_entropy};

use crate::{
    BATCH_SIZE, CONTEXT_SIZE, LEARNING_RATE,
    data::{BatchSampler, BatchSource},
};

pub struct Training<'a, B, M> {
    data: B,
    optimizer: AdamW,
    model: &'a M,
}

impl<'a, B, M> Training<'a, B, M>
where
    M: Trainable,
    B: BatchSampler,
{
    pub fn new(data: B, model: &'a M) -> Self {
        let trainable_params = model.all_vars();
        let optimizer = AdamW::new_lr(trainable_params, LEARNING_RATE).unwrap();
        Training {
            data,
            optimizer,
            model,
        }
    }

    pub fn step(&mut self) {
        let batch = self.data.sample_training_batch(BATCH_SIZE, CONTEXT_SIZE);
        let logits = self.model.forward_t(&batch.context, true).unwrap();
        let loss = calculate_loss(&logits, &batch.target);
        self.optimizer.backward_step(&loss).unwrap();
    }

    pub fn estimate_loss(&mut self) {
        let train_loss = self.estimate_loss_impl(BatchSource::Training);
        let validation_loss = self.estimate_loss_impl(BatchSource::Validation);

        println!(
            "Training loss: {}, Validation loss: {}",
            train_loss.to_vec0::<f32>().unwrap(),
            validation_loss.to_vec0::<f32>().unwrap()
        )
    }

    fn estimate_loss_impl(&mut self, source: BatchSource) -> Tensor {
        let mut losses = Vec::with_capacity(200);
        for _ in 0..200 {
            let batch = self.data.sample_batch(source, BATCH_SIZE, CONTEXT_SIZE);
            let logits = self.model.forward_t(&batch.context, false).unwrap();
            let loss = calculate_loss(&logits, &batch.target);
            losses.push(loss);
        }
        // Stack losses into a tensor
        let losses_tensor = Tensor::stack(&losses, 0).unwrap();
        // Take the mean over all losses (assuming mean() exists)
        let mean_loss = losses_tensor.mean(0).unwrap();
        mean_loss
    }
}

pub trait Trainable: ModuleT {
    fn all_vars(&self) -> Vec<Var>;
}

fn calculate_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    // B, T, C => B * T, C
    let (b, t, c) = logits.dims3().unwrap();
    let logits = logits.reshape((b * t, c)).unwrap();
    let targets = targets.reshape((b * t,)).unwrap();
    let loss = cross_entropy(&logits, &targets).unwrap();
    loss
}
