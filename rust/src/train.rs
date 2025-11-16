use candle_core::Tensor;
use candle_nn::loss::cross_entropy;

pub fn calculate_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    // B, T, C => B * T, C
    let (b, t, c) = logits.dims3().unwrap();
    let logits = logits.reshape((b * t, c)).unwrap();
    let targets = targets.reshape((b * t,)).unwrap();
    let loss = cross_entropy(&logits, &targets).unwrap();
    loss
}
