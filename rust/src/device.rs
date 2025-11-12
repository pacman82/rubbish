use candle_core::{
    Device,
    utils::{cuda_is_available, metal_is_available},
};

/// Pick a device to run, based on compile time flags and system capabilities. We prefer CUDA over
/// Metal and Metal over CPU.
pub fn choose_device() -> Device {
    // These operations can not fail, because we check for their availability before attempting to
    // create a device. The constructors do not allocate any system resources, they just initialize
    // an enum and check if support had been compiled in.

    let device = if cuda_is_available() {
        Device::new_cuda(0).unwrap()
    } else if metal_is_available() {
        Device::new_metal(0).unwrap()
    } else {
        Device::Cpu
    };
    eprintln!("Selected device: {device:?}");
    device
}
