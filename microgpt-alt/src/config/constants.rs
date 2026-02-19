//! Central place for all configuration constants.
//!
//! Default values and environment variable key names used by the config builder.
//! Keeping them here avoids magic numbers and repeated string literals across the config module.

/// Environment variable prefix for microgpt-alt (e.g. `MICROGPT_ALT_N_EMBED`).
pub(crate) const ENV_PREFIX: &str = "MICROGPT_ALT_";

// --- Env key suffixes (full key = ENV_PREFIX + suffix) ---

pub(crate) const ENV_SEED: &str = "SEED";
pub(crate) const ENV_INPUT_PATH: &str = "INPUT_PATH";
pub(crate) const ENV_CHECKPOINT_PATH: &str = "CHECKPOINT_PATH";
pub(crate) const ENV_N_EMBED: &str = "N_EMBED";
pub(crate) const ENV_N_HEAD: &str = "N_HEAD";
pub(crate) const ENV_N_LAYER: &str = "N_LAYER";
pub(crate) const ENV_BLOCK_SIZE: &str = "BLOCK_SIZE";
pub(crate) const ENV_INIT_STD: &str = "INIT_STD";
pub(crate) const ENV_RMSNORM_EPS: &str = "RMSNORM_EPS";
pub(crate) const ENV_LEARNING_RATE: &str = "LEARNING_RATE";
pub(crate) const ENV_BETA1: &str = "BETA1";
pub(crate) const ENV_BETA2: &str = "BETA2";
pub(crate) const ENV_EPSILON: &str = "EPSILON";
pub(crate) const ENV_GRAD_CLIP: &str = "GRAD_CLIP";
pub(crate) const ENV_NUM_STEPS: &str = "NUM_STEPS";
pub(crate) const ENV_LOSS_LOG_EVERY: &str = "LOSS_LOG_EVERY";
pub(crate) const ENV_TEMPERATURE: &str = "TEMPERATURE";
pub(crate) const ENV_SAMPLE_SIZE: &str = "SAMPLE_SIZE";

// --- Default values ---

pub(crate) const DEFAULT_SEED: u64 = 42;
pub(crate) const DEFAULT_INPUT_PATH: &str = "data/input.txt";
pub(crate) const DEFAULT_CHECKPOINT_PATH: &str = "microgpt_alt.ckpt";
pub(crate) const DEFAULT_N_EMBED: usize = 16;
pub(crate) const DEFAULT_N_HEAD: usize = 4;
pub(crate) const DEFAULT_N_LAYER: usize = 1;
pub(crate) const DEFAULT_BLOCK_SIZE: usize = 16;
pub(crate) const DEFAULT_INIT_STD: f64 = 0.08;
pub(crate) const DEFAULT_RMSNORM_EPS: f64 = 1e-5;
pub(crate) const DEFAULT_LEARNING_RATE: f64 = 0.01;
pub(crate) const DEFAULT_BETA1: f64 = 0.85;
pub(crate) const DEFAULT_BETA2: f64 = 0.99;
pub(crate) const DEFAULT_EPSILON: f64 = 1e-8;
pub(crate) const DEFAULT_GRAD_CLIP: f64 = 1.0;
pub(crate) const DEFAULT_NUM_STEPS: usize = 1000;
pub(crate) const DEFAULT_LOSS_LOG_EVERY: usize = 10;
pub(crate) const DEFAULT_TEMPERATURE: f64 = 0.5;
pub(crate) const DEFAULT_SAMPLE_SIZE: usize = 20;
