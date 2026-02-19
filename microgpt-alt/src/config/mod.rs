//! Configuration for model, training, inference, and paths.
//!
//! Load from environment via [`from_env`] and validate with [`Config::validate`].

mod builder;

use std::path::PathBuf;

pub use builder::from_env;

/// Central configuration for the microgpt-alt pipeline.
///
/// Holds model dimensions, training and inference parameters, and paths.
/// Use [`from_env`] to build from environment variables and [`Config::validate`] before use.
#[derive(Clone, Debug)]
pub struct Config {
    /// Seed for RNG (reproducibility).
    pub seed: u64,
    /// Path to input corpus (one document per line).
    pub input_path: PathBuf,
    /// Path to save/load checkpoint.
    pub checkpoint_path: PathBuf,

    /// Embedding dimension (must be divisible by `n_head`).
    pub n_embed: usize,
    /// Number of attention heads.
    pub n_head: usize,
    /// Number of transformer layers.
    pub n_layer: usize,
    /// Maximum context length (tokens).
    pub block_size: usize,

    /// Weight init standard deviation.
    pub init_std: f64,
    /// RMSNorm epsilon.
    pub rmsnorm_eps: f64,

    /// Adam learning rate.
    pub learning_rate: f64,
    /// Adam beta1.
    pub beta1: f64,
    /// Adam beta2.
    pub beta2: f64,
    /// Adam epsilon.
    pub epsilon: f64,
    /// Gradient clipping (max norm; 0 = disabled).
    pub grad_clip: f64,

    /// Number of training steps.
    pub num_steps: usize,
    /// Log loss every this many steps.
    pub loss_log_every: usize,

    /// Sampling temperature (0 < T <= 1).
    pub temperature: f64,
    /// Number of samples to generate after training.
    pub sample_size: usize,
}

impl Config {
    /// Returns default configuration (suitable for tests and fallbacks).
    #[must_use]
    pub fn default_config() -> Self {
        Self {
            seed: 42,
            input_path: PathBuf::from("data/input.txt"),
            checkpoint_path: PathBuf::from("microgpt_alt.ckpt"),
            n_embed: 16,
            n_head: 4,
            n_layer: 1,
            block_size: 16,
            init_std: 0.08,
            rmsnorm_eps: 1e-5,
            learning_rate: 0.01,
            beta1: 0.85,
            beta2: 0.99,
            epsilon: 1e-8,
            grad_clip: 1.0,
            num_steps: 1000,
            loss_log_every: 10,
            temperature: 0.5,
            sample_size: 20,
        }
    }

    /// Validates configuration. Returns `Ok(())` if valid, or an error message.
    ///
    /// Ensures: `n_embed` divisible by `n_head`, `block_size > 0`, and other sanity checks.
    pub fn validate(&self) -> Result<(), String> {
        if self.n_head == 0 {
            return Err("n_head must be greater than 0".to_string());
        }
        if !self.n_embed.is_multiple_of(self.n_head) {
            return Err(format!(
                "n_embed ({}) must be divisible by n_head ({})",
                self.n_embed, self.n_head
            ));
        }
        if self.block_size == 0 {
            return Err("block_size must be greater than 0".to_string());
        }
        if self.n_layer == 0 {
            return Err("n_layer must be greater than 0".to_string());
        }
        if self.temperature <= 0.0 || self.temperature > 1.0 {
            return Err("temperature must be in (0, 1]".to_string());
        }
        Ok(())
    }

    /// Head dimension (n_embed / n_head).
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.n_embed / self.n_head
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let cfg = Config::default_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_rejects_n_embed_not_divisible_by_n_head() {
        let mut cfg = Config::default_config();
        cfg.n_embed = 15;
        cfg.n_head = 4;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_accepts_n_embed_divisible_by_n_head() {
        let mut cfg = Config::default_config();
        cfg.n_embed = 16;
        cfg.n_head = 4;
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_block_size() {
        let mut cfg = Config::default_config();
        cfg.block_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_n_head() {
        let mut cfg = Config::default_config();
        cfg.n_head = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_temperature_out_of_range() {
        let mut cfg = Config::default_config();
        cfg.temperature = 0.0;
        assert!(cfg.validate().is_err());
        cfg.temperature = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn from_env_falls_back_to_defaults() {
        let cfg = from_env();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.head_dim(), cfg.n_embed / cfg.n_head);
    }

    #[test]
    fn from_env_overrides_with_env_vars() {
        std::env::set_var(format!("{}N_EMBED", super::builder::ENV_PREFIX), "32");
        std::env::set_var(format!("{}N_HEAD", super::builder::ENV_PREFIX), "4");
        let cfg = from_env();
        assert_eq!(cfg.n_embed, 32);
        assert_eq!(cfg.n_head, 4);
        std::env::remove_var(format!("{}N_EMBED", super::builder::ENV_PREFIX));
        std::env::remove_var(format!("{}N_HEAD", super::builder::ENV_PREFIX));
    }
}
