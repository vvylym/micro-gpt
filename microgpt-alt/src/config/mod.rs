//! Configuration for model, training, inference, and paths.
//!
//! Load from environment via [`from_env`] and validate with [`Config::validate`].
//! Default values and env key names are centralized in the `constants` submodule.

mod builder;
mod constants;
mod error;

use std::path::PathBuf;

use constants::{
    DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_BLOCK_SIZE, DEFAULT_CHECKPOINT_PATH, DEFAULT_EPSILON,
    DEFAULT_GRAD_CLIP, DEFAULT_INIT_STD, DEFAULT_INPUT_PATH, DEFAULT_LEARNING_RATE,
    DEFAULT_LOSS_LOG_EVERY, DEFAULT_NUM_STEPS, DEFAULT_N_EMBED, DEFAULT_N_HEAD, DEFAULT_N_LAYER,
    DEFAULT_RMSNORM_EPS, DEFAULT_SAMPLE_SIZE, DEFAULT_SEED, DEFAULT_TEMPERATURE,
};

pub use builder::{env_key, env_parsed, env_string, from_env};
pub use error::ConfigError;

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

impl Default for Config {
    /// Returns default configuration (suitable for tests and fallbacks).
    fn default() -> Self {
        Self {
            seed: DEFAULT_SEED,
            input_path: PathBuf::from(DEFAULT_INPUT_PATH),
            checkpoint_path: PathBuf::from(DEFAULT_CHECKPOINT_PATH),
            n_embed: DEFAULT_N_EMBED,
            n_head: DEFAULT_N_HEAD,
            n_layer: DEFAULT_N_LAYER,
            block_size: DEFAULT_BLOCK_SIZE,
            init_std: DEFAULT_INIT_STD,
            rmsnorm_eps: DEFAULT_RMSNORM_EPS,
            learning_rate: DEFAULT_LEARNING_RATE,
            beta1: DEFAULT_BETA1,
            beta2: DEFAULT_BETA2,
            epsilon: DEFAULT_EPSILON,
            grad_clip: DEFAULT_GRAD_CLIP,
            num_steps: DEFAULT_NUM_STEPS,
            loss_log_every: DEFAULT_LOSS_LOG_EVERY,
            temperature: DEFAULT_TEMPERATURE,
            sample_size: DEFAULT_SAMPLE_SIZE,
        }
    }
}

impl Config {
    /// Validates configuration. Returns `Ok(())` if valid, or a [`ConfigError`].
    ///
    /// Ensures: `n_embed` divisible by `n_head`, `block_size > 0`, and other sanity checks.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.n_head == 0 {
            return Err(ConfigError::Validation(
                "n_head must be greater than 0".to_string(),
            ));
        }
        if !self.n_embed.is_multiple_of(self.n_head) {
            return Err(ConfigError::Validation(format!(
                "n_embed ({}) must be divisible by n_head ({})",
                self.n_embed, self.n_head
            )));
        }
        if self.block_size == 0 {
            return Err(ConfigError::Validation(
                "block_size must be greater than 0".to_string(),
            ));
        }
        if self.n_layer == 0 {
            return Err(ConfigError::Validation(
                "n_layer must be greater than 0".to_string(),
            ));
        }
        if self.temperature <= 0.0 || self.temperature > 1.0 {
            return Err(ConfigError::Validation(
                "temperature must be in (0, 1]".to_string(),
            ));
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
    use super::constants::{ENV_N_EMBED, ENV_N_HEAD, ENV_SEED};
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let cfg = Config::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_rejects_n_embed_not_divisible_by_n_head() {
        let cfg = Config {
            n_embed: 15,
            n_head: 4,
            ..Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_accepts_n_embed_divisible_by_n_head() {
        let cfg = Config {
            n_embed: 16,
            n_head: 4,
            ..Config::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_block_size() {
        let cfg = Config {
            block_size: 0,
            ..Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_n_head() {
        let cfg = Config {
            n_head: 0,
            ..Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_temperature_out_of_range() {
        let cfg_zero = Config {
            temperature: 0.0,
            ..Config::default()
        };
        assert!(cfg_zero.validate().is_err());
        let cfg_high = Config {
            temperature: 1.5,
            ..Config::default()
        };
        assert!(cfg_high.validate().is_err());
    }

    /// Lock so env tests don't run in parallel and pollute each other.
    static CONFIG_ENV_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();

    #[test]
    fn from_env_falls_back_to_defaults() {
        let _g = CONFIG_ENV_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap();
        std::env::remove_var(env_key(ENV_N_EMBED));
        std::env::remove_var(env_key(ENV_SEED));
        let cfg = from_env().unwrap();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.head_dim(), cfg.n_embed / cfg.n_head);
    }

    #[test]
    fn from_env_overrides_with_env_vars() {
        let _g = CONFIG_ENV_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap();
        let key_n_embed = env_key(ENV_N_EMBED);
        let key_n_head = env_key(ENV_N_HEAD);
        std::env::set_var(&key_n_embed, "32");
        std::env::set_var(&key_n_head, "4");
        let cfg = from_env().unwrap();
        assert_eq!(cfg.n_embed, 32);
        assert_eq!(cfg.n_head, 4);
        std::env::remove_var(key_n_embed);
        std::env::remove_var(key_n_head);
    }

    #[test]
    fn from_env_returns_error_on_invalid_parse() {
        let _g = CONFIG_ENV_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap();
        let key = env_key(ENV_SEED);
        std::env::set_var(&key, "not_a_number");
        let res = from_env();
        std::env::remove_var(key);
        assert!(matches!(res, Err(ConfigError::Parse { .. })));
    }

    #[test]
    fn config_error_validation_display() {
        let e = ConfigError::Validation("n_head must be > 0".to_string());
        assert!(e.to_string().contains("config validation"));
        assert!(e.to_string().contains("n_head"));
        assert_eq!(e.message(), "n_head must be > 0");
    }

    #[test]
    fn config_error_parse_display() {
        let e = ConfigError::Parse {
            key: "MICROGPT_ALT_SEED".to_string(),
            value: "abc".to_string(),
            message: "invalid digit".to_string(),
        };
        assert!(e.to_string().contains("MICROGPT_ALT_SEED"));
        assert!(e.to_string().contains("abc"));
        assert_eq!(e.message(), "invalid digit");
    }

    #[test]
    fn env_string_unset_returns_none() {
        let key = "MICROGPT_ALT_UNLIKELY_KEY_12345";
        assert_eq!(env_string(key).unwrap(), None);
    }

    #[test]
    fn env_parsed_unset_returns_none() {
        let key = "MICROGPT_ALT_UNLIKELY_KEY_67890";
        assert_eq!(env_parsed::<u64>(key).unwrap(), None);
    }

    #[test]
    fn env_parsed_invalid_returns_parse_error() {
        let _g = CONFIG_ENV_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap();
        let key = env_key(ENV_N_EMBED);
        std::env::set_var(&key, "not_usize");
        let res = env_parsed::<usize>(&key);
        std::env::remove_var(key);
        assert!(matches!(res, Err(ConfigError::Parse { .. })));
    }
}
