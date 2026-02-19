//! Build Config from environment variables.

use std::path::PathBuf;

use super::Config;

/// Environment variable prefix for microgpt-alt (e.g. `MICROGPT_ALT_N_EMBED`).
pub(crate) const ENV_PREFIX: &str = "MICROGPT_ALT_";

/// Builds [`Config`] from environment variables, falling back to [`Config::default_config`] for unset values.
///
/// Environment variables (all optional):  
/// `MICROGPT_ALT_SEED`, `MICROGPT_ALT_INPUT_PATH`, `MICROGPT_ALT_CHECKPOINT_PATH`,  
/// `MICROGPT_ALT_N_EMBED`, `MICROGPT_ALT_N_HEAD`, `MICROGPT_ALT_N_LAYER`, `MICROGPT_ALT_BLOCK_SIZE`,  
/// `MICROGPT_ALT_INIT_STD`, `MICROGPT_ALT_RMSNORM_EPS`,  
/// `MICROGPT_ALT_LEARNING_RATE`, `MICROGPT_ALT_BETA1`, `MICROGPT_ALT_BETA2`, `MICROGPT_ALT_EPSILON`, `MICROGPT_ALT_GRAD_CLIP`,  
/// `MICROGPT_ALT_NUM_STEPS`, `MICROGPT_ALT_LOSS_LOG_EVERY`,  
/// `MICROGPT_ALT_TEMPERATURE`, `MICROGPT_ALT_SAMPLE_SIZE`.
#[must_use]
pub fn from_env() -> Config {
    let default = Config::default_config();
    let seed = std::env::var(concat_env("SEED"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default.seed);
    let input_path = std::env::var(concat_env("INPUT_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| default.input_path.clone());
    let checkpoint_path = std::env::var(concat_env("CHECKPOINT_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| default.checkpoint_path.clone());
    let n_embed = std::env::var(concat_env("N_EMBED"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default.n_embed);
    let n_head = std::env::var(concat_env("N_HEAD"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default.n_head);
    let n_layer = std::env::var(concat_env("N_LAYER"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default.n_layer);
    let block_size = std::env::var(concat_env("BLOCK_SIZE"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default.block_size);
    let init_std = std::env::var(concat_env("INIT_STD"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default.init_std);
    let rmsnorm_eps = std::env::var(concat_env("RMSNORM_EPS"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default.rmsnorm_eps);
    let learning_rate = std::env::var(concat_env("LEARNING_RATE"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default.learning_rate);
    let beta1 = std::env::var(concat_env("BETA1"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default.beta1);
    let beta2 = std::env::var(concat_env("BETA2"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default.beta2);
    let epsilon = std::env::var(concat_env("EPSILON"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default.epsilon);
    let grad_clip = std::env::var(concat_env("GRAD_CLIP"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default.grad_clip);
    let num_steps = std::env::var(concat_env("NUM_STEPS"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default.num_steps);
    let loss_log_every = std::env::var(concat_env("LOSS_LOG_EVERY"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default.loss_log_every);
    let temperature = std::env::var(concat_env("TEMPERATURE"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default.temperature);
    let sample_size = std::env::var(concat_env("SAMPLE_SIZE"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default.sample_size);

    Config {
        seed,
        input_path,
        checkpoint_path,
        n_embed,
        n_head,
        n_layer,
        block_size,
        init_std,
        rmsnorm_eps,
        learning_rate,
        beta1,
        beta2,
        epsilon,
        grad_clip,
        num_steps,
        loss_log_every,
        temperature,
        sample_size,
    }
}

fn concat_env(suffix: &str) -> String {
    format!("{ENV_PREFIX}{suffix}")
}
