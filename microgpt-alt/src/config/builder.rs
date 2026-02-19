//! Build [`Config`] from environment variables.
//!
//! Uses [`env_string`] and [`env_parsed`] to read env vars with a single place for key names
//! (see [`crate::config::constants`]) and typed errors ([`ConfigError`]). For file-based or
//! layered config (e.g. config file + env overrides), consider using the `config` crate in a
//! future iteration.

use std::path::PathBuf;

use super::constants::{
    ENV_BETA1, ENV_BETA2, ENV_BLOCK_SIZE, ENV_CHECKPOINT_PATH, ENV_EPSILON, ENV_GRAD_CLIP,
    ENV_INIT_STD, ENV_INPUT_PATH, ENV_LEARNING_RATE, ENV_LOSS_LOG_EVERY, ENV_NUM_STEPS,
    ENV_N_EMBED, ENV_N_HEAD, ENV_N_LAYER, ENV_PREFIX, ENV_RMSNORM_EPS, ENV_SAMPLE_SIZE, ENV_SEED,
    ENV_TEMPERATURE,
};
use super::Config;
use super::ConfigError;

/// Returns the full environment variable key for a given suffix (e.g. `SEED` → `MICROGPT_ALT_SEED`).
#[must_use]
pub fn env_key(suffix: &str) -> String {
    format!("{ENV_PREFIX}{suffix}")
}

/// Reads an environment variable as a string.
///
/// Returns `Some(value)` if the variable is set and valid UTF-8, `None` if unset.
/// Returns `Err(ConfigError::EnvVar)` if the variable is set but invalid (e.g. not Unicode).
pub fn env_string(key: &str) -> Result<Option<String>, ConfigError> {
    match std::env::var(key) {
        Ok(s) => Ok(Some(s)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(ConfigError::EnvVar {
            key: key.to_string(),
            message: e.to_string(),
        }),
    }
}

/// Reads an environment variable and parses it into type `T`.
///
/// Returns `Ok(Some(value))` if set and parse succeeds, `Ok(None)` if unset, and
/// `Err(ConfigError::Parse)` if set but parsing fails (e.g. `SEED=abc` for `u64`).
pub fn env_parsed<T>(key: &str) -> Result<Option<T>, ConfigError>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let s = match std::env::var(key) {
        Ok(s) => s,
        Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(e) => {
            return Err(ConfigError::EnvVar {
                key: key.to_string(),
                message: e.to_string(),
            })
        }
    };
    match s.parse() {
        Ok(t) => Ok(Some(t)),
        Err(e) => Err(ConfigError::Parse {
            key: key.to_string(),
            value: s,
            message: e.to_string(),
        }),
    }
}

/// Builds [`Config`] from environment variables, falling back to [`Config::default`] for unset values.
///
/// Returns [`ConfigError`] if any *set* variable fails to parse (e.g. `MICROGPT_ALT_SEED=abc`).
/// Environment variable names are defined in the config `constants` submodule.
pub fn from_env() -> Result<Config, ConfigError> {
    let default = Config::default();

    let seed = env_parsed::<u64>(&env_key(ENV_SEED))?.unwrap_or(default.seed);
    let input_path = env_string(&env_key(ENV_INPUT_PATH))?
        .map(PathBuf::from)
        .unwrap_or_else(|| default.input_path.clone());
    let checkpoint_path = env_string(&env_key(ENV_CHECKPOINT_PATH))?
        .map(PathBuf::from)
        .unwrap_or_else(|| default.checkpoint_path.clone());
    let n_embed = env_parsed::<usize>(&env_key(ENV_N_EMBED))?.unwrap_or(default.n_embed);
    let n_head = env_parsed::<usize>(&env_key(ENV_N_HEAD))?.unwrap_or(default.n_head);
    let n_layer = env_parsed::<usize>(&env_key(ENV_N_LAYER))?.unwrap_or(default.n_layer);
    let block_size = env_parsed::<usize>(&env_key(ENV_BLOCK_SIZE))?.unwrap_or(default.block_size);
    let init_std = env_parsed::<f64>(&env_key(ENV_INIT_STD))?.unwrap_or(default.init_std);
    let rmsnorm_eps = env_parsed::<f64>(&env_key(ENV_RMSNORM_EPS))?.unwrap_or(default.rmsnorm_eps);
    let learning_rate =
        env_parsed::<f64>(&env_key(ENV_LEARNING_RATE))?.unwrap_or(default.learning_rate);
    let beta1 = env_parsed::<f64>(&env_key(ENV_BETA1))?.unwrap_or(default.beta1);
    let beta2 = env_parsed::<f64>(&env_key(ENV_BETA2))?.unwrap_or(default.beta2);
    let epsilon = env_parsed::<f64>(&env_key(ENV_EPSILON))?.unwrap_or(default.epsilon);
    let grad_clip = env_parsed::<f64>(&env_key(ENV_GRAD_CLIP))?.unwrap_or(default.grad_clip);
    let num_steps = env_parsed::<usize>(&env_key(ENV_NUM_STEPS))?.unwrap_or(default.num_steps);
    let loss_log_every =
        env_parsed::<usize>(&env_key(ENV_LOSS_LOG_EVERY))?.unwrap_or(default.loss_log_every);
    let temperature = env_parsed::<f64>(&env_key(ENV_TEMPERATURE))?.unwrap_or(default.temperature);
    let sample_size =
        env_parsed::<usize>(&env_key(ENV_SAMPLE_SIZE))?.unwrap_or(default.sample_size);

    Ok(Config {
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
    })
}
