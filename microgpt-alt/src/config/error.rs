//! Configuration errors.
//!
//! All errors produced by the config module (validation and env loading) use [`ConfigError`].
//! Callers can match on variants to handle specific cases or use [`ConfigError::message`] for logging.

use std::fmt;

/// Errors produced when building or validating configuration.
///
/// # Variants
///
/// - **Validation**: Configuration values are inconsistent or out of range (e.g. `n_embed` not divisible by `n_head`).
///   *When*: After building a `Config` and calling `validate()`.
///   *Recovery*: Fix the values (via env or code) so that `validate()` passes; see the error message for the rule that failed.
///
/// - **EnvVar**: An environment variable could not be read (e.g. invalid Unicode).
///   *When*: When using env helpers to read a key.
///   *Recovery*: Ensure the variable contains valid Unicode, or use a different key.
///
/// - **Parse**: An environment variable was set but could not be parsed into the expected type (e.g. `SEED=abc`).
///   *When*: When using `env_parsed` and the value is not valid for the target type.
///   *Recovery*: Set the env var to a valid value or unset it to use the default; the error message indicates the key and invalid value.
#[derive(Debug, Clone)]
pub enum ConfigError {
    /// Configuration validation failed (e.g. invalid dimensions or ranges).
    Validation(String),

    /// Failed to read an environment variable (e.g. key not present or invalid Unicode).
    EnvVar {
        /// The full environment variable name that was read.
        key: String,
        /// Optional underlying cause (e.g. NotUnicode).
        message: String,
    },

    /// Environment variable was set but could not be parsed into the expected type.
    Parse {
        /// The full environment variable name.
        key: String,
        /// The raw value that failed to parse.
        value: String,
        /// Human-readable parse reason (e.g. "invalid u64").
        message: String,
    },
}

impl ConfigError {
    /// Returns a short message suitable for logging or user display.
    #[must_use]
    pub fn message(&self) -> &str {
        match self {
            ConfigError::Validation(m) => m,
            ConfigError::EnvVar { message, .. } => message,
            ConfigError::Parse { message, .. } => message,
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Validation(m) => write!(f, "config validation: {m}"),
            ConfigError::EnvVar { key, message } => write!(f, "env var {key}: {message}"),
            ConfigError::Parse {
                key,
                value,
                message,
            } => {
                write!(f, "env var {key}={value:?}: {message}")
            }
        }
    }
}

impl std::error::Error for ConfigError {}
