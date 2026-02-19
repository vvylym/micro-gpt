//! Errors produced when loading or validating input data.
//!
//! All errors from the data module use [`Error`]. A future crate-level error type
//! can wrap this type for unified handling.

use std::fmt;

/// Errors produced by the data loading module.
///
/// # Variants
///
/// - **Io**: Failed to read the file (e.g. file not found, permission denied, invalid UTF-8).
///   *When*: Opening or reading the path in [`load_lines`](super::load_lines).
///   *Recovery*: Ensure the path exists, is readable, and contains valid UTF-8; check the source for details.
///
/// - **EmptyFile**: The file was read successfully but contains no lines (empty or only whitespace that yields no lines).
///   *When*: After reading the file in [`load_lines`](super::load_lines) and splitting into lines.
///   *Recovery*: Provide a non-empty input file with at least one line of content.
#[derive(Debug)]
pub enum Error {
    /// I/O error while reading the input file.
    Io(std::io::Error),

    /// The input file is empty (no lines).
    EmptyFile,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "data io: {e}"),
            Error::EmptyFile => write!(f, "data: input file is empty"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::EmptyFile => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
