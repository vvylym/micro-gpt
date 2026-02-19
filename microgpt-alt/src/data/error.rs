//! Errors produced when loading or validating input data.
//!
//! All errors from the data module use [`DataError`]. A future crate-level error type
//! can wrap this type for unified handling.

use std::fmt;

/// Errors produced by the data loading module.
///
/// # Variants
///
/// - **Io**: Failed to read the file (e.g. file not found, permission denied, invalid UTF-8).
///   *When*: Opening or reading the path in [`PathLoader`](super::PathLoader) or [`load_from_path`](super::load_from_path).
///   *Recovery*: Ensure the path exists, is readable, and contains valid UTF-8; check the source for details.
///
/// - **EmptyFile**: The file was read successfully but yields no data (no lines, or all lines empty after trim).
///   *When*: After reading and parsing in [`DataLoader::load`](super::DataLoader::load) when the resulting [`Data`](super::Data) would be empty.
///   *Recovery*: Provide a non-empty input file with at least one non-empty line.
///
/// - **EmptyDataItem**: A line was empty after trimming (invalid input).
///   *When*: When building a [`DataItem`](super::DataItem) from a line that trims to an empty string.
///   *Recovery*: Remove or fix the empty line in the input file.
#[derive(Debug)]
pub enum DataError {
    /// I/O error while reading the input file.
    Io(std::io::Error),

    /// The input file is empty or yields no non-empty lines.
    EmptyFile,

    /// A line was empty after trimming.
    EmptyDataItem,
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataError::Io(e) => write!(f, "data io: {e}"),
            DataError::EmptyFile => write!(f, "data: input file is empty"),
            DataError::EmptyDataItem => write!(f, "data: empty line (data item) in input"),
        }
    }
}

impl std::error::Error for DataError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DataError::Io(e) => Some(e),
            DataError::EmptyFile | DataError::EmptyDataItem => None,
        }
    }
}

impl From<std::io::Error> for DataError {
    fn from(e: std::io::Error) -> Self {
        DataError::Io(e)
    }
}
