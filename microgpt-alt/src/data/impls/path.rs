//! [`DataLoader`](super::super::DataLoader) implementation that reads from a file path (UTF-8, one line per [`DataItem`](super::super::DataItem)).

use std::fs;
use std::path::Path;

use super::super::{Data, DataError, DataItem, DataLoader};

/// Loads data from a file path (UTF-8, one line per [`DataItem`](super::super::DataItem), trimmed; empty lines error).
#[derive(Clone, Debug)]
pub struct PathLoader<P>(pub P);

impl<P> PathLoader<P>
where
    P: AsRef<Path>,
{
    /// Creates a loader for the given path.
    #[must_use]
    pub fn new(path: P) -> Self {
        PathLoader(path)
    }
}

impl<P> DataLoader for PathLoader<P>
where
    P: AsRef<Path>,
{
    fn load(&self) -> Result<Data, DataError> {
        let content = fs::read_to_string(self.0.as_ref())?;
        let items: Result<Vec<DataItem>, DataError> = content.lines().map(DataItem::new).collect();
        let items = items?;
        Data::new(items)
    }
}

/// Convenience: load data from a path using [`PathLoader`].
///
/// # Errors
///
/// - [`DataError::Io`] when the path cannot be read or content is not valid UTF-8.
/// - [`DataError::EmptyDataItem`] when a line is empty after trimming.
/// - [`DataError::EmptyFile`] when the file yields no non-empty lines.
pub fn load_from_path(path: impl AsRef<Path>) -> Result<Data, DataError> {
    PathLoader::new(path).load()
}
