//! Types for loaded data: [`DataItem`] (one non-empty line) and [`Data`] (non-empty list of items).
//!
//! Validation at construction ensures empty items and empty data produce the right [`DataError`].

use std::fmt;

use super::DataError;

/// A single non-empty data item (e.g. one trimmed line of the corpus).
///
/// Wrapping the inner string allows validation: empty strings are rejected with [`DataError::EmptyDataItem`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DataItem(String);

impl DataItem {
    /// Creates a [`DataItem`] from a line (trimmed). Returns [`DataError::EmptyDataItem`] if empty after trim.
    ///
    /// # Errors
    ///
    /// - [`DataError::EmptyDataItem`] when `line` is empty or only whitespace.
    pub fn new(line: &str) -> Result<Self, DataError> {
        let s = line.trim();
        if s.is_empty() {
            return Err(DataError::EmptyDataItem);
        }
        Ok(DataItem(s.to_string()))
    }

    /// Returns the inner string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DataItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Loaded data: a non-empty list of [`DataItem`]s.
///
/// Construction fails with [`DataError::EmptyFile`] when the list is empty, so callers can distinguish
/// "no data" from "no lines" or "all lines empty".
#[derive(Clone, Debug)]
pub struct Data(Vec<DataItem>);

impl Data {
    /// Builds [`Data`] from a non-empty list of items. Returns [`DataError::EmptyFile`] if `items` is empty.
    ///
    /// # Errors
    ///
    /// - [`DataError::EmptyFile`] when `items` is empty.
    pub fn new(items: Vec<DataItem>) -> Result<Self, DataError> {
        if items.is_empty() {
            return Err(DataError::EmptyFile);
        }
        Ok(Data(items))
    }

    /// Returns the number of items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if there are no items (should not happen for a valid [`Data`]).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the items as a slice.
    #[must_use]
    pub fn items(&self) -> &[DataItem] {
        &self.0
    }

    /// Returns the items as string slices (e.g. for downstream use).
    #[must_use]
    pub fn lines(&self) -> Vec<&str> {
        self.0.iter().map(DataItem::as_str).collect()
    }
}
