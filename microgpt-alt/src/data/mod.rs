//! Input data loading (e.g. corpus lines from a file).
//!
//! This module defines the **trait** ([`DataLoader`]), **models** ([`DataItem`], [`Data`]), and **error** ([`DataError`]).
//! Implementations (e.g. [`PathLoader`] for file paths) are in the `impls` submodule. The trait allows sync load now and async later.

mod error;
mod impls;
mod types;

pub use error::DataError;
pub use impls::{load_from_path, PathLoader};
pub use types::{Data, DataItem};

/// Trait for loading input data.
///
/// Implementations can be sync (e.g. [`PathLoader`]) or, in the future, async (e.g. an
/// `AsyncDataLoader` or `async fn load` when the crate adds async support).
pub trait DataLoader {
    /// Loads data. Returns [`Data`] or a [`DataError`].
    fn load(&self) -> Result<Data, DataError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::error::Error as _;
    use std::io::Write;
    use std::path::Path;

    #[test]
    fn load_from_path_temp_file_returns_data() {
        let dir = std::env::temp_dir();
        let path = dir.join("microgpt_alt_data_test_lines.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "first line").unwrap();
        writeln!(f, "  second line  ").unwrap();
        writeln!(f, "third").unwrap();
        f.sync_all().unwrap();
        drop(f);

        let result = load_from_path(&path);
        let _ = std::fs::remove_file(&path);
        let data = result.unwrap();
        assert_eq!(data.len(), 3);
        assert_eq!(data.items()[0].as_str(), "first line");
        assert_eq!(data.items()[1].as_str(), "second line");
        assert_eq!(data.items()[2].as_str(), "third");
        assert_eq!(data.lines(), ["first line", "second line", "third"]);
    }

    #[test]
    fn load_from_path_empty_file_returns_empty_file_error() {
        let dir = std::env::temp_dir();
        let path = dir.join("microgpt_alt_data_test_empty.txt");
        let _ = std::fs::File::create(&path).unwrap();

        let result = load_from_path(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(DataError::EmptyFile)));
    }

    #[test]
    fn load_from_path_whitespace_only_returns_empty_data_item_error() {
        let dir = std::env::temp_dir();
        let path = dir.join("microgpt_alt_data_test_ws.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "   ").unwrap();
        writeln!(f).unwrap();
        f.sync_all().unwrap();
        drop(f);

        let result = load_from_path(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(DataError::EmptyDataItem)));
    }

    #[test]
    fn load_from_path_empty_line_returns_empty_data_item_error() {
        let dir = std::env::temp_dir();
        let path = dir.join("microgpt_alt_data_test_empty_line.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "first").unwrap();
        writeln!(f, "   ").unwrap();
        writeln!(f, "third").unwrap();
        f.sync_all().unwrap();
        drop(f);

        let result = load_from_path(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(DataError::EmptyDataItem)));
    }

    #[test]
    fn load_from_path_missing_file_returns_io_error() {
        let path = Path::new("/nonexistent/microgpt_alt_never_exists.txt");
        let result = load_from_path(path);
        assert!(matches!(result, Err(DataError::Io(_))));
    }

    #[test]
    fn data_item_new_rejects_empty() {
        assert!(DataItem::new("").is_err());
        assert!(DataItem::new("   ").is_err());
        assert!(matches!(DataItem::new("  "), Err(DataError::EmptyDataItem)));
    }

    #[test]
    fn data_item_new_accepts_non_empty() {
        let a = DataItem::new("hello").unwrap();
        assert_eq!(a.as_str(), "hello");
        let b = DataItem::new("  world  ").unwrap();
        assert_eq!(b.as_str(), "world");
    }

    #[test]
    fn data_new_rejects_empty_vec() {
        assert!(matches!(Data::new(vec![]), Err(DataError::EmptyFile)));
    }

    #[test]
    fn data_new_accepts_non_empty_vec() {
        let items = vec![DataItem::new("a").unwrap(), DataItem::new("b").unwrap()];
        let data = Data::new(items).unwrap();
        assert_eq!(data.len(), 2);
        assert!(!data.is_empty());
    }

    #[test]
    fn data_error_display_and_from_io() {
        let e = DataError::from(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        let s = e.to_string();
        assert!(s.contains("data io") || s.contains("file not found"));
        assert!(e.source().is_some());
    }

    #[test]
    fn data_error_empty_file_display() {
        let e = DataError::EmptyFile;
        assert!(e.to_string().contains("empty"));
        assert!(e.source().is_none());
    }

    #[test]
    fn data_error_empty_data_item_display() {
        let e = DataError::EmptyDataItem;
        assert!(e.to_string().to_lowercase().contains("empty"));
        assert!(e.source().is_none());
    }

    #[test]
    fn path_loader_implements_trait() {
        let path = Path::new("/nonexistent/microgpt_alt_never_exists.txt");
        let loader = PathLoader::new(path);
        let result = loader.load();
        assert!(matches!(result, Err(DataError::Io(_))));
    }
}
