//! Input data loading (e.g. corpus lines from a file).
//!
//! Self-contained module with its own [`Error`] type. Load lines with [`load_lines`].

mod error;

use std::fs;
use std::path::Path;

pub use error::Error;

/// Loads non-empty lines from a text file at `path`.
///
/// Each line is trimmed; empty lines after trimming are omitted. Returns [`Error::Io`] if the
/// file cannot be read (e.g. not found, permission denied, invalid UTF-8), and [`Error::EmptyFile`]
/// if the file yields no lines.
///
/// # Errors
///
/// - [`Error::Io`] when the path cannot be read or content is not valid UTF-8.
/// - [`Error::EmptyFile`] when the file is empty or contains only whitespace.
pub fn load_lines(path: impl AsRef<Path>) -> Result<Vec<String>, Error> {
    let content = fs::read_to_string(path.as_ref()).map_err(Error::from)?;
    let lines: Vec<String> = content
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    if lines.is_empty() {
        return Err(Error::EmptyFile);
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::error::Error as _;
    use std::io::Write;

    #[test]
    fn load_lines_from_temp_file_returns_lines() {
        let dir = std::env::temp_dir();
        let path = dir.join("microgpt_alt_data_test_lines.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "first line").unwrap();
        writeln!(f, "  second line  ").unwrap();
        writeln!(f, "third").unwrap();
        f.sync_all().unwrap();
        drop(f);

        let result = load_lines(&path);
        let _ = std::fs::remove_file(&path);
        let lines = result.unwrap();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "first line");
        assert_eq!(lines[1], "second line");
        assert_eq!(lines[2], "third");
    }

    #[test]
    fn load_lines_empty_file_returns_empty_file_error() {
        let dir = std::env::temp_dir();
        let path = dir.join("microgpt_alt_data_test_empty.txt");
        let _ = std::fs::File::create(&path).unwrap();

        let result = load_lines(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(Error::EmptyFile)));
    }

    #[test]
    fn load_lines_whitespace_only_returns_empty_file_error() {
        let dir = std::env::temp_dir();
        let path = dir.join("microgpt_alt_data_test_ws.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "   ").unwrap();
        writeln!(f).unwrap();
        f.sync_all().unwrap();
        drop(f);

        let result = load_lines(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(Error::EmptyFile)));
    }

    #[test]
    fn load_lines_missing_file_returns_io_error() {
        let path = std::path::Path::new("/nonexistent/microgpt_alt_never_exists.txt");
        let result = load_lines(path);
        assert!(matches!(result, Err(Error::Io(_))));
    }

    #[test]
    fn error_display_and_from_io() {
        let e = Error::from(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        let s = e.to_string();
        assert!(s.contains("data io") || s.contains("file not found"));
        assert!(e.source().is_some());
    }

    #[test]
    fn error_empty_file_display() {
        let e = Error::EmptyFile;
        assert!(e.to_string().contains("empty"));
        assert!(e.source().is_none());
    }
}
