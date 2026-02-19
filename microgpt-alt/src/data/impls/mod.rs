//! Implementations of [`DataLoader`](super::DataLoader).
//!
//! One file per implementation: e.g. [`path`] for loading from a file path.

mod path;

pub use path::{load_from_path, PathLoader};
