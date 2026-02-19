//! Implementations of [`Tokenizer`](super::super::Tokenizer).
//!
//! One file per implementation: e.g. [`char`] for character-level tokenizer.

mod char_impl;

pub use char_impl::CharTokenizer;
