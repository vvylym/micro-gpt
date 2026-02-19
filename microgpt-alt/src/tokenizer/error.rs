//! Errors produced when encoding or decoding with a tokenizer.
//!
//! All errors from the tokenizer module use [`TokenizerError`].

use std::fmt;

/// Errors produced by the tokenizer module.
///
/// # Variants
///
/// - **UnknownSymbol**: A symbol (e.g. character) was encountered that is not in the vocabulary.
///   *When*: During [`encode`](super::Tokenizer::encode) when the input contains a symbol not seen when the tokenizer was built.
///   *Recovery*: Build the tokenizer from a corpus that includes this symbol, or add an UNK token (e.g. in a future BPE impl).
///
/// - **InvalidId**: A token id is out of range for the vocabulary.
///   *When*: During [`decode`](super::Tokenizer::decode) when an id is not in `[0, vocab_size)`.
///   *Recovery*: Ensure the ids were produced by this tokenizer’s `encode` or are otherwise valid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    /// A symbol not in the vocabulary was encountered during encode.
    UnknownSymbol(String),

    /// A token id is out of range during decode.
    InvalidId(usize),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::UnknownSymbol(s) => write!(f, "tokenizer: unknown symbol {s:?}"),
            TokenizerError::InvalidId(id) => write!(f, "tokenizer: invalid id {id}"),
        }
    }
}

impl std::error::Error for TokenizerError {}
