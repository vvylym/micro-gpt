//! Tokenization: encode text to token ids and decode back.
//!
//! This module defines the **trait** ([`Tokenizer`]) and **error** ([`TokenizerError`]).
//! Implementations live in the `impls` submodule (e.g. [`CharTokenizer`] for character-level).
//! Vocabulary is an implementation detail of [`CharTokenizer`] (private `vocab` module).

mod error;
mod impls;
mod vocab;

pub use error::TokenizerError;
pub use impls::CharTokenizer;
pub use vocab::Vocab;

/// Trait for tokenizers: encode text to ids and decode ids to text.
pub trait Tokenizer {
    /// Encodes a string into a sequence of token ids.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError::UnknownSymbol`] if a symbol (e.g. character) is not in the vocabulary.
    fn encode(&self, s: &str) -> Result<Vec<usize>, TokenizerError>;

    /// Decodes a sequence of token ids into a string.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError::InvalidId`] if an id is out of range.
    fn decode(&self, ids: &[usize]) -> Result<String, TokenizerError>;

    /// Vocabulary size (number of distinct tokens).
    fn vocab_size(&self) -> usize;

    /// Token id used for beginning-of-sequence.
    fn bos_id(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_tokenizer_vocab_size_and_bos_id() {
        let t = CharTokenizer::from_corpus("abc", "<BOS>");
        assert_eq!(t.vocab_size(), 4, "BOS + a, b, c");
        assert_eq!(t.bos_id(), 0);
    }

    #[test]
    fn char_tokenizer_encode_decode_round_trip() {
        let t = CharTokenizer::from_corpus("hello", "<BOS>");
        let ids = t.encode("hello").unwrap();
        let decoded = t.decode(&ids).unwrap();
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn char_tokenizer_encode_then_decode_round_trip() {
        let t = CharTokenizer::from_corpus("abc", "<BOS>");
        let s = "abc";
        let ids = t.encode(s).unwrap();
        assert_eq!(ids.len(), 3);
        let back = t.decode(&ids).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn char_tokenizer_unknown_char_returns_error() {
        let t = CharTokenizer::from_corpus("ab", "<BOS>");
        let result = t.encode("abc");
        assert!(matches!(result, Err(TokenizerError::UnknownSymbol(_))));
    }

    #[test]
    fn char_tokenizer_decode_invalid_id_returns_error() {
        let t = CharTokenizer::from_corpus("a", "<BOS>");
        let result = t.decode(&[0, 100]);
        assert!(matches!(result, Err(TokenizerError::InvalidId(100))));
    }

    #[test]
    fn char_tokenizer_bos_id_consistent_with_vocab() {
        let t = CharTokenizer::from_corpus("x", "<BOS>");
        assert!(t.bos_id() < t.vocab_size());
        let sym = t.decode(&[t.bos_id()]).unwrap();
        assert_eq!(sym, "<BOS>");
    }

    #[test]
    fn vocab_grow_adds_new_symbol() {
        let mut v = Vocab::new(["a".to_string(), "b".to_string()]);
        assert_eq!(v.len(), 2);
        let id = v.grow("c".to_string());
        assert_eq!(id, 2);
        assert_eq!(v.len(), 3);
        assert_eq!(v.get_symbol(2), Some("c"));
        assert_eq!(v.get_id("c"), Some(2));
    }

    #[test]
    fn vocab_grow_duplicate_returns_existing_id() {
        let mut v = Vocab::new(["a".to_string()]);
        let id1 = v.grow("b".to_string());
        let id2 = v.grow("b".to_string());
        assert_eq!(id1, id2);
        assert_eq!(v.len(), 2);
    }
}
