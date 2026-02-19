//! Character-level tokenizer: one token per character, vocab built from a corpus (unique chars + BOS).

use super::super::Vocab;
use super::super::{Tokenizer, TokenizerError};

/// Character-level tokenizer. Vocab is built from a corpus: unique characters plus a BOS symbol.
#[derive(Clone, Debug)]
pub struct CharTokenizer {
    vocab: Vocab,
    bos_id: usize,
}

impl CharTokenizer {
    /// Builds a char tokenizer from a corpus string. Vocab = BOS (at index 0) + unique chars in corpus order.
    ///
    /// # Errors
    ///
    /// Never errors; corpus can be empty (vocab will be just BOS).
    #[must_use]
    pub fn from_corpus(corpus: &str, bos_symbol: &str) -> Self {
        let mut symbols = vec![bos_symbol.to_string()];
        for ch in corpus.chars() {
            let s = ch.to_string();
            if !symbols.contains(&s) {
                symbols.push(s);
            }
        }
        let vocab = Vocab::new(symbols);
        let bos_id = 0;
        CharTokenizer { vocab, bos_id }
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, s: &str) -> Result<Vec<usize>, TokenizerError> {
        let mut ids = Vec::with_capacity(s.chars().count());
        for ch in s.chars() {
            let sym = ch.to_string();
            let id = self
                .vocab
                .get_id(&sym)
                .ok_or(TokenizerError::UnknownSymbol(sym))?;
            ids.push(id);
        }
        Ok(ids)
    }

    fn decode(&self, ids: &[usize]) -> Result<String, TokenizerError> {
        let mut s = String::new();
        for &id in ids {
            let sym = self
                .vocab
                .get_symbol(id)
                .ok_or(TokenizerError::InvalidId(id))?;
            s.push_str(sym);
        }
        Ok(s)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn bos_id(&self) -> usize {
        self.bos_id
    }
}
