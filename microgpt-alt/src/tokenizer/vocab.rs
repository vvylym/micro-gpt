//! Vocabulary: mapping between token ids and symbols (e.g. characters or BPE substrings).
//!
//! Used by tokenizer implementations. Supports growing the vocab (for BPE later).

use std::collections::HashMap;

/// Maps token ids to symbols and back. Ids are contiguous from `0` to `len - 1`.
///
/// Symbols are stored as strings so the same vocab works for char (single-char string) and BPE (substring) tokenizers.
#[derive(Clone, Debug)]
pub struct Vocab {
    id_to_sym: Vec<String>,
    sym_to_id: HashMap<String, usize>,
}

impl Vocab {
    /// Builds a new vocab with the given symbols in order. Duplicate symbols are skipped (first occurrence wins).
    #[must_use]
    pub fn new(symbols: impl IntoIterator<Item = String>) -> Self {
        let mut id_to_sym = Vec::new();
        let mut sym_to_id = HashMap::new();
        for s in symbols {
            if sym_to_id.contains_key(&s) {
                continue;
            }
            let id = id_to_sym.len();
            id_to_sym.push(s.clone());
            sym_to_id.insert(s, id);
        }
        Vocab {
            id_to_sym,
            sym_to_id,
        }
    }

    /// Returns the number of symbols (vocab size).
    #[must_use]
    pub fn len(&self) -> usize {
        self.id_to_sym.len()
    }

    /// Returns `true` if the vocab is empty.
    #[must_use]
    #[allow(dead_code)] // used in tests; useful for generic code
    pub fn is_empty(&self) -> bool {
        self.id_to_sym.is_empty()
    }

    /// Returns the symbol for `id`, or `None` if out of range.
    #[must_use]
    pub fn get_symbol(&self, id: usize) -> Option<&str> {
        self.id_to_sym.get(id).map(String::as_str)
    }

    /// Returns the id for `symbol`, or `None` if not in vocab.
    #[must_use]
    pub fn get_id(&self, symbol: &str) -> Option<usize> {
        self.sym_to_id.get(symbol).copied()
    }

    /// Adds a new symbol and returns its id. If the symbol already exists, returns its existing id.
    /// Reserved for BPE and used in tests.
    #[allow(dead_code)]
    pub fn grow(&mut self, symbol: String) -> usize {
        if let Some(&id) = self.sym_to_id.get(&symbol) {
            return id;
        }
        let id = self.id_to_sym.len();
        self.id_to_sym.push(symbol.clone());
        self.sym_to_id.insert(symbol, id);
        id
    }
}
