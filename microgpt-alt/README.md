# microgpt-alt

Refactored minimal GPT implementation: **trait-based** and **modular**, with config, data loading, tokenizer, and (planned) model, autograd, loss, optimizer, sampling, and checkpoint.

## Structure

- **config** — Configuration (model dims, paths, training/inference options). Env-based builder (`MICROGPT_ALT_*`), validation, and `Default`.
- **data** — Data loading: `DataLoader` trait, `DataItem` / `Data` types, `DataError`. Implementation: `PathLoader` and `load_from_path()` for file-based input (one non-empty line per item).
- **tokenizer** — Tokenization: `Tokenizer` trait (encode, decode, vocab_size, bos_id), `TokenizerError`. Implementation: `CharTokenizer` (vocab from corpus: BOS + unique chars).

Conventions: traits in each module’s `mod.rs`; structs in dedicated files; implementations in `impls/` (one file per impl). Each module has its own error type (e.g. `ConfigError`, `DataError`, `TokenizerError`).

## Usage

From the **workspace root**:

```bash
make run-alt

# or
cargo run -p microgpt-alt
```

Currently the binary is a stub; the library exposes `config`, `data`, and `tokenizer` for use by a full training/inference pipeline (to be added in later features).

## Build and test

From the workspace root:

```bash
cargo build -p microgpt-alt
cargo test -p microgpt-alt
```

Or use workspace-wide targets: `make check`, `make test`, `make ci`.

## License

Dual-licensed under MIT and Apache-2.0 (see repository root).
