# MicroGPT

A minimal, dependency-minimal implementation that **trains and runs a GPT** in a single file holding the full algorithm.

This repo for learning purposes provides a **Rust implementation** (`src/main.rs`) — the same algorithm as the [original Python](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

The port is **aligned** as much as possible with the original: 
- Tokenization (character-level + BOS), 
- RMSNorm, 
- Multi-head attention with KV cache, 
- MLP blocks, 
- Cross-entropy loss, 
- Adam with linear LR decay, 
- Temperature-scaled sampling at inference.

 Randomness (shuffle, init, sampling) follows the same logical order; concrete random streams may differ across languages.

## References

- [**Karpathy's Blog**](https://karpathy.github.io/2026/02/12/microgpt/) — step-by-step walkthrough: dataset, tokenizer, autograd, parameters, architecture (linear, softmax, RMSNorm, attention, MLP), training loop, and inference.
- [**Python Gist**](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)

## Installation

**Requirements:**:
- [Rust](https://www.rust-lang.org/)
- [Make]() (optional)

```bash
git clone https://github.com/vvylym/migro-gpt

cd micro-gpt
```

Optional: ensure a dataset exists. The Rust binary reads from a path configured in code (one document per line, e.g. names). If missing, create or symlink your input file to that path.

## Usage

**Train and run inference (Rust):**

```bash
cargo run

# or
make run 
```

This will:

1. Load documents from the configured input path, shuffle them, and build a character-level vocab + BOS.
2. Initialize a small transformer.
3. Train with Adam and linear learning-rate decay.
4. Sample 20 items.

## Contributing

Contributions that will keep the implementation minimal and aligned with the reference are welcome.

## License

Dual-licensed under:

- **MIT License** (see [LICENSE-MIT](LICENSE-MIT))
- **Apache License 2.0** (see [LICENSE-APACHE](LICENSE-APACHE))
