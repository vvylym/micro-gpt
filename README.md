# MicroGPT

A minimal, dependency-minimal implementation in Rust that **trains and runs a GPT** in a single file holding the full algorithm and **aligned** as much as possible with the original, including: 
- Tokenization (character-level + BOS), 
- RMSNorm, 
- Multi-head attention with KV cache, 
- MLP blocks, 
- Cross-entropy loss, 
- Adam with linear LR decay, 
- Temperature-scaled sampling at inference.

Randomness (shuffle, init, sampling) follows the same logical order; concrete random streams may differ across languages.

## References

- [**Karpathy's Blog**](https://karpathy.github.io/2026/02/12/microgpt/)
- [**Python Gist**](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)

## Installation

**Requirements:**:
- [Rust](https://www.rust-lang.org/)
- [Make]() (optional)

```bash
git clone https://github.com/vvylym/migro-gpt

cd micro-gpt
```

## Usage

**Train and run inference (Rust):**

```bash
cargo run

# or
make run 
```

## Contributing

Contributions are welcome.

## License

Dual-licensed under:

- **MIT License** (see [LICENSE-MIT](LICENSE-MIT))
- **Apache License 2.0** (see [LICENSE-APACHE](LICENSE-APACHE))
