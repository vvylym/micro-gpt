# MicroGPT

A minimal, dependency-minimal implementation in Rust that **trains and runs a GPT**, aligned with the original design: character-level tokenization + BOS, RMSNorm, multi-head attention with KV cache, MLP blocks, cross-entropy loss, Adam with linear LR decay, and temperature-scaled sampling at inference.

The repository is a **Cargo workspace** with two crates:

| Crate | Description |
|-------|-------------|
| **microgpt-original** | Faithful port: full algorithm in one place (lib + binary). Run training and inference with default or env-configured input path. |
| **microgpt-alt** | Refactored implementation: config, data loading, tokenizer (trait + char impl), and future modules (model, autograd, etc.). Trait-based and modular. |

## References

- [**Karpathy's Blog**](https://karpathy.github.io/2026/02/12/microgpt/)
- [**Python Gist**](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)

## Requirements

- [Rust](https://www.rust-lang.org/)
- [Make](https://www.gnu.org/software/make/) (optional, for `make run-original`, `make run-alt`, `make ci`, etc.)

## Installation

```bash
git clone https://github.com/vvylym/micro-gpt
cd micro-gpt
```

## Usage

**Run the original port (train + inference):**

```bash
make run-original

# or
cargo run -p microgpt-original
```

Input: one document per line. Default path is `data/input.txt`, or set `MICROGPT_INPUT_PATH`.

**Run the refactored crate (stub binary for now):**

```bash
make run-alt

# or
cargo run -p microgpt-alt
```

**Build and test the whole workspace:**

```bash
cargo build --workspace
cargo test --workspace
```

**Makefile targets** (from workspace root):

| Target | Description |
|--------|-------------|
| `make check` | `cargo check --workspace --all-features --all-targets` |
| `make fmt` / `make fmt-check` | Format code / check formatting |
| `make clippy` | Lint with clippy |
| `make test` | Run all tests |
| `make doc` | Build docs for the workspace |
| `make deny` | Run cargo-deny (licenses, bans, sources) |
| `make clean` | `cargo clean` |
| `make run-original` | Run microgpt-original binary |
| `make run-alt` | Run microgpt-alt binary |
| `make machete` | Check for unused dependencies (installs if needed) |
| `make coverage` | Generate lcov report |
| `make coverage-check` | Run tests with coverage, fail if lines < 95% |
| `make pmat` | Run pmat quality gate (installs if needed) |
| `make ci` | Full CI: fmt-check, clippy, test, doc, deny, machete, coverage-check, pmat |

See [microgpt-original/README.md](microgpt-original/README.md) and [microgpt-alt/README.md](microgpt-alt/README.md) for crate-specific details.

## Contributing

Contributions are welcome.

## License

Dual-licensed under:

- **MIT License** (see [LICENSE-MIT](LICENSE-MIT))
- **Apache License 2.0** (see [LICENSE-APACHE](LICENSE-APACHE))
