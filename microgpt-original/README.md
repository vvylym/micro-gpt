# microgpt-original

Faithful port of the minimal GPT: one lib holding the full algorithm (scalar autograd, transformer forward, training loop, inference). Kept **aligned** with the original design.

- Scalar autograd and computation graph
- Transformer forward (embedding, RMSNorm, multi-head attention with KV cache, MLP, lm_head)
- Training with cross-entropy loss and Adam
- Temperature-scaled sampling at inference
- Character-level tokenization with BOS

Input is one document per line. Default path: `data/input.txt`, or set `MICROGPT_INPUT_PATH`.

## Usage

From the **workspace root**:

```bash
make run-original

# or
cargo run -p microgpt-original
```

Custom input file:

```bash
MICROGPT_INPUT_PATH=/path/to/corpus.txt make run-original
```

## Build and test

From the workspace root:

```bash
cargo build -p microgpt-original
cargo test -p microgpt-original
```

Or use workspace-wide targets: `make check`, `make test`, `make ci`.

## License

Dual-licensed under MIT and Apache-2.0 (see repository root).
