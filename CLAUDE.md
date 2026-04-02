# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install as editable package
pip install -e .

# Run tests (requires transformers)
pip install transformers
python -m unittest discover tests

# Run a single test file
python -m unittest tests.test_huggingface_import

# Run example projects
python projects/adder/adder.py
python projects/chargpt/chargpt.py  # requires a text file (e.g. input.txt)
```

No linting configuration is set up in this project.

## Architecture

The core library lives in `mingpt/` (~840 lines total):

- **`model.py`** — GPT model. The full architecture (embedding, N transformer blocks, LM head) is in the `GPT` class. Each `Block` contains `CausalSelfAttention` (multi-head masked self-attention) and a feedforward layer, both with pre-normalization and residual connections. Key methods: `from_pretrained()` loads HuggingFace GPT-2 weights; `configure_optimizers()` applies weight decay only to Linear weights (not biases/layernorm); `generate()` does autoregressive sampling with temperature and top-k.

- **`trainer.py`** — Minimal training loop. `Trainer.run()` uses random sampling with replacement (not epoch-based), clips gradients at 1.0, and supports a callback dict (e.g. `{'on_batch_end': fn}`) for extensibility. Device (CPU/CUDA) is auto-detected.

- **`utils.py`** — `CfgNode` is a lightweight config object with dot-access, `merge_from_dict()`, and `merge_from_args()` for CLI overrides (e.g. `--model.n_layer=6`). `set_seed()` seeds Python/numpy/torch together.

- **`bpe.py`** — GPT-2 BPE tokenizer. `BPETokenizer` wraps `Encoder` and returns PyTorch tensors. Tokenizer files (`encoder.json`, `vocab.bpe`) are auto-downloaded and cached in `~/.cache/mingpt/`.

## Configuration pattern

Both `GPT` and `Trainer` expose a `get_default_config()` classmethod returning a `CfgNode`. Projects merge user overrides into it before construction:

```python
config = GPT.get_default_config()
config.model_type = 'gpt2'
config.vocab_size = 50257
config.block_size = 1024
model = GPT(config)
```

Built-in model presets (set via `model_type`): `openai-gpt`, `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `gpt-mini`, `gpt-micro`, `gpt-nano`. Setting `model_type` fills in `n_layer`, `n_head`, and `n_embd` automatically.

## Terminology

- **LAR** — Latent AutoRegression. The technique implemented in `projects/latent_ar/` where middle transformer layers are fine-tuned to autoregressively predict the next latent representation, enabling inference-time generation that bypasses the token encoder.

## Dataset contract

Custom datasets must return `(x, y)` tensors of token indices where `y` is `x` shifted by one (next-token prediction). Block size is enforced by the dataset, not the model. See `projects/adder/adder.py` and `projects/chargpt/chargpt.py` for examples.
