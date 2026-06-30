# GGUF Model Provenance Checker

Prescreen whether GGUF model B is plausibly derived from model A. Two static gates — no inference, no GPU. Stops early on hard disproof.

## Checks

**Check 1 — Tokenizer:** Compares tokenizer type, vocabulary, and BPE merge rules. Vocab size or merge count mismatch is a hard disproof. Entry differences <1% pass (common in fine-tunes that add chat-template tokens).

**Check 2 — Embedding CKA:** Samples `n_tokens` rows from `token_embd.weight` and computes linear Centered Kernel Alignment between A and B. CKA is invariant to rotation, permutation, and isotropic scaling, so it stays high under quantization/pruning and drops to the independent-model baseline otherwise. Pass if CKA ≥ threshold.

---

## Requirements

Python 3.11+ with:

```bash
pip install gguf numpy
```

---

## Run

```bash
python provenance_check.py model_a.gguf model_b.gguf
python provenance_check.py model_a.gguf model_b.gguf --threshold 0.85
python provenance_check.py model_a.gguf model_b.gguf --n-tokens 2048 --seed 123
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold` | `0.80` | CKA accept threshold — calibrate against known-independent same-tokenizer models |
| `--n-tokens` | `4096` | Token rows sampled for CKA |
| `--seed` | `42` | RNG seed for reproducible sampling |

Exit code `0` = plausible derivative, `1` = not a plausible derivative, `2` = input error.

---

## Interpreting results

| Result | Meaning |
|--------|---------|
| **PASS** (both checks) | B is a plausible derivative of A — confirm via benchmark |
| **FAIL** (tokenizer) | Hard disproof: different vocab/merges |
| **FAIL** (CKA) | Embedding geometry too different; not a plausible derivative |

The threshold MUST be calibrated: measure CKA for a few known-independent models that share the same tokenizer, then set the gate above that baseline.

---

## Limitations

- **Prescreen only.** A PASS is not confirmation — follow up with benchmarking.
- **Quantized tensors supported.** `_dequant_rows` dequantizes on the fly; requires `embed_dim % block_size == 0` (all valid GGUF files satisfy this).
- **Cannot detect weight replacement.** A model could share the tokenizer but have entirely retrained embeddings; Check 2 would catch this, but only if the CKA threshold is set correctly.
- **From-scratch / distilled models correctly fail.** They share no weights with A and will fall below the CKA threshold.
- **For high-stakes decisions**, follow up with dynamic checks: run both models on the same prompt distribution and compare output logits or KL-divergence.
