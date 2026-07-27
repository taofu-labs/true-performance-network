#!/usr/bin/env python3
"""GGUF derivative prescreen — cheap, download-only check that B derives from A.
 
Two static gates (no inference, no GPU):
  1. Tokenizer identity  — same vocab + merges. A necessary condition: derivatives
                           keep the tokenizer.
  2. Embedding CKA       — linear Centered Kernel Alignment between A's and B's
                           token-embedding geometry. CKA is invariant to rotation,
                           permutation, and isotropic scaling of the hidden dim, so it
                           stays high under quantization, pruning, and width reduction,
                           and drops to the independent-model baseline otherwise.
 
This is a PRESCREEN. A PASS is confirmed later by benchmarking the model. It assumes
from-scratch / distilled entries are disallowed — those share no weights with A and
correctly fail CKA. The accept threshold MUST be calibrated against a handful of
known-INDEPENDENT same-tokenizer models (measure their CKA, set the gate above it).
"""
 
import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
 
import numpy as np
 
try:
    from gguf import GGUFReader, GGMLQuantizationType
    from gguf.quants import dequantize as gguf_dequantize
    from gguf.constants import GGML_QUANT_SIZES
except ImportError:
    print("ERROR: gguf package not installed. Run: pip install gguf", file=sys.stderr)
    sys.exit(2)
 
 
# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
 
@dataclass
class TokenizerResult:
    passed: bool
    vocab_size_a: int | None = None
    vocab_size_b: int | None = None
    notes: list[str] = field(default_factory=list)
 
 
@dataclass
class EmbeddingResult:
    passed: bool
    cka: float = 0.0
    threshold: float = 0.0
    n_tokens: int = 0
    type_a: str = ""
    type_b: str = ""
    notes: list[str] = field(default_factory=list)
 
 
@dataclass
class ProvenanceResult:
    is_derivative: bool
    tokenizer: TokenizerResult | None = None
    embeddings: EmbeddingResult | None = None
 
 
# ---------------------------------------------------------------------------
# GGUF metadata helpers
# ---------------------------------------------------------------------------
 
def _get_str(reader: GGUFReader, key: str) -> str | None:
    fld = reader.fields.get(key)
    if fld is None or not fld.data:
        return None
    try:
        return bytes(fld.parts[fld.data[0]].tolist()).decode("utf-8")
    except Exception:
        return None
 
 
def _get_str_array(reader: GGUFReader, key: str) -> list[str] | None:
    fld = reader.fields.get(key)
    if fld is None:
        return None
    return [bytes(fld.parts[i].tolist()).decode("utf-8", errors="replace") for i in fld.data]
 
 
# ---------------------------------------------------------------------------
# Check 1 — Tokenizer
# ---------------------------------------------------------------------------
 
def check_tokenizer(reader_a: GGUFReader, reader_b: GGUFReader, max_diff_pct: float = 1.0) -> TokenizerResult:
    tok_a, tok_b = _get_str(reader_a, "tokenizer.ggml.model"), _get_str(reader_b, "tokenizer.ggml.model")
    if tok_a and tok_b and tok_a != tok_b:
        return TokenizerResult(False, notes=[f"tokenizer type mismatch: {tok_a!r} vs {tok_b!r}"])
 
    vocab_a, vocab_b = _get_str_array(reader_a, "tokenizer.ggml.tokens"), _get_str_array(reader_b, "tokenizer.ggml.tokens")
    if vocab_a is None or vocab_b is None:
        return TokenizerResult(False, notes=["tokenizer.ggml.tokens missing in one or both files"])
    if len(vocab_a) != len(vocab_b):
        return TokenizerResult(False, len(vocab_a), len(vocab_b), [f"vocab size mismatch: {len(vocab_a)} vs {len(vocab_b)}"])
 
    vocab_diff_pct = sum(a != b for a, b in zip(vocab_a, vocab_b)) / len(vocab_a) * 100
 
    merges_a, merges_b = _get_str_array(reader_a, "tokenizer.ggml.merges"), _get_str_array(reader_b, "tokenizer.ggml.merges")
    if (merges_a is None) != (merges_b is None):
        return TokenizerResult(False, len(vocab_a), len(vocab_b), ["merges present in one file only"])
    if merges_a and merges_b and len(merges_a) != len(merges_b):
        return TokenizerResult(False, len(vocab_a), len(vocab_b), [f"merge count mismatch: {len(merges_a)} vs {len(merges_b)}"])
    merge_diff_pct = sum(a != b for a, b in zip(merges_a, merges_b)) / len(merges_a) * 100 if merges_a else 0.0
 
    passed = vocab_diff_pct < max_diff_pct and merge_diff_pct < max_diff_pct
    notes = [f"vocab differ {vocab_diff_pct:.2f}%, merges differ {merge_diff_pct:.2f}% (limit {max_diff_pct}%)"]
    return TokenizerResult(passed, len(vocab_a), len(vocab_b), notes)
 
 
# ---------------------------------------------------------------------------
# Check 2 — Embedding CKA
# ---------------------------------------------------------------------------
 
def _dequant_rows(tensor, row_ids: np.ndarray) -> np.ndarray:
    """Dequantize only `row_ids` to f32. Requires embed_dim % block == 0"""
    qtype = tensor.tensor_type
    embed_dim, vocab = int(tensor.shape[0]), int(tensor.shape[-1])
    block, tsize = GGML_QUANT_SIZES[qtype]
    if embed_dim % block != 0:
        raise ValueError(f"embed_dim={embed_dim} not divisible by {qtype.name} block size {block} — not a valid quantized file")
    row_bytes = (embed_dim // block) * tsize
    raw = tensor.data.view(np.uint8).reshape(vocab, row_bytes)[row_ids]
    return gguf_dequantize(raw.ravel(), qtype).reshape(len(row_ids), embed_dim).astype(np.float32)
 
 
def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA of two (n, d) matrices, in [0, 1]. Invariant to orthogonal transforms + isotropic scale."""
    x, y = x - x.mean(0), y - y.mean(0)  # column-center: strip the shared anisotropic offset
    cross = np.linalg.norm(x.T @ y) ** 2
    denom = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(cross / denom) if denom > 0 else 0.0
 
 
def check_embeddings(reader_a: GGUFReader, reader_b: GGUFReader,
                     threshold: float = 0.80, n_tokens: int = 4096, seed: int = 42) -> EmbeddingResult:
    ta = next((t for t in reader_a.tensors if t.name == "token_embd.weight"), None)
    tb = next((t for t in reader_b.tensors if t.name == "token_embd.weight"), None)
    if ta is None or tb is None:
        return EmbeddingResult(False, notes=[f"token_embd.weight missing in model {'A' if ta is None else 'B'}"])
 
    vocab_a, vocab_b = int(ta.shape[-1]), int(tb.shape[-1])
    if vocab_a != vocab_b:  # rows must index the same tokens
        return EmbeddingResult(False, type_a=ta.tensor_type.name, type_b=tb.tensor_type.name,
                               notes=[f"vocab size mismatch in tensors: {vocab_a} vs {vocab_b}"])
 
    ids = np.random.default_rng(seed).choice(vocab_a, size=min(n_tokens, vocab_a), replace=False)
    try:
        cka = _linear_cka(_dequant_rows(ta, ids), _dequant_rows(tb, ids))
    except (ValueError, NotImplementedError) as e:
        return EmbeddingResult(False, type_a=ta.tensor_type.name, type_b=tb.tensor_type.name,
                               notes=[f"dequantization failed: {e}"])
 
    return EmbeddingResult(cka >= threshold, cka, threshold, len(ids), ta.tensor_type.name, tb.tensor_type.name)
 
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
 
def check_provenance(path_a: str | Path, path_b: str | Path,
                     threshold: float = 0.80, n_tokens: int = 4096, seed: int = 42) -> ProvenanceResult:
    """Return ProvenanceResult; is_derivative=True if B plausibly derives from A (confirm via benchmark)."""
    try:
        ra, rb = GGUFReader(str(path_a), "r"), GGUFReader(str(path_b), "r")
    except Exception as e:
        return ProvenanceResult(False, tokenizer=TokenizerResult(False, notes=[str(e)]))
 
    tok = check_tokenizer(ra, rb)
    if not tok.passed:
        return ProvenanceResult(False, tokenizer=tok)
 
    emb = check_embeddings(ra, rb, threshold=threshold, n_tokens=n_tokens, seed=seed)
    return ProvenanceResult(emb.passed, tokenizer=tok, embeddings=emb)
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def _print_result(r: ProvenanceResult) -> None:
    tok = r.tokenizer
    if tok:
        print(f"\nCheck 1 (tokenizer): {'PASS' if tok.passed else 'FAIL'}")
        if tok.vocab_size_a is not None:
            print(f"  vocab size  A={tok.vocab_size_a}  B={tok.vocab_size_b}")
        for n in tok.notes:
            print(f"  {n}")
 
    emb = r.embeddings
    if emb:
        print(f"\nCheck 2 (embedding CKA): {'PASS' if emb.passed else 'FAIL'}")
        print(f"  token_embd.weight  A={emb.type_a}  B={emb.type_b}  (n={emb.n_tokens} tokens)")
        print(f"  CKA = {emb.cka:.4f}   threshold = {emb.threshold:.4f}")
        for n in emb.notes:
            print(f"  {n}")
 
    print(f"\nResult: {'PASS — plausible derivative (confirm via benchmark)' if r.is_derivative else 'FAIL — not a plausible derivative'}")
 
 
def main() -> None:
    p = argparse.ArgumentParser(description="Prescreen whether GGUF model B is derived from model A.")
    p.add_argument("model_a")
    p.add_argument("model_b")
    p.add_argument("--threshold", type=float, default=0.80, help="CKA accept threshold (calibrate vs independent models)")
    p.add_argument("--n-tokens", type=int, default=4096, help="number of token rows to sample for CKA")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
 
    for f in (args.model_a, args.model_b):
        if not Path(f).exists():
            print(f"ERROR: file not found: {f}", file=sys.stderr)
            sys.exit(2)
 
    result = check_provenance(args.model_a, args.model_b, threshold=args.threshold, n_tokens=args.n_tokens, seed=args.seed)
    _print_result(result)
    sys.exit(0 if result.is_derivative else 1)
 
 
if __name__ == "__main__":
    main()