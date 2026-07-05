import json
from typing import Dict, List, Optional
from pydantic import BaseModel, field_validator


"""Benchmark score claim"""
class Claim(BaseModel):
    b: str
    s: float = 0.0


class MinerSubmission(BaseModel):
    """Parsed from on-chain commitment string after auto-reveal."""
    spec: int = 1
    competition_id: str
    claims: List[Claim]
    repository: str  # bare HF repo ID: "user/repo"
    file: str
    file_size: int
    file_sha256: str
    huggingface_revision: str  # immutable HF commit SHA the model was uploaded at

    @field_validator("repository")
    @classmethod
    def repository_must_not_be_url(cls, v: str) -> str:
        if v.startswith("https://") or v.startswith("http://"):
            raise ValueError("repository must be a bare HF repo ID (e.g. user/repo), not a URL")
        return v

    @field_validator("huggingface_revision")
    @classmethod
    def revision_must_be_git_sha(cls, v: str) -> str:
        # HF commit oids are 40-char git SHA-1; accept 7-64 hex to tolerate short/variant forms.
        if not (7 <= len(v) <= 64) or not all(c in "0123456789abcdefABCDEF" for c in v):
            raise ValueError("huggingface_revision must be a hex commit SHA (7-64 chars)")
        return v.lower()

    @field_validator("file_sha256")
    @classmethod
    def sha256_must_be_hex64(cls, v: str) -> str:
        if len(v) != 64 or not all(c in "0123456789abcdefABCDEF" for c in v):
            raise ValueError("file_sha256 must be 64 hex characters")
        return v.lower()

    @field_validator("file_size")
    @classmethod
    def size_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("file_size must be positive")
        return v

    @property
    def self_reported_scores(self) -> Dict[str, float]:
        return {c.b: c.s for c in self.claims}





class ScoringResult(BaseModel):
    hotkey: str
    competition_id: str
    passed_floors: bool
    disqualified: bool
    disqualification_reason: Optional[str] = None
    actual_scores: Dict[str, float]
    final_score: float
    actual_file_size_bytes: int
    lying_detected: bool
    eval_backend: str = "stub"


# ── Commit/reveal helpers ─────────────────────────────────────────────────────

def build_reveal_payload(
    competition_id: str,
    repository: str,
    file: str,
    file_sha256: str,
    file_size: int,
    claims: List[Claim],
    huggingface_revision: str,
) -> str:
    """Minified JSON string submitted to chain via TLE encryption."""
    data = {
        "spec": 1,
        "competition_id": competition_id,
        "repository": repository,
        "file": file,
        "file_sha256": file_sha256,
        "file_size": file_size,
        "huggingface_revision": huggingface_revision,
        "claims": [{"b": c.b, "s": c.s} for c in claims],
    }
    return json.dumps(data, separators=(",", ":"))


def parse_reveal_payload(raw: str) -> Optional[MinerSubmission]:
    """Parse an on-chain revealed JSON string into a MinerSubmission. Returns None if malformed."""
    try:
        data = json.loads(raw)
        return MinerSubmission.model_validate(data)
    except Exception:
        return None
