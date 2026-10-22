import json
import re
from typing import Dict, List, Optional
from pydantic import AliasChoices, BaseModel, Field, field_validator


# Coordinator run ids are either a short id ("r1234") or a full UUID. Both are
# accepted by the coordinator's /status/:run_id route.
_RUN_ID_RE = re.compile(r"^(r[1-9][0-9]{0,18}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$")

PAYLOAD_SPEC_VERSION = 3

# Specs a reveal may still carry. Spec 3 dropped the `max_memory` self-report,
# which pydantic ignores as an extra field, so a spec-2 payload is otherwise
# identical and still scores correctly. Rejecting it would strand every miner
# who committed before the validator upgraded — their reveal is already sealed
# on chain and cannot be re-made.
SUPPORTED_SPEC_VERSIONS = frozenset({2, PAYLOAD_SPEC_VERSION})


"""Coordinator benchmark run submitted by the miner"""
class BenchmarkRun(BaseModel):
    b: str  # benchmark name
    r: str  # coordinator run id (short id or UUID)

    @field_validator("r")
    @classmethod
    def run_id_must_be_well_formed(cls, v: str) -> str:
        if not _RUN_ID_RE.match(v):
            raise ValueError("run id must be a coordinator short id (r123) or a UUID")
        return v


class MinerSubmission(BaseModel):
    """Parsed from on-chain commitment string after auto-reveal."""
    spec: int = PAYLOAD_SPEC_VERSION
    competition_id: str
    runs: List[BenchmarkRun]
    repository: str  # bare HF repo ID: "user/repo"
    file: str
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

    @field_validator("spec")
    @classmethod
    def spec_must_be_supported(cls, v: int) -> int:
        if v not in SUPPORTED_SPEC_VERSIONS:
            raise ValueError(
                f"unsupported payload spec {v}, expected one of {sorted(SUPPORTED_SPEC_VERSIONS)}"
            )
        return v

    @property
    def run_ids(self) -> Dict[str, str]:
        """{benchmark_name: coordinator_run_id}. Later duplicates win."""
        return {run.b: run.r for run in self.runs}





class ScoringResult(BaseModel):
    hotkey: str
    competition_id: str
    final_score: float
    # GPU bytes the coordinator measured, in KB. Not a miner claim.
    #
    # Followers are run by other operators and upgrade at their own pace, so
    # this is read over the wire from a leader that may be on either version.
    # The old name is accepted so a mixed-version pair keeps setting weights.
    measured_memory_kb: int = Field(
        validation_alias=AliasChoices("measured_memory_kb", "max_memory_kb")
    )


# ── Commit/reveal helpers ─────────────────────────────────────────────────────

def build_reveal_payload(
    competition_id: str,
    repository: str,
    file: str,
    file_sha256: str,
    runs: List[BenchmarkRun],
    huggingface_revision: str,
) -> str:
    """Minified JSON string submitted to chain via TLE encryption."""
    data = {
        "spec": PAYLOAD_SPEC_VERSION,
        "competition_id": competition_id,
        "repository": repository,
        "file": file,
        "file_sha256": file_sha256,
        "huggingface_revision": huggingface_revision,
        "runs": [{"b": r.b, "r": r.r} for r in runs],
    }
    return json.dumps(data, separators=(",", ":"))


def parse_reveal_payload(raw: str) -> Optional[MinerSubmission]:
    """Parse an on-chain revealed JSON string into a MinerSubmission. Returns None if malformed."""
    try:
        data = json.loads(raw)
        return MinerSubmission.model_validate(data)
    except Exception:
        return None
