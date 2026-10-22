from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, model_validator


class CompetitionPhase(str, Enum):
    OPEN = "open"                  # start_block → commit_end_block
    REVEALING = "revealing"        # commit_end_block → commit_end_block + reveal_grace_blocks
    SCORING = "scoring"            # commit_end_block + grace → scoring_end_block
    DISTRIBUTING = "distributing"  # scoring_end_block → distribution_end_block
    COMPLETE = "complete"          # distribution_end_block reached


class CompetitionType(str, Enum):
    BENCHMARK_FLOOR = "benchmark_floor"  # benchmarks are pass/fail floors; rank by lowest measured memory
    # Measured memory is a pass/fail cap; rank by highest benchmark composite.
    GPU_MEMORY_CEILING = "gpu_memory_ceiling"


# Reserved run name: miners commit the KL run as {"b": "kl", "r": "r1234"}, so it
# arrives through MinerSubmission.run_ids without a payload spec bump.
KL_RUN_NAME = "kl"


class BenchmarkTask(BaseModel):
    name: str
    min_score: float
    # Fluid epoch a run must have used to count (e.g. "epoch-000001-r7"). The
    # coordinator accepts runs on several epochs at once; scores across epochs
    # aren't comparable. None = any epoch.
    fluid_epoch_id: Optional[str] = None


class ModelRequirements(BaseModel):
    format: str = "gguf"


class CompetitionSpec(BaseModel):
    schema_version: int = 1
    id: str
    name: str
    description: Optional[str] = None
    model_repo: Optional[str] = None  # HF repo ID of the base model miners should optimize
    # Platform corpus the KL run must measure on. Its text is never exposed (the
    # catalog publishes rounded counts only), so miners cannot tune against it.
    # None disables KL scoring.
    kl_corpus_id: Optional[str] = None

    start_block: int
    commit_end_block: int
    scoring_end_block: int

    top_n: int = 5
    emission_distribution: List[float]
    benchmarks: List[BenchmarkTask]
    model_requirements: ModelRequirements = ModelRequirements()
    reveal_grace_blocks: int = 150

    emission_weight: float = 1.0  # this competition's fixed share of total network weight (0,1]
    distribution_blocks: int = 0  # blocks after scoring_end_block it keeps distributing that share

    competition_type: CompetitionType = CompetitionType.BENCHMARK_FLOOR
    # Cap on the GPU memory the coordinator measures while loading the candidate,
    # in KB. Required for GPU_MEMORY_CEILING competitions.
    #
    # This is GPU bytes at full offload, as the benchmark coordinator measures
    # them — not host RAM. A threshold carried over from the old CPU-side RAM
    # measurement does not transfer.
    max_gpu_memory_kb: Optional[int] = None

    @model_validator(mode="after")
    def validate_emission_distribution(self):
        if len(self.emission_distribution) != self.top_n:
            raise ValueError(
                f"emission_distribution has {len(self.emission_distribution)} entries "
                f"but top_n={self.top_n}. They must match."
            )
        total = sum(self.emission_distribution)
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"emission_distribution sums to {total}, must be 1.0")
        return self

    @model_validator(mode="after")
    def validate_kl_run_name_not_a_benchmark(self):
        """A benchmark named `kl` would collide with the multiplier in verified_scores."""
        if any(t.name == KL_RUN_NAME for t in self.benchmarks):
            raise ValueError(f"'{KL_RUN_NAME}' is reserved for the KL comparison run and cannot name a benchmark")
        return self

    @model_validator(mode="after")
    def validate_kl_corpus_has_a_base(self):
        """A KL run is only meaningful against a pinned base: without one the
        miner names the base themselves, and can pick a weak model their quant
        barely diverges from, which makes the multiplier free.

        Enforced on the model rather than in creation_errors because
        POST /v1/competitions validates the spec and stores it without ever
        calling creation_errors.
        """
        if self.kl_corpus_id and not self.model_repo:
            raise ValueError(
                "kl_corpus_id requires model_repo: it is the base the KL run compares against"
            )
        return self

    @model_validator(mode="after")
    def validate_max_gpu_memory_kb(self):
        if self.competition_type == CompetitionType.GPU_MEMORY_CEILING and not self.max_gpu_memory_kb:
            raise ValueError("max_gpu_memory_kb is required when competition_type is gpu_memory_ceiling")
        return self

    @model_validator(mode="after")
    def validate_emission_weight(self):
        if not (0 < self.emission_weight <= 1.0):
            raise ValueError(f"emission_weight must be in (0, 1], got {self.emission_weight}")
        if self.distribution_blocks < 0:
            raise ValueError(f"distribution_blocks must be >= 0, got {self.distribution_blocks}")
        return self

    def phase(self, current_block: int) -> CompetitionPhase:
        if current_block < self.commit_end_block:
            return CompetitionPhase.OPEN
        if current_block < self.scoring_starts_at():
            return CompetitionPhase.REVEALING
        if current_block < self.scoring_end_block:
            return CompetitionPhase.SCORING
        if current_block < self.distribution_end_block():
            return CompetitionPhase.DISTRIBUTING
        return CompetitionPhase.COMPLETE

    def scoring_starts_at(self) -> int:
        """First block where validators should begin scoring (after grace period)."""
        return self.commit_end_block + self.reveal_grace_blocks

    def distribution_end_block(self) -> int:
        """Last block (exclusive) this competition's emission_weight is still distributed."""
        return self.scoring_end_block + self.distribution_blocks

    def is_distributing(self, current_block: int) -> bool:
        """True once scoring has ended and until the distribution window closes."""
        return self.scoring_end_block <= current_block < self.distribution_end_block()

    def blocks_until_next_phase(self, current_block: int) -> int:
        phase = self.phase(current_block)
        if phase == CompetitionPhase.OPEN:
            return max(0, self.commit_end_block - current_block)
        if phase == CompetitionPhase.REVEALING:
            return max(0, self.scoring_starts_at() - current_block)
        if phase == CompetitionPhase.SCORING:
            return max(0, self.scoring_end_block - current_block)
        if phase == CompetitionPhase.DISTRIBUTING:
            return max(0, self.distribution_end_block() - current_block)
        return 0

    def is_active(self, current_block: int) -> bool:
        return self.start_block <= current_block < self.distribution_end_block()
