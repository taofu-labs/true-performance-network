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
    BENCHMARK_FLOOR = "benchmark_floor"                  # benchmarks are pass/fail floors; rank by lowest max_memory
    RAM_CEILING = "ram_ceiling"  # max_memory is a pass/fail cap; rank by highest benchmark composite


class BenchmarkTask(BaseModel):
    name: str
    min_score: float
    weight: float


class ModelRequirements(BaseModel):
    format: str = "gguf"


class CompetitionSpec(BaseModel):
    schema_version: int = 1
    id: str
    name: str
    description: Optional[str] = None
    model_repo: Optional[str] = None  # HF repo ID of the base model miners should optimize

    start_block: int
    commit_end_block: int
    scoring_end_block: int

    top_n: int = 5
    emission_distribution: List[float]
    benchmarks: List[BenchmarkTask]
    model_requirements: ModelRequirements = ModelRequirements()
    reveal_grace_blocks: int = 150
    score_tolerance: float = 0.02
    ram_check_context_length: int = 4096

    emission_weight: float = 1.0  # this competition's fixed share of total network weight (0,1]
    distribution_blocks: int = 0  # blocks after scoring_end_block it keeps distributing that share

    competition_type: CompetitionType = CompetitionType.BENCHMARK_FLOOR
    max_memory_kb: Optional[int] = None  # required cap for RAM_CEILING competitions

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
    def validate_max_memory_kb(self):
        if self.competition_type == CompetitionType.RAM_CEILING and not self.max_memory_kb:
            raise ValueError("max_memory_kb is required when competition_type is ram_ceiling")
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
