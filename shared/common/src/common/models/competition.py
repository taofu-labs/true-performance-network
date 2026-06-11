from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, model_validator


class CompetitionPhase(str, Enum):
    OPEN = "open"        # start_block → commit_end_block
    SCORING = "scoring"  # commit_end_block + grace → scoring_end_block
    COMPLETE = "complete"


class BenchmarkTask(BaseModel):
    name: str
    min_score: float
    weight: float


class ModelRequirements(BaseModel):
    format: str = "gguf"


class EvalConfig(BaseModel):
    backend: str = "stub"
    note: Optional[str] = None


class CompetitionSpec(BaseModel):
    schema_version: int = 1
    id: str
    name: str
    description: Optional[str] = None
    model_repo: Optional[str] = None  # HF repo ID of the base model miners should optimize

    # Phase boundaries — authoritative
    start_block: int
    commit_end_block: int
    scoring_end_block: int

    # Human-readable dates — display only
    start_date: Optional[str] = None
    commit_end_date: Optional[str] = None
    scoring_end_date: Optional[str] = None

    top_n: int = 5
    emission_distribution: List[float]
    benchmarks: List[BenchmarkTask]
    model_requirements: ModelRequirements = ModelRequirements()
    reveal_grace_blocks: int = 150
    score_tolerance: float = 0.02
    eval: EvalConfig = EvalConfig()

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

    def phase(self, current_block: int) -> CompetitionPhase:
        if current_block < self.commit_end_block:
            return CompetitionPhase.OPEN
        if current_block < self.scoring_end_block:
            return CompetitionPhase.SCORING
        return CompetitionPhase.COMPLETE

    def scoring_starts_at(self) -> int:
        """First block where validators should begin scoring (after grace period)."""
        return self.commit_end_block + self.reveal_grace_blocks

    def blocks_until_next_phase(self, current_block: int) -> int:
        phase = self.phase(current_block)
        if phase == CompetitionPhase.OPEN:
            return max(0, self.commit_end_block - current_block)
        if phase == CompetitionPhase.SCORING:
            return max(0, self.scoring_end_block - current_block)
        return 0

    def is_active(self, current_block: int) -> bool:
        return self.start_block <= current_block < self.scoring_end_block
