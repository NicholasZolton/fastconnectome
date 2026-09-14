"""Public value types shared across models and runtimes."""

from dataclasses import dataclass, field
from typing import Generic, TypeVar

ActionT = TypeVar("ActionT")
ActivityT = TypeVar("ActivityT")
ObservationT = TypeVar("ObservationT")
MetricValue = str | int | float | bool


@dataclass(frozen=True, slots=True)
class ModelInfo:
    model: str
    release: str
    dynamics: str
    backend: str
    neurons: int
    connections: int


@dataclass(frozen=True, slots=True)
class ActivitySummary:
    total_spikes: int
    selected_spikes: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LearningSummary:
    enabled: bool
    changed_connections: int


@dataclass(frozen=True, slots=True)
class Timing:
    neural_ms: float
    compute_seconds: float
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class SimulationResult(Generic[ActivityT]):
    activity: ActivityT
    activity_summary: ActivitySummary
    learning_summary: LearningSummary
    timing: Timing
    metrics: dict[str, MetricValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DecodedAction(Generic[ActionT]):
    action: ActionT
    metrics: dict[str, MetricValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StepResult(Generic[ActionT]):
    action: ActionT
    activity: ActivitySummary
    learning: LearningSummary
    timing: Timing
    metrics: dict[str, MetricValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EnvironmentStep(Generic[ObservationT]):
    observation: ObservationT
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict[str, MetricValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunResult:
    environment_steps: int
    agent_steps: int
    total_reward: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class SessionStep(Generic[ObservationT, ActionT]):
    transition: EnvironmentStep[ObservationT]
    agent: StepResult[ActionT]
