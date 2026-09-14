"""Composable contracts behind a FastConnectome agent."""

from pathlib import Path
from typing import Protocol, TypeVar

from fastconnectome.types import DecodedAction, EnvironmentStep, ModelInfo, SimulationResult

ObservationT_contra = TypeVar("ObservationT_contra", contravariant=True)
EnvironmentObservationT = TypeVar("EnvironmentObservationT")
StimulusT_contra = TypeVar("StimulusT_contra", contravariant=True)
StimulusT_co = TypeVar("StimulusT_co", covariant=True)
ReinforcementT_contra = TypeVar("ReinforcementT_contra", contravariant=True)
ReinforcementT_co = TypeVar("ReinforcementT_co", covariant=True)
ActivityT_contra = TypeVar("ActivityT_contra", contravariant=True)
ActivityT = TypeVar("ActivityT")
ActionT_contra = TypeVar("ActionT_contra", contravariant=True)
ActionT = TypeVar("ActionT")


class ObservationEncoder(Protocol[ObservationT_contra, StimulusT_co]):
    def encode(self, observation: ObservationT_contra) -> StimulusT_co: ...


class ReinforcementEncoder(Protocol[ReinforcementT_co]):
    def encode(self, reward: float) -> ReinforcementT_co: ...


class Simulator(Protocol[StimulusT_contra, ReinforcementT_contra, ActivityT]):
    @property
    def info(self) -> ModelInfo: ...

    def advance(
        self,
        stimulus: StimulusT_contra,
        reinforcement: ReinforcementT_contra,
        duration_ms: float,
    ) -> SimulationResult[ActivityT]: ...

    def reset(self, *, keep_learning: bool = False) -> None: ...

    def save(self, path: Path) -> None: ...

    def restore(self, path: Path) -> None: ...


class ActionDecoder(Protocol[ActivityT_contra, ActionT]):
    def decode(
        self, activity: ActivityT_contra, duration_ms: float
    ) -> DecodedAction[ActionT]: ...

    def reset(self) -> None: ...


class Environment(Protocol[EnvironmentObservationT, ActionT_contra]):
    def reset(self) -> EnvironmentObservationT: ...

    def step(
        self, action: ActionT_contra
    ) -> EnvironmentStep[EnvironmentObservationT]: ...
