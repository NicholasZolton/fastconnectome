"""Stateful agent composed from explicit connectome adapters."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Generic, TYPE_CHECKING, TypeVar

from fastconnectome.protocols import (
    ActionDecoder,
    ObservationEncoder,
    ReinforcementEncoder,
    Simulator,
)
from fastconnectome.types import ModelInfo, StepResult

if TYPE_CHECKING:
    from numpy import uint8
    from numpy.typing import NDArray

    from fastconnectome.models.malecns import CurrentPulse, NeuralActivity
    from fastconnectome.models.malecns.adapters import Turn

ObservationT = TypeVar("ObservationT")
StimulusT = TypeVar("StimulusT")
ReinforcementT = TypeVar("ReinforcementT")
ActivityT = TypeVar("ActivityT")
ActionT = TypeVar("ActionT")


class Agent(Generic[ObservationT, StimulusT, ReinforcementT, ActivityT, ActionT]):
    """Advance a stateful simulator and decode its activity into an action."""

    def __init__(
        self,
        *,
        simulator: Simulator[StimulusT, ReinforcementT, ActivityT],
        observation: ObservationEncoder[ObservationT, StimulusT],
        action: ActionDecoder[ActivityT, ActionT],
        reinforcement: ReinforcementEncoder[ReinforcementT],
        neural_ms: float = 20.0,
    ) -> None:
        if neural_ms <= 0:
            raise ValueError("neural_ms must be positive")
        self._simulator = simulator
        self._observation = observation
        self._action = action
        self._reinforcement = reinforcement
        self.neural_ms = neural_ms

    @property
    def info(self) -> ModelInfo:
        return self._simulator.info

    def step(self, observation: ObservationT, *, reward: float = 0.0) -> StepResult[ActionT]:
        started = perf_counter()
        stimulus = self._observation.encode(observation)
        reinforcement = self._reinforcement.encode(reward)
        simulation = self._simulator.advance(stimulus, reinforcement, self.neural_ms)
        decoded = self._action.decode(simulation.activity, self.neural_ms)
        timing = replace(simulation.timing, wall_seconds=perf_counter() - started)
        return StepResult(
            action=decoded.action,
            activity=simulation.activity_summary,
            learning=simulation.learning_summary,
            timing=timing,
            metrics={**simulation.metrics, **decoded.metrics},
        )

    def reset(self, *, keep_learning: bool = False) -> None:
        self._simulator.reset(keep_learning=keep_learning)
        self._action.reset()

    def save(self, path: str | Path) -> None:
        self._simulator.save(Path(path))

    def restore(self, path: str | Path) -> None:
        self._simulator.restore(Path(path))
        self._action.reset()

    @classmethod
    def from_preset(
        cls,
        name: str,
        *,
        data_dir: str | Path = Path("data"),
        dynamics: str = "stonkfly-v1",
        backend: str = "cpu",
    ) -> Agent[
        NDArray[uint8],
        NDArray[uint8],
        CurrentPulse | None,
        NeuralActivity,
        Turn,
    ]:
        from fastconnectome.presets import build_preset

        return build_preset(
            name,
            data_dir=Path(data_dir),
            dynamics=dynamics,
            backend=backend,
        )
