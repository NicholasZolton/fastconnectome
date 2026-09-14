"""Stateful agent composed from explicit connectome adapters."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from types import TracebackType
from typing import Generic, Literal, overload, TYPE_CHECKING, TypeVar
import zipfile

from fastconnectome.protocols import (
    ActionDecoder,
    Checkpointable,
    Closable,
    Configurable,
    ObservationEncoder,
    PolicyArtifact,
    ReinforcementEncoder,
    Simulator,
)
from fastconnectome.types import ModelInfo, StepResult

if TYPE_CHECKING:
    from numpy import uint8
    from numpy.typing import NDArray

    from fastconnectome.models.malecns import (
        ApproachChoice,
        CurrentPulse,
        KCCue,
        MaleCNSStimulus,
        NeuralActivity,
        Turn,
    )

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

    def close(self) -> None:
        """Release resources held by execution backends such as Metal."""

        if isinstance(self._simulator, Closable):
            self._simulator.close()

    def __enter__(self) -> Agent[
        ObservationT,
        StimulusT,
        ReinforcementT,
        ActivityT,
        ActionT,
    ]:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def save(self, path: str | Path) -> None:
        if not isinstance(self._action, Checkpointable):
            raise TypeError("The action decoder does not support checkpoints")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
            root = Path(directory)
            self._simulator.save(root / "simulator.npz")
            self._action.save(root / "decoder.npz")
            manifest = self._manifest("fastconnectome-agent-checkpoint")
            (root / "manifest.json").write_text(
                json.dumps(manifest, indent=2, allow_nan=False) + "\n"
            )
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_STORED
            ) as archive:
                for name in ("manifest.json", "simulator.npz", "decoder.npz"):
                    archive.write(root / name, name)
        os.replace(temporary, destination)

    def restore(self, path: str | Path) -> None:
        if not isinstance(self._action, Checkpointable):
            raise TypeError("The action decoder does not support checkpoints")
        source = Path(path)
        with tempfile.TemporaryDirectory(dir=source.parent) as directory:
            root = Path(directory)
            with zipfile.ZipFile(source, "r") as archive:
                expected = {"manifest.json", "simulator.npz", "decoder.npz"}
                if set(archive.namelist()) != expected:
                    raise ValueError("Unexpected agent checkpoint contents")
                for name in expected:
                    (root / name).write_bytes(archive.read(name))
            raw: object = json.loads((root / "manifest.json").read_text())
            if raw != self._manifest("fastconnectome-agent-checkpoint"):
                raise ValueError("Agent checkpoint configuration mismatch")
            self._simulator.restore(root / "simulator.npz")
            self._action.restore(root / "decoder.npz")

    def export_policy(self, path: str | Path) -> None:
        if not isinstance(self._simulator, PolicyArtifact):
            raise TypeError("The simulator does not support policy artifacts")
        self._simulator.export_policy(
            Path(path), self._manifest("fastconnectome-policy")
        )

    def _import_policy(self, path: Path) -> None:
        if not isinstance(self._simulator, PolicyArtifact):
            raise TypeError("The simulator does not support policy artifacts")
        self._simulator.import_policy(
            path, self._manifest("fastconnectome-policy")
        )

    def _manifest(self, kind: str) -> dict[str, object]:
        return {
            "schema": 1,
            "kind": kind,
            "model": asdict(self.info),
            "neural_ms": self.neural_ms,
            "observation": self._configuration(self._observation, "observation"),
            "action": self._configuration(self._action, "action"),
            "reinforcement": self._configuration(
                self._reinforcement, "reinforcement"
            ),
            "simulator": self._configuration(self._simulator, "simulator"),
        }

    @staticmethod
    def _configuration(component: object, role: str) -> dict[str, object]:
        if not isinstance(component, Configurable):
            raise TypeError(f"The {role} component does not expose its configuration")
        return dict(component.configuration())

    @overload
    @classmethod
    def from_preset(
        cls,
        name: Literal["malecns-visual-turning"],
        *,
        data_dir: str | Path = Path("data"),
        dynamics: str = "stonkfly-v1",
        backend: str = "auto",
        learning: bool = True,
    ) -> VisualTurningPreset: ...

    @overload
    @classmethod
    def from_preset(
        cls,
        name: Literal["malecns-kc-conditioning"],
        *,
        data_dir: str | Path = Path("data"),
        dynamics: str = "stonkfly-v1",
        backend: str = "auto",
        learning: bool = True,
    ) -> KCConditioningPreset: ...

    @classmethod
    def from_preset(
        cls,
        name: str,
        *,
        data_dir: str | Path = Path("data"),
        dynamics: str = "stonkfly-v1",
        backend: str = "auto",
        learning: bool = True,
    ) -> PresetAgent:
        from fastconnectome.presets import build_preset

        return build_preset(
            name,
            data_dir=Path(data_dir),
            dynamics=dynamics,
            backend=backend,
            learning=learning,
        )

    @overload
    @classmethod
    def load_policy(
        cls,
        path: str | Path,
        *,
        data_dir: str | Path = Path("data"),
        backend: str = "auto",
        preset: Literal["malecns-visual-turning"] = "malecns-visual-turning",
    ) -> VisualTurningPreset: ...

    @overload
    @classmethod
    def load_policy(
        cls,
        path: str | Path,
        *,
        data_dir: str | Path = Path("data"),
        backend: str = "auto",
        preset: Literal["malecns-kc-conditioning"],
    ) -> KCConditioningPreset: ...

    @classmethod
    def load_policy(
        cls,
        path: str | Path,
        *,
        data_dir: str | Path = Path("data"),
        backend: str = "auto",
        preset: str = "malecns-visual-turning",
    ) -> PresetAgent:
        from fastconnectome.presets import load_policy

        return load_policy(
            Path(path),
            data_dir=Path(data_dir),
            backend=backend,
            preset=preset,
        )


if TYPE_CHECKING:
    VisualTurningPreset = Agent[
        NDArray[uint8],
        NDArray[uint8],
        CurrentPulse | None,
        NeuralActivity,
        Turn,
    ]
    KCConditioningPreset = Agent[
        KCCue,
        MaleCNSStimulus,
        CurrentPulse | None,
        NeuralActivity,
        ApproachChoice,
    ]
    PresetAgent = VisualTurningPreset | KCConditioningPreset
