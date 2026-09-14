"""Reference MaleCNS simulator backed by Stonkfly's native CPU kernel."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from fastconnectome.types import (
    ActivitySummary,
    LearningSummary,
    ModelInfo,
    SimulationResult,
    Timing,
)

if TYPE_CHECKING:
    from stonkfly.neural.visual import VisualMemoryBrain

RGBFrame = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class CurrentPulse:
    population: str
    duration_ms: float
    current: float
    label: str


@dataclass(frozen=True, slots=True)
class PopulationIndex:
    ids: NDArray[np.int64]
    cell_types: NDArray[np.str_]
    sides: NDArray[np.str_]

    def select(self, cell_type: str, side: str | None = None) -> NDArray[np.int32]:
        selected = self.cell_types == cell_type
        if side is not None:
            selected &= self.sides == side
        return np.flatnonzero(selected).astype(np.int32)


@dataclass(frozen=True, slots=True)
class NeuralActivity:
    counts: NDArray[np.int32]
    populations: PopulationIndex


class MaleCNS:
    """The retained 166,700-neuron MaleCNS graph with Stonkfly v1 dynamics."""

    def __init__(
        self,
        data_dir: str | Path = Path("data"),
        *,
        learning: bool = True,
        neural_bin_ms: float = 10.0,
    ) -> None:
        if neural_bin_ms <= 0 or neural_bin_ms > 10:
            raise ValueError("neural_bin_ms must be in (0, 10]")
        self.data_dir = Path(data_dir).resolve()
        self.learning = learning
        self.neural_bin_ms = neural_bin_ms
        required = [self.data_dir / "graph.npz", self.data_dir / "annotations.feather"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing prepared MaleCNS files: "
                + ", ".join(missing)
                + ". Run `fastconnectome prepare malecns-v1`."
            )

        os.environ["STONKFLY_DATA"] = str(self.data_dir)
        try:
            from stonkfly.neural.common import DATA, annotations
            from stonkfly.neural.visual import VisualMemoryBrain
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "MaleCNS support is optional; install `fastconnectome[malecns]`"
            ) from error
        if DATA != self.data_dir:
            raise RuntimeError("Stonkfly was imported with another data directory")

        self._brain: VisualMemoryBrain = VisualMemoryBrain(
            path=self.data_dir / "graph.npz"
        )
        self._brain.weights_frozen = not learning
        with np.load(self.data_dir / "graph.npz") as graph:
            ids = graph["ids"].astype(np.int64)
            connection_count = len(graph["post"])
        annotations_frame = annotations(ids)
        self.populations = PopulationIndex(
            ids=ids,
            cell_types=np.asarray(
                annotations_frame.type.fillna("").astype(str), dtype=np.str_
            ),
            sides=np.asarray(
                annotations_frame.somaSide.fillna("").astype(str), dtype=np.str_
            ),
        )
        self._pending_pulse: CurrentPulse | None = None
        self._pulse_remaining_ms = 0.0
        self._pulse_indices = np.array([], dtype=np.int32)
        self._info = ModelInfo(
            model="malecns",
            release="v1.0",
            dynamics="stonkfly-v1",
            backend="cpu",
            neurons=self._brain.n,
            connections=connection_count,
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    def advance(
        self,
        stimulus: RGBFrame,
        reinforcement: CurrentPulse | None,
        duration_ms: float,
    ) -> SimulationResult[NeuralActivity]:
        if duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        if reinforcement is not None:
            indices = self.populations.select(reinforcement.population)
            if not len(indices):
                raise ValueError(f"Unknown reinforcement population {reinforcement.population!r}")
            self._pending_pulse = reinforcement
            self._pulse_remaining_ms = reinforcement.duration_ms
            self._pulse_indices = indices

        counts = np.zeros(self._brain.n, dtype=np.int32)
        compute_seconds = 0.0
        remaining_ms = duration_ms
        applied_label = (
            self._pending_pulse.label if self._pending_pulse is not None else "none"
        )
        while remaining_ms > 1e-12:
            interval_ms = min(self.neural_bin_ms, remaining_ms)
            stimulation: tuple[NDArray[np.int32], float] | None = None
            if self._pending_pulse is not None and self._pulse_remaining_ms > 0:
                interval_ms = min(interval_ms, self._pulse_remaining_ms)
                stimulation = (self._pulse_indices, self._pending_pulse.current)
            current_counts, elapsed = self._brain.rgb_step(
                stimulus,
                interval_ms,
                learning=self.learning,
                stimulation=stimulation,
            )
            counts += current_counts
            compute_seconds += elapsed
            remaining_ms -= interval_ms
            if self._pending_pulse is not None:
                self._pulse_remaining_ms = max(
                    0.0, self._pulse_remaining_ms - interval_ms
                )
                if self._pulse_remaining_ms == 0:
                    self._pending_pulse = None
                    self._pulse_indices = np.array([], dtype=np.int32)

        memory = self._brain.memory()
        selected_spikes = {
            "KC": int(counts[self._brain.circuit["kc"]].sum()),
            "PAM11": int(counts[self._brain.circuit["reward"]].sum()),
            "PPL101": int(counts[self._brain.circuit["aversive"]].sum()),
        }
        return SimulationResult(
            activity=NeuralActivity(counts, self.populations),
            activity_summary=ActivitySummary(int(counts.sum()), selected_spikes),
            learning_summary=LearningSummary(
                enabled=self.learning,
                changed_connections=int(memory["changed_edges"]),
            ),
            timing=Timing(duration_ms, compute_seconds, compute_seconds),
            metrics={
                "reinforcement": applied_label,
                "pulse_remaining_ms": self._pulse_remaining_ms,
            },
        )

    def reset(self, *, keep_learning: bool = False) -> None:
        self._brain.reset(keep_memory=keep_learning)
        self._pending_pulse = None
        self._pulse_remaining_ms = 0.0
        self._pulse_indices = np.array([], dtype=np.int32)

    def save(self, path: Path) -> None:
        self._brain.checkpoint(path)

    def restore(self, path: Path) -> None:
        self._brain.restore(path)
        self._pending_pulse = None
        self._pulse_remaining_ms = 0.0
        self._pulse_indices = np.array([], dtype=np.int32)
