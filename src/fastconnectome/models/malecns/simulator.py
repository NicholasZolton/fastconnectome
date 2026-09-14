"""Reference MaleCNS simulator backed by Stonkfly's native CPU kernel."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import numpy as np
from numpy.typing import NDArray

from fastconnectome.types import (
    ActivitySummary,
    LearningSummary,
    ModelInfo,
    SimulationResult,
    Timing,
)

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

        # Stonkfly is an optional, untyped runtime dependency whose arrays are
        # validated immediately below and by its own constructor.
        self._brain: Any = VisualMemoryBrain(
            path=self.data_dir / "graph.npz"
        )
        self._brain.weights_frozen = not learning
        with np.load(self.data_dir / "graph.npz") as graph:
            ids = graph["ids"].astype(np.int64)
            connection_count = len(graph["post"])
            self._graph_hashes = {
                "graph_ids_sha256": self._array_digest(ids),
                "graph_ptr_sha256": self._array_digest(graph["ptr"]),
                "graph_post_sha256": self._array_digest(graph["post"]),
            }
        raw_weight = getattr(self._brain, "weight", None)
        if not isinstance(raw_weight, np.ndarray) or raw_weight.dtype != np.float32:
            raise RuntimeError("Stonkfly exposed an incompatible weight array")
        self._weight: NDArray[np.float32] = np.asarray(raw_weight, dtype=np.float32)
        self._plastic_edges: NDArray[np.int64] = np.asarray(
            self._brain.circuit["edges"], dtype=np.int64
        )
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
        self._deployment_weights: NDArray[np.float32] | None = None
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

    def configuration(self) -> dict[str, str | float]:
        return {
            "type": "malecns-stonkfly-v1",
            "neural_bin_ms": self.neural_bin_ms,
        }

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
            current_counts, elapsed = self._rgb_step(
                stimulus,
                interval_ms,
                stimulation,
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

    def _rgb_step(
        self,
        stimulus: RGBFrame,
        duration_ms: float,
        stimulation: tuple[NDArray[np.int32], float] | None,
    ) -> tuple[NDArray[np.int32], float]:
        return self._brain.rgb_step(
            stimulus,
            duration_ms,
            learning=self.learning,
            stimulation=stimulation,
        )

    def reset(self, *, keep_learning: bool = False) -> None:
        if self._deployment_weights is None:
            self._brain.reset(keep_memory=keep_learning)
        else:
            self._brain.reset(keep_memory=False)
            self._weight[self._plastic_edges] = self._deployment_weights
            self._brain.memory_u.fill(0)
            self._brain.memory_w[:] = (
                self._deployment_weights / self._brain.baseline_plastic - 1
            )
            self._brain.weights_frozen = True
        self._pending_pulse = None
        self._pulse_remaining_ms = 0.0
        self._pulse_indices = np.array([], dtype=np.int32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        with tempfile.TemporaryDirectory(dir=path.parent) as directory:
            root = Path(directory)
            self._brain.checkpoint(root / "brain.npz")
            state: dict[str, object] = {
                "schema": 1,
                "learning": self.learning,
                "deployment": self._deployment_weights is not None,
                "pulse_remaining_ms": self._pulse_remaining_ms,
                "pulse": None,
            }
            if self._pending_pulse is not None:
                state["pulse"] = {
                    "population": self._pending_pulse.population,
                    "duration_ms": self._pending_pulse.duration_ms,
                    "current": self._pending_pulse.current,
                    "label": self._pending_pulse.label,
                }
            (root / "state.json").write_text(
                json.dumps(state, indent=2, allow_nan=False) + "\n"
            )
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_STORED
            ) as archive:
                archive.write(root / "brain.npz", "brain.npz")
                archive.write(root / "state.json", "state.json")
        os.replace(temporary, path)

    def restore(self, path: Path) -> None:
        with tempfile.TemporaryDirectory(dir=path.parent) as directory:
            root = Path(directory)
            with zipfile.ZipFile(path, "r") as archive:
                if set(archive.namelist()) != {"brain.npz", "state.json"}:
                    raise ValueError("Unexpected MaleCNS checkpoint contents")
                (root / "brain.npz").write_bytes(archive.read("brain.npz"))
                raw: object = json.loads(archive.read("state.json"))
            if not isinstance(raw, dict) or raw.get("schema") != 1:
                raise ValueError("Invalid MaleCNS checkpoint state")
            learning = raw.get("learning")
            deployment = raw.get("deployment")
            remaining = raw.get("pulse_remaining_ms")
            pulse_raw = raw.get("pulse")
            if (
                not isinstance(learning, bool)
                or not isinstance(deployment, bool)
                or not isinstance(remaining, (int, float))
            ):
                raise ValueError("Invalid MaleCNS checkpoint state")
            pulse_remaining_ms = float(remaining)
            if not np.isfinite(pulse_remaining_ms) or pulse_remaining_ms < 0:
                raise ValueError("Invalid reinforcement pulse duration")
            pulse = self._parse_pulse(pulse_raw)
            if (pulse is None) != (pulse_remaining_ms == 0):
                raise ValueError("Inconsistent reinforcement pulse state")
            self._brain.restore(root / "brain.npz")
            if learning == self._brain.weights_frozen:
                raise ValueError("Inconsistent MaleCNS learning mode")
        self.learning = learning
        self._deployment_weights = (
            self._weight[self._plastic_edges].copy() if deployment else None
        )
        self._pending_pulse = pulse
        self._pulse_remaining_ms = pulse_remaining_ms
        self._pulse_indices = (
            np.array([], dtype=np.int32)
            if pulse is None
            else self._population_indices(pulse.population)
        )

    def export_policy(self, path: Path, manifest: dict[str, object]) -> None:
        edges = self._plastic_edges.copy()
        weights = self._weight[edges].copy()
        policy = {
            "changed_connections": int(
                np.count_nonzero(weights != self._brain.baseline_plastic)
            ),
            "weights_sha256": self._array_digest(weights),
        }
        artifact_manifest = {
            **manifest,
            "provenance": self._provenance(),
            "policy": policy,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                manifest=json.dumps(artifact_manifest, allow_nan=False),
                plastic_edges=edges,
                weights=weights,
            )
        os.replace(temporary, path)

    def import_policy(self, path: Path, manifest: dict[str, object]) -> None:
        with np.load(path, allow_pickle=False) as archive:
            raw: object = json.loads(str(archive["manifest"]))
            edges = archive["plastic_edges"]
            weights = archive["weights"]
        if not isinstance(raw, dict):
            raise ValueError("Invalid policy manifest")
        stored_manifest = {
            key: value
            for key, value in raw.items()
            if key not in {"policy", "provenance"}
        }
        if (
            not self._policy_manifest_matches(stored_manifest, manifest)
            or raw.get("provenance") != self._provenance()
        ):
            raise ValueError("Policy provenance or configuration mismatch")
        expected_edges = self._plastic_edges
        policy = raw.get("policy")
        if (
            not isinstance(policy, dict)
            or edges.dtype != np.int64
            or edges.shape != expected_edges.shape
            or not np.array_equal(edges, expected_edges)
            or weights.dtype != np.float32
            or weights.shape != self._brain.baseline_plastic.shape
            or not np.isfinite(weights).all()
        ):
            raise ValueError("Invalid learned-weight overlay")
        changed_connections = int(
            np.count_nonzero(weights != self._brain.baseline_plastic)
        )
        if (
            not isinstance(policy.get("changed_connections"), int)
            or isinstance(policy.get("changed_connections"), bool)
            or policy.get("changed_connections") != changed_connections
            or policy.get("weights_sha256") != self._array_digest(weights)
        ):
            raise ValueError("Invalid learned-weight overlay metadata")
        self._brain.reset(keep_memory=False)
        self._weight[expected_edges] = weights
        self._brain.memory_u.fill(0)
        self._brain.memory_w[:] = weights / self._brain.baseline_plastic - 1
        self._brain.weights_frozen = True
        self.learning = False
        self._deployment_weights = weights.copy()
        self._pending_pulse = None
        self._pulse_remaining_ms = 0.0
        self._pulse_indices = np.array([], dtype=np.int32)

    @staticmethod
    def _policy_manifest_matches(
        stored: Mapping[str, object], expected: Mapping[str, object]
    ) -> bool:
        stored_copy = dict(stored)
        expected_copy = dict(expected)
        stored_model = stored_copy.get("model")
        expected_model = expected_copy.get("model")
        if not isinstance(stored_model, dict) or not isinstance(expected_model, dict):
            return False
        stored_model_copy = dict(stored_model)
        expected_model_copy = dict(expected_model)
        stored_model_copy.pop("backend", None)
        expected_model_copy.pop("backend", None)
        stored_copy["model"] = stored_model_copy
        expected_copy["model"] = expected_model_copy
        return stored_copy == expected_copy

    def _provenance(self) -> dict[str, object]:
        return {
            **self._graph_hashes,
            "plastic_edges_sha256": self._array_digest(self._plastic_edges),
            "stonkfly_build": self._brain.build,
            "learning_rate": self._brain.eta,
            "configuration": self._brain.configuration_signature(),
        }

    @staticmethod
    def _array_digest(array: NDArray[np.generic]) -> str:
        contiguous = np.ascontiguousarray(array)
        return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()

    def _population_indices(self, population: str) -> NDArray[np.int32]:
        indices = self.populations.select(population)
        if not len(indices):
            raise ValueError(f"Unknown reinforcement population {population!r}")
        return indices

    @staticmethod
    def _parse_pulse(raw: object) -> CurrentPulse | None:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("Invalid reinforcement pulse")
        population = raw.get("population")
        duration = raw.get("duration_ms")
        current = raw.get("current")
        label = raw.get("label")
        if (
            not isinstance(population, str)
            or not isinstance(duration, (int, float))
            or not isinstance(current, (int, float))
            or not isinstance(label, str)
            or not np.isfinite(duration)
            or float(duration) <= 0
            or not np.isfinite(current)
        ):
            raise ValueError("Invalid reinforcement pulse")
        return CurrentPulse(population, float(duration), float(current), label)
