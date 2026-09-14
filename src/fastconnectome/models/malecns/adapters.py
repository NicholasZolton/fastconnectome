"""Sensory, motor, and reinforcement adapters for MaleCNS."""

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
import math

import numpy as np
from numpy.typing import NDArray

from fastconnectome.models.malecns.simulator import CurrentPulse, NeuralActivity
from fastconnectome.types import DecodedAction

RGBFrame = NDArray[np.uint8]


class Turn(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    HOLD = "hold"


class CompoundEye:
    """Validate RGB display input consumed by Stonkfly's retinal projection."""

    def encode(self, observation: RGBFrame) -> RGBFrame:
        if (
            observation.ndim != 3
            or observation.shape[2] != 3
            or observation.dtype != np.uint8
        ):
            raise ValueError("CompoundEye requires an H×W×3 uint8 RGB frame")
        return observation


@dataclass(frozen=True, slots=True)
class DopamineValence:
    positive: str = "PAM11"
    negative: str = "PPL101"
    duration_ms: float = 50.0
    current: float = 20.0
    deadband: float = 0.0

    def __post_init__(self) -> None:
        if (
            self.duration_ms <= 0
            or not math.isfinite(self.current)
            or self.current <= 0
            or not math.isfinite(self.deadband)
            or self.deadband < 0
        ):
            raise ValueError("Invalid dopamine reinforcement configuration")

    def encode(self, reward: float) -> CurrentPulse | None:
        if not math.isfinite(reward):
            raise ValueError("Reward must be finite")
        if reward > self.deadband:
            return CurrentPulse(
                self.positive, self.duration_ms, self.current, "positive"
            )
        if reward < -self.deadband:
            return CurrentPulse(
                self.negative, self.duration_ms, self.current, "negative"
            )
        return None


class BilateralTurn:
    """Decode a rolling left/right firing-rate difference with a dead zone."""

    def __init__(
        self,
        cell_type: str = "DNp20",
        *,
        left_side: str = "L",
        right_side: str = "R",
        window_ms: float = 200.0,
        deadband_hz: float = 2.0,
        gate_cell_type: str | None = None,
    ) -> None:
        if window_ms <= 0 or deadband_hz < 0:
            raise ValueError("Invalid bilateral decoder configuration")
        self.cell_type = cell_type
        self.left_side = left_side
        self.right_side = right_side
        self.window_ms = window_ms
        self.deadband_hz = deadband_hz
        self.gate_cell_type = gate_cell_type
        self._left = np.array([], dtype=np.int32)
        self._right = np.array([], dtype=np.int32)
        self._gate = np.array([], dtype=np.int32)
        self._samples: deque[tuple[int, int, int, float]] = deque()
        self._sample_ms = 0.0

    def _resolve(self, activity: NeuralActivity) -> None:
        if len(self._left):
            return
        self._left = activity.populations.select(self.cell_type, self.left_side)
        self._right = activity.populations.select(self.cell_type, self.right_side)
        if not len(self._left) or not len(self._right):
            raise ValueError(f"Missing bilateral {self.cell_type} populations")
        if self.gate_cell_type is not None:
            self._gate = activity.populations.select(self.gate_cell_type)
            if not len(self._gate):
                raise ValueError(f"Missing gate population {self.gate_cell_type}")

    def decode(
        self, activity: NeuralActivity, duration_ms: float
    ) -> DecodedAction[Turn]:
        if duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        self._resolve(activity)
        counts = activity.counts
        sample = (
            int(counts[self._left].sum()),
            int(counts[self._right].sum()),
            int(counts[self._gate].sum()) if len(self._gate) else 0,
            duration_ms,
        )
        self._samples.append(sample)
        self._sample_ms += duration_ms
        while len(self._samples) > 1 and self._sample_ms > self.window_ms + 1e-9:
            self._sample_ms -= self._samples.popleft()[3]

        left_spikes = sum(current[0] for current in self._samples)
        right_spikes = sum(current[1] for current in self._samples)
        gate_spikes = sum(current[2] for current in self._samples)
        seconds = self._sample_ms / 1000.0
        left_hz = left_spikes / len(self._left) / seconds
        right_hz = right_spikes / len(self._right) / seconds
        difference_hz = right_hz - left_hz
        gated = self.gate_cell_type is None or gate_spikes > 0
        if not gated or abs(difference_hz) < self.deadband_hz:
            action = Turn.HOLD
        elif difference_hz > 0:
            action = Turn.RIGHT
        else:
            action = Turn.LEFT
        return DecodedAction(
            action,
            {
                "left_hz": left_hz,
                "right_hz": right_hz,
                "difference_hz": difference_hz,
                "gate_spikes": gate_spikes,
                "readout_window_ms": self._sample_ms,
            },
        )

    def reset(self) -> None:
        self._samples.clear()
        self._sample_ms = 0.0
