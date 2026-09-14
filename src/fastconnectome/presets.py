"""Auditable presets that bind model, adapters, and timing choices."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fastconnectome.agent import Agent
from fastconnectome.models.malecns import (
    BilateralTurn,
    CompoundEye,
    CurrentPulse,
    DopamineValence,
    MaleCNS,
    NeuralActivity,
    Turn,
)

RGBFrame = NDArray[np.uint8]


def build_preset(
    name: str,
    *,
    data_dir: Path,
    dynamics: str,
    backend: str,
) -> Agent[RGBFrame, RGBFrame, CurrentPulse | None, NeuralActivity, Turn]:
    if name != "malecns-visual-turning":
        raise ValueError(f"Unknown preset {name!r}")
    if dynamics != "stonkfly-v1":
        raise ValueError(f"Unsupported dynamics {dynamics!r}")
    if backend != "cpu":
        raise ValueError(f"Unsupported backend {backend!r}")
    return Agent(
        simulator=MaleCNS(data_dir),
        observation=CompoundEye(),
        action=BilateralTurn(),
        reinforcement=DopamineValence(),
        neural_ms=20.0,
    )
