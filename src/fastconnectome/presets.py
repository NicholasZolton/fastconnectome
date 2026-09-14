"""Auditable presets that bind model, adapters, and timing choices."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fastconnectome.agent import Agent
from fastconnectome.artifacts import float_value, object_dict, read_npz_manifest, string_value
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
    learning: bool = True,
) -> Agent[RGBFrame, RGBFrame, CurrentPulse | None, NeuralActivity, Turn]:
    if name != "malecns-visual-turning":
        raise ValueError(f"Unknown preset {name!r}")
    if dynamics != "stonkfly-v1":
        raise ValueError(f"Unsupported dynamics {dynamics!r}")
    if backend != "cpu":
        raise ValueError(f"Unsupported backend {backend!r}")
    return Agent(
        simulator=MaleCNS(data_dir, learning=learning),
        observation=CompoundEye(),
        action=BilateralTurn(),
        reinforcement=DopamineValence(),
        neural_ms=20.0,
    )


def load_policy(
    path: Path,
    *,
    data_dir: Path,
    backend: str,
) -> Agent[RGBFrame, RGBFrame, CurrentPulse | None, NeuralActivity, Turn]:
    manifest = read_npz_manifest(path)
    if manifest.get("schema") != 1 or manifest.get("kind") != "fastconnectome-policy":
        raise ValueError("Unsupported FastConnectome policy artifact")
    if backend != "cpu":
        raise ValueError(f"Unsupported backend {backend!r}")
    model = object_dict(manifest.get("model"), "model")
    if (
        model.get("model") != "malecns"
        or model.get("release") != "v1.0"
        or model.get("dynamics") != "stonkfly-v1"
        or model.get("backend") != backend
    ):
        raise ValueError("Unsupported policy model or backend")

    observation_configuration = object_dict(
        manifest.get("observation"), "observation"
    )
    if observation_configuration != {"type": "compound-eye-v1"}:
        raise ValueError("Unsupported observation adapter")

    simulator_configuration = object_dict(manifest.get("simulator"), "simulator")
    if simulator_configuration.get("type") != "malecns-stonkfly-v1":
        raise ValueError("Unsupported simulator")
    neural_bin_ms = float_value(
        simulator_configuration.get("neural_bin_ms"), "simulator.neural_bin_ms"
    )

    action_configuration = object_dict(manifest.get("action"), "action")
    if action_configuration.get("type") != "bilateral-turn-v1":
        raise ValueError("Unsupported action decoder")
    gate_raw = action_configuration.get("gate_cell_type")
    if gate_raw is not None and not isinstance(gate_raw, str):
        raise ValueError("action.gate_cell_type must be a string")
    action = BilateralTurn(
        cell_type=string_value(action_configuration.get("cell_type"), "action.cell_type"),
        left_side=string_value(action_configuration.get("left_side"), "action.left_side"),
        right_side=string_value(
            action_configuration.get("right_side"), "action.right_side"
        ),
        window_ms=float_value(action_configuration.get("window_ms"), "action.window_ms"),
        deadband_hz=float_value(
            action_configuration.get("deadband_hz"), "action.deadband_hz"
        ),
        gate_cell_type=gate_raw,
    )

    reinforcement_configuration = object_dict(
        manifest.get("reinforcement"), "reinforcement"
    )
    if reinforcement_configuration.get("type") != "dopamine-valence-v1":
        raise ValueError("Unsupported reinforcement adapter")
    reinforcement = DopamineValence(
        positive=string_value(
            reinforcement_configuration.get("positive"), "reinforcement.positive"
        ),
        negative=string_value(
            reinforcement_configuration.get("negative"), "reinforcement.negative"
        ),
        duration_ms=float_value(
            reinforcement_configuration.get("duration_ms"),
            "reinforcement.duration_ms",
        ),
        current=float_value(
            reinforcement_configuration.get("current"), "reinforcement.current"
        ),
        deadband=float_value(
            reinforcement_configuration.get("deadband"), "reinforcement.deadband"
        ),
    )

    simulator = MaleCNS(
        data_dir,
        learning=False,
        neural_bin_ms=neural_bin_ms,
    )
    agent = Agent(
        simulator=simulator,
        observation=CompoundEye(),
        action=action,
        reinforcement=reinforcement,
        neural_ms=float_value(manifest.get("neural_ms"), "neural_ms"),
    )
    agent._import_policy(path)
    return agent
