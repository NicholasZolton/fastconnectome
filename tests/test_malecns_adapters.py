from pathlib import Path

import numpy as np
import pytest

from fastconnectome.models.malecns import (
    BilateralTurn,
    CompoundEye,
    DopamineValence,
    NeuralActivity,
    PopulationIndex,
    Turn,
)


def activity(counts: list[int]) -> NeuralActivity:
    populations = PopulationIndex(
        ids=np.array([10, 11, 12], dtype=np.int64),
        cell_types=np.array(["DNp20", "DNp20", "DNpe017"], dtype=np.str_),
        sides=np.array(["L", "R", ""], dtype=np.str_),
    )
    return NeuralActivity(np.array(counts, dtype=np.int32), populations)


def test_compound_eye_requires_rgb_uint8() -> None:
    eye = CompoundEye()
    frame = np.zeros((10, 20, 3), dtype=np.uint8)

    assert eye.encode(frame) is frame
    with pytest.raises(ValueError, match="RGB"):
        eye.encode(np.zeros((10, 20), dtype=np.uint8))


def test_dopamine_valence_maps_reward_sign_to_named_pulses() -> None:
    reinforcement = DopamineValence(duration_ms=50, current=20)
    positive = reinforcement.encode(1)
    negative = reinforcement.encode(-1)

    assert reinforcement.encode(0) is None
    assert positive is not None and positive.population == "PAM11"
    assert negative is not None and negative.population == "PPL101"


def test_bilateral_turn_uses_rolling_firing_rates() -> None:
    decoder = BilateralTurn(window_ms=200, deadband_hz=2)

    first = decoder.decode(activity([0, 1, 0]), 100)
    second = decoder.decode(activity([2, 0, 0]), 100)

    assert first.action is Turn.RIGHT
    assert second.action is Turn.LEFT
    assert second.metrics["readout_window_ms"] == 200


def test_bilateral_turn_checkpoint_preserves_readout_window(tmp_path: Path) -> None:
    original = BilateralTurn(window_ms=200, deadband_hz=2)
    original.decode(activity([0, 1, 0]), 100)
    checkpoint = tmp_path / "decoder.npz"
    original.save(checkpoint)

    restored = BilateralTurn(window_ms=200, deadband_hz=2)
    restored.restore(checkpoint)

    assert restored.decode(activity([2, 0, 0]), 100) == original.decode(
        activity([2, 0, 0]), 100
    )
