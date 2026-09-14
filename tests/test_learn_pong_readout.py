from pathlib import Path

import numpy as np

from learn_pong_readout import (
    ACTIONS,
    FeatureTransform,
    ReinforceReadout,
    load_readout,
    save_readout,
)
from fastconnectome.types import ModelInfo


def test_reinforce_readout_learns_action_specific_credit() -> None:
    features = np.asarray([[-1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    expected = [0, 1]
    policy = ReinforceReadout(2, learning_rate=0.05, seed=3)
    rng = np.random.default_rng(8)

    for _ in range(500):
        row = int(rng.integers(2))
        action = policy.choose(features[row], explore=True)
        reward = 1.0 if action == expected[row] else -1.0
        policy.reinforce(features[row], action, reward)

    assert [policy.choose(row, explore=False) for row in features] == expected


def test_readout_artifact_round_trip(tmp_path: Path) -> None:
    transform = FeatureTransform(
        indices=np.asarray([1, 4], dtype=np.int64),
        mean=np.asarray([0.5, 1.5], dtype=np.float64),
        scale=np.asarray([0.25, 0.75], dtype=np.float64),
    )
    weights = np.arange(6, dtype=np.float64).reshape(2, 3)
    policy = ReinforceReadout(3, weights=weights)
    path = tmp_path / "readout.npz"

    save_readout(
        path,
        policy,
        transform,
        ModelInfo("fixture", "1", "test", "cpu", 5, 4),
        100.0,
    )
    restored_policy, restored_transform = load_readout(path)

    assert np.array_equal(restored_policy.weights, weights)
    assert np.array_equal(restored_transform.indices, transform.indices)
    assert np.array_equal(restored_transform.mean, transform.mean)
    assert np.array_equal(restored_transform.scale, transform.scale)
    assert len(ACTIONS) == 2
