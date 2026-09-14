"""Learn a one-step Pong alignment policy from frozen MaleCNS activity."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fastconnectome.models.malecns import MaleCNS, MetalMaleCNS, Turn
from fastconnectome.presets import resolve_backend
from fastconnectome.types import ModelInfo

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
WIDTH = 320
HEIGHT = 180
PADDLE_CENTER_X = WIDTH // 2
POSITIONS = np.asarray(
    [32, 48, 64, 80, 96, 112, 128, 144, 176, 192, 208, 224, 240, 256, 272, 288],
    dtype=np.int32,
)
HELDOUT_POSITIONS = frozenset({48, 80, 240, 272})
ACTIONS = (Turn.LEFT, Turn.RIGHT)


def render_alignment_frame(ball_x: int) -> NDArray[np.uint8]:
    """Render a high-contrast ball and a fixed central paddle."""

    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[78:103, ball_x - 12 : ball_x + 13] = (245, 245, 245)
    frame[148:160, 136:184] = (40, 80, 255)
    return frame


def expected_action(ball_x: int) -> Turn:
    return Turn.LEFT if ball_x < PADDLE_CENTER_X else Turn.RIGHT


@dataclass(frozen=True, slots=True)
class FeatureTransform:
    indices: NDArray[np.int64]
    mean: FloatArray
    scale: FloatArray

    @classmethod
    def fit(cls, counts: IntArray) -> "FeatureTransform":
        logged = np.log1p(counts.astype(np.float64))
        mean = logged.mean(axis=0)
        spread = logged.std(axis=0)
        indices = np.flatnonzero(spread > 1e-8).astype(np.int64)
        if not len(indices):
            raise RuntimeError("Training images produced no distinguishable neural features")
        return cls(
            indices=indices,
            mean=mean[indices],
            scale=np.maximum(spread[indices], 0.25),
        )

    def encode(self, counts: IntArray) -> FloatArray:
        if counts.ndim != 2 or counts.shape[1] <= int(self.indices.max()):
            raise ValueError("Neural feature shape does not match the readout")
        logged = np.log1p(counts[:, self.indices].astype(np.float64))
        normalized = (logged - self.mean) / self.scale
        return np.column_stack((normalized, np.ones(len(normalized))))


class ReinforceReadout:
    """A two-action softmax readout with explicit reward-to-action credit."""

    def __init__(
        self,
        feature_count: int,
        *,
        learning_rate: float = 0.03,
        seed: int = 4,
        weights: FloatArray | None = None,
    ) -> None:
        if feature_count <= 0 or learning_rate <= 0:
            raise ValueError("Invalid readout configuration")
        self.learning_rate = learning_rate
        self.weights = (
            np.zeros((len(ACTIONS), feature_count), dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64).copy()
        )
        if self.weights.shape != (len(ACTIONS), feature_count):
            raise ValueError("Readout weight shape mismatch")
        self._rng = np.random.default_rng(seed)
        self._reward_baseline = 0.0

    def probabilities(self, features: FloatArray) -> FloatArray:
        logits = self.weights @ features
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        return probabilities

    def choose(self, features: FloatArray, *, explore: bool) -> int:
        probabilities = self.probabilities(features)
        if explore:
            return int(self._rng.choice(len(ACTIONS), p=probabilities))
        return int(np.argmax(probabilities))

    def reinforce(self, features: FloatArray, action: int, reward: float) -> None:
        probabilities = self.probabilities(features)
        advantage = reward - self._reward_baseline
        self._reward_baseline = 0.99 * self._reward_baseline + 0.01 * reward
        action_gradient = -probabilities
        action_gradient[action] += 1
        self.weights += (
            self.learning_rate
            * advantage
            * action_gradient[:, np.newaxis]
            * features[np.newaxis, :]
        )


def extract_counts(
    data_dir: Path,
    backend: str,
    positions: NDArray[np.int32],
    neural_ms: float,
) -> tuple[IntArray, ModelInfo]:
    selected_backend = resolve_backend(backend)
    simulator: MaleCNS | MetalMaleCNS
    if selected_backend == "metal":
        simulator = MetalMaleCNS(data_dir, learning=False)
    else:
        simulator = MaleCNS(data_dir, learning=False)
    rows: list[IntArray] = []
    try:
        for ball_x in positions:
            simulator.reset()
            result = simulator.advance(
                render_alignment_frame(int(ball_x)),
                None,
                neural_ms,
            )
            rows.append(result.activity.counts.copy())
        return np.stack(rows), simulator.info
    finally:
        if isinstance(simulator, MetalMaleCNS):
            simulator.close()


def accuracy(
    policy: ReinforceReadout,
    features: FloatArray,
    positions: NDArray[np.int32],
) -> float:
    correct = 0
    for row, ball_x in zip(features, positions, strict=True):
        action = ACTIONS[policy.choose(row, explore=False)]
        correct += int(action == expected_action(int(ball_x)))
    return correct / len(positions)


def save_readout(
    path: Path,
    policy: ReinforceReadout,
    transform: FeatureTransform,
    info: ModelInfo,
    neural_ms: float,
) -> None:
    manifest = {
        "schema": 1,
        "kind": "fastconnectome-reinforce-readout",
        "model": asdict(info),
        "neural_ms": neural_ms,
        "actions": [action.value for action in ACTIONS],
        "connectome_learning": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            manifest=json.dumps(manifest, allow_nan=False),
            weights=policy.weights,
            feature_indices=transform.indices,
            feature_mean=transform.mean,
            feature_scale=transform.scale,
        )


def load_readout(path: Path) -> tuple[ReinforceReadout, FeatureTransform]:
    with np.load(path, allow_pickle=False) as archive:
        raw: object = json.loads(str(archive["manifest"]))
        weights = archive["weights"]
        indices = archive["feature_indices"]
        mean = archive["feature_mean"]
        scale = archive["feature_scale"]
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != 1
        or raw.get("kind") != "fastconnectome-reinforce-readout"
        or raw.get("actions") != [action.value for action in ACTIONS]
        or raw.get("connectome_learning") is not False
        or indices.ndim != 1
        or indices.dtype != np.int64
        or mean.shape != indices.shape
        or scale.shape != indices.shape
        or weights.shape != (len(ACTIONS), len(indices) + 1)
        or not np.isfinite(weights).all()
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("Invalid learned-readout artifact")
    transform = FeatureTransform(indices, mean, scale)
    policy = ReinforceReadout(weights.shape[1], weights=weights)
    return policy, transform


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an action-aware readout on frozen full-connectome features"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs/pong-readout.npz"))
    parser.add_argument("--backend", choices=["auto", "cpu", "metal"], default="auto")
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--neural-ms", type=float, default=100.0)
    args = parser.parse_args()
    if args.updates <= 0 or args.neural_ms <= 0:
        parser.error("updates and neural-ms must be positive")

    training_mask = np.asarray(
        [int(position) not in HELDOUT_POSITIONS for position in POSITIONS],
        dtype=np.bool_,
    )
    training_positions = POSITIONS[training_mask]
    heldout_positions = POSITIONS[~training_mask]

    print("Encoding training and held-out images through the full connectome...")
    counts, info = extract_counts(args.data_dir, args.backend, POSITIONS, args.neural_ms)
    transform = FeatureTransform.fit(counts[training_mask])
    features = transform.encode(counts)
    training_features = features[training_mask]
    heldout_features = features[~training_mask]
    policy = ReinforceReadout(
        features.shape[1],
        seed=args.seed,
    )
    before = accuracy(policy, heldout_features, heldout_positions)

    state_rng = np.random.default_rng(args.seed + 1)
    rewards: list[float] = []
    for _ in range(args.updates):
        row = int(state_rng.integers(len(training_positions)))
        feature = training_features[row]
        action_index = policy.choose(feature, explore=True)
        reward = (
            1.0
            if ACTIONS[action_index] == expected_action(int(training_positions[row]))
            else -1.0
        )
        policy.reinforce(feature, action_index, reward)
        rewards.append(reward)

    training_accuracy = accuracy(policy, training_features, training_positions)
    heldout_accuracy = accuracy(policy, heldout_features, heldout_positions)
    save_readout(args.output, policy, transform, info, args.neural_ms)

    print("Loading the frozen readout and re-encoding held-out images...")
    deployed_policy, deployed_transform = load_readout(args.output)
    deployed_counts, _ = extract_counts(
        args.data_dir,
        args.backend,
        heldout_positions,
        args.neural_ms,
    )
    deployed_accuracy = accuracy(
        deployed_policy,
        deployed_transform.encode(deployed_counts),
        heldout_positions,
    )
    tail = rewards[-min(100, len(rewards)) :]

    print(f"backend: {info.backend}; varying neural features: {len(transform.indices):,}")
    print(f"held-out accuracy before training: {before:.0%}")
    print(f"training-position accuracy: {training_accuracy:.0%}")
    print(f"held-out accuracy after training: {heldout_accuracy:.0%}")
    print(f"fresh frozen deployment accuracy: {deployed_accuracy:.0%}")
    print(f"mean reward over final {len(tail)} updates: {np.mean(tail):+.2f}")
    print(f"readout: {args.output}")
    print("Connectome weights stayed frozen; only the action-aware readout learned.")


if __name__ == "__main__":
    main()
