import json
from pathlib import Path

from fastconnectome import Agent, EnvironmentStep, TrainingSession
from fastconnectome.types import (
    ActivitySummary,
    DecodedAction,
    LearningSummary,
    ModelInfo,
    SimulationResult,
    Timing,
)


class Observation:
    def encode(self, observation: int) -> int:
        return observation

    def configuration(self) -> dict[str, str]:
        return {"type": "observation"}


class Reward:
    def encode(self, reward: float) -> float:
        return reward

    def configuration(self) -> dict[str, str]:
        return {"type": "reward"}


class StatefulSimulator:
    def __init__(self) -> None:
        self.state = 0

    @property
    def info(self) -> ModelInfo:
        return ModelInfo("fixture", "1", "fixture", "cpu", 1, 0)

    def configuration(self) -> dict[str, str]:
        return {"type": "stateful"}

    def advance(
        self,
        stimulus: int,
        reinforcement: float,
        duration_ms: float,
    ) -> SimulationResult[int]:
        self.state += stimulus + int(reinforcement)
        return SimulationResult(
            self.state,
            ActivitySummary(self.state),
            LearningSummary(True, self.state),
            Timing(duration_ms, 0, 0),
        )

    def reset(self, *, keep_learning: bool = False) -> None:
        self.state = 0

    def save(self, path: Path) -> None:
        path.write_text(str(self.state))

    def restore(self, path: Path) -> None:
        self.state = int(path.read_text())


class StatefulDecoder:
    def __init__(self) -> None:
        self.previous = 0

    def decode(self, activity: int, duration_ms: float) -> DecodedAction[int]:
        action = activity - self.previous
        self.previous = activity
        return DecodedAction(action)

    def reset(self) -> None:
        self.previous = 0

    def configuration(self) -> dict[str, str]:
        return {"type": "stateful"}

    def save(self, path: Path) -> None:
        path.write_text(str(self.previous))

    def restore(self, path: Path) -> None:
        self.previous = int(path.read_text())


class StatefulEnvironment:
    def __init__(self) -> None:
        self.value = 0

    def reset(self) -> int:
        self.value = 1
        return self.value

    def step(self, action: int) -> EnvironmentStep[int]:
        self.value += action
        return EnvironmentStep(self.value, reward=float(self.value % 2))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"value": self.value}))

    def restore(self, path: Path) -> int:
        raw: object = json.loads(path.read_text())
        if not isinstance(raw, dict) or not isinstance(raw.get("value"), int):
            raise ValueError("Invalid fixture environment")
        self.value = int(raw["value"])
        return self.value


def test_training_checkpoint_replays_next_step_exactly(tmp_path: Path) -> None:
    agent = Agent(
        simulator=StatefulSimulator(),
        observation=Observation(),
        action=StatefulDecoder(),
        reinforcement=Reward(),
    )
    session = TrainingSession(agent, StatefulEnvironment())
    session.run(2)
    checkpoint = tmp_path / "training.fccheckpoint"
    session.save_checkpoint(checkpoint)

    expected = session.step()
    expected_steps = session.steps
    session.restore_checkpoint(checkpoint)
    actual = session.step()

    assert actual.transition == expected.transition
    assert actual.agent.action == expected.agent.action
    assert actual.agent.activity == expected.agent.activity
    assert actual.agent.learning == expected.agent.learning
    assert actual.agent.metrics == expected.agent.metrics
    assert session.steps == expected_steps
