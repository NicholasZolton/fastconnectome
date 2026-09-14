from pathlib import Path

from fastconnectome import Agent, EnvironmentStep, RealtimeRunner, run_episode
from fastconnectome.types import (
    ActivitySummary,
    DecodedAction,
    LearningSummary,
    ModelInfo,
    SimulationResult,
    Timing,
)


class NumberObservation:
    def encode(self, observation: int) -> int:
        return observation


class NumberReward:
    def encode(self, reward: float) -> float:
        return reward


class RuntimeSimulator:
    @property
    def info(self) -> ModelInfo:
        return ModelInfo("fixture", "1", "test", "cpu", 1, 0)

    def advance(
        self, stimulus: int, reinforcement: float, duration_ms: float
    ) -> SimulationResult[int]:
        value = stimulus + int(reinforcement)
        return SimulationResult(
            value,
            ActivitySummary(value),
            LearningSummary(False, 0),
            Timing(duration_ms, 0, 0),
        )

    def reset(self, *, keep_learning: bool = False) -> None:
        return None

    def save(self, path: Path) -> None:
        return None

    def restore(self, path: Path) -> None:
        return None


class NumberDecoder:
    def decode(self, activity: int, duration_ms: float) -> DecodedAction[int]:
        return DecodedAction(activity)

    def reset(self) -> None:
        return None


class NumberEnvironment:
    def __init__(self) -> None:
        self.value = 0

    def reset(self) -> int:
        self.value = 0
        return self.value

    def step(self, action: int) -> EnvironmentStep[int]:
        self.value += 1
        return EnvironmentStep(
            self.value,
            reward=1.0,
            terminated=self.value == 3,
        )


def test_synchronous_runner_uses_previous_transition_reward() -> None:
    agent = Agent(
        simulator=RuntimeSimulator(),
        observation=NumberObservation(),
        action=NumberDecoder(),
        reinforcement=NumberReward(),
    )

    result = run_episode(
        agent,
        NumberEnvironment(),
        initial_action=0,
        max_steps=10,
    )

    assert result.environment_steps == 3
    assert result.agent_steps == 3
    assert result.total_reward == 3
    assert result.terminated


def test_realtime_runner_holds_latest_action() -> None:
    agent = Agent(
        simulator=RuntimeSimulator(),
        observation=NumberObservation(),
        action=NumberDecoder(),
        reinforcement=NumberReward(),
    )

    result = RealtimeRunner(
        agent,
        NumberEnvironment(),
        initial_action=0,
        environment_hz=200,
    ).run(max_steps=3)

    assert result.environment_steps == 3
    assert result.agent_steps >= 1
    assert result.terminated
