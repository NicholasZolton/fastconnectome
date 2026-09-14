from pathlib import Path

from fastconnectome import Agent
from fastconnectome.types import (
    ActivitySummary,
    DecodedAction,
    LearningSummary,
    ModelInfo,
    SimulationResult,
    Timing,
)


class LengthEncoder:
    def encode(self, observation: str) -> int:
        return len(observation)

    def configuration(self) -> dict[str, str]:
        return {"type": "length"}


class SignedReward:
    def encode(self, reward: float) -> bool:
        return reward > 0

    def configuration(self) -> dict[str, str]:
        return {"type": "signed"}


class FakeSimulator:
    def __init__(self) -> None:
        self.restored: Path | None = None
        self.did_restore = False
        self.kept_learning = False

    @property
    def info(self) -> ModelInfo:
        return ModelInfo("fixture", "1", "test", "cpu", 2, 1)

    def advance(
        self, stimulus: int, reinforcement: bool, duration_ms: float
    ) -> SimulationResult[int]:
        activity = stimulus + int(reinforcement)
        return SimulationResult(
            activity,
            ActivitySummary(activity),
            LearningSummary(True, 1),
            Timing(duration_ms, 0.01, 0.01),
            {"stimulus": stimulus},
        )

    def configuration(self) -> dict[str, str]:
        return {"type": "fixture"}

    def reset(self, *, keep_learning: bool = False) -> None:
        self.kept_learning = keep_learning

    def save(self, path: Path) -> None:
        self.restored = path
        path.write_text("simulator")

    def restore(self, path: Path) -> None:
        self.restored = path
        self.did_restore = path.read_text() == "simulator"


class EvenDecoder:
    def __init__(self) -> None:
        self.resets = 0

    def decode(self, activity: int, duration_ms: float) -> DecodedAction[str]:
        return DecodedAction("even" if activity % 2 == 0 else "odd")

    def reset(self) -> None:
        self.resets += 1

    def configuration(self) -> dict[str, str]:
        return {"type": "even"}

    def save(self, path: Path) -> None:
        path.write_text(str(self.resets))

    def restore(self, path: Path) -> None:
        self.resets = int(path.read_text())


def make_agent() -> tuple[
    Agent[str, int, bool, int, str], FakeSimulator, EvenDecoder
]:
    simulator = FakeSimulator()
    decoder = EvenDecoder()
    agent = Agent(
        simulator=simulator,
        observation=LengthEncoder(),
        action=decoder,
        reinforcement=SignedReward(),
        neural_ms=20,
    )
    return agent, simulator, decoder


def test_agent_composes_adapters_and_reports_timing() -> None:
    agent, _, _ = make_agent()

    result = agent.step("fly", reward=1)

    assert result.action == "even"
    assert result.activity.total_spikes == 4
    assert result.timing.neural_ms == 20
    assert result.timing.wall_seconds >= 0
    assert agent.info.model == "fixture"


def test_agent_resets_decoder_and_simulator(tmp_path: Path) -> None:
    agent, simulator, decoder = make_agent()

    checkpoint = tmp_path / "agent.fccheckpoint"
    agent.save(checkpoint)
    agent.reset(keep_learning=True)
    agent.restore(checkpoint)

    assert simulator.kept_learning
    assert simulator.did_restore
    assert decoder.resets == 0
