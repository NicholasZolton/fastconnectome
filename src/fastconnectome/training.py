"""Deterministic synchronous training with whole-session checkpoints."""

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Generic, TypeVar
import zipfile

from fastconnectome.agent import Agent
from fastconnectome.protocols import CheckpointableEnvironment
from fastconnectome.types import RunResult, SessionStep

ObservationT = TypeVar("ObservationT")
StimulusT = TypeVar("StimulusT")
ReinforcementT = TypeVar("ReinforcementT")
ActivityT = TypeVar("ActivityT")
ActionT = TypeVar("ActionT")


class TrainingSession(
    Generic[ObservationT, StimulusT, ReinforcementT, ActivityT, ActionT]
):
    """Run one reproducible environment and agent at neural-step boundaries."""

    def __init__(
        self,
        agent: Agent[ObservationT, StimulusT, ReinforcementT, ActivityT, ActionT],
        environment: CheckpointableEnvironment[ObservationT, ActionT],
    ) -> None:
        self.agent = agent
        self.environment = environment
        self._observation = environment.reset()
        self._pending_reward = 0.0
        self.steps = 0
        self.total_reward = 0.0
        self.terminated = False
        self.truncated = False

    def step(self) -> SessionStep[ObservationT, ActionT]:
        if self.terminated or self.truncated:
            raise RuntimeError("The training session has ended")
        result = self.agent.step(
            self._observation,
            reward=self._pending_reward,
        )
        transition = self.environment.step(result.action)
        self._observation = transition.observation
        self._pending_reward = transition.reward
        self.steps += 1
        self.total_reward += transition.reward
        self.terminated = transition.terminated
        self.truncated = transition.truncated
        return SessionStep(transition, result)

    def run(self, max_steps: int) -> RunResult:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        completed = 0
        reward_before = self.total_reward
        while completed < max_steps and not (self.terminated or self.truncated):
            self.step()
            completed += 1
        return RunResult(
            environment_steps=completed,
            agent_steps=completed,
            total_reward=self.total_reward - reward_before,
            terminated=self.terminated,
            truncated=self.truncated,
        )

    def reset(self, *, reset_agent: bool = True, keep_learning: bool = True) -> None:
        if reset_agent:
            self.agent.reset(keep_learning=keep_learning)
        self._observation = self.environment.reset()
        self._pending_reward = 0.0
        self.steps = 0
        self.total_reward = 0.0
        self.terminated = False
        self.truncated = False

    def save_checkpoint(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
            root = Path(directory)
            self.agent.save(root / "agent.fccheckpoint")
            self.environment.save(root / "environment.npz")
            state = {
                "schema": 1,
                "kind": "fastconnectome-training-checkpoint",
                "steps": self.steps,
                "total_reward": self.total_reward,
                "pending_reward": self._pending_reward,
                "terminated": self.terminated,
                "truncated": self.truncated,
            }
            (root / "session.json").write_text(
                json.dumps(state, indent=2, allow_nan=False) + "\n"
            )
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_STORED
            ) as archive:
                for name in (
                    "session.json",
                    "agent.fccheckpoint",
                    "environment.npz",
                ):
                    archive.write(root / name, name)
        os.replace(temporary, destination)

    def restore_checkpoint(self, path: str | Path) -> None:
        source = Path(path)
        with tempfile.TemporaryDirectory(dir=source.parent) as directory:
            root = Path(directory)
            expected = {
                "session.json",
                "agent.fccheckpoint",
                "environment.npz",
            }
            with zipfile.ZipFile(source, "r") as archive:
                if set(archive.namelist()) != expected:
                    raise ValueError("Unexpected training checkpoint contents")
                for name in expected:
                    (root / name).write_bytes(archive.read(name))
            raw: object = json.loads((root / "session.json").read_text())
            state = self._validate_state(raw)
            self.agent.restore(root / "agent.fccheckpoint")
            observation = self.environment.restore(root / "environment.npz")
        self._observation = observation
        self._pending_reward = float(state["pending_reward"])
        self.steps = int(state["steps"])
        self.total_reward = float(state["total_reward"])
        self.terminated = bool(state["terminated"])
        self.truncated = bool(state["truncated"])

    @staticmethod
    def _validate_state(raw: object) -> dict[str, int | float | bool | str]:
        if not isinstance(raw, dict):
            raise ValueError("Invalid training checkpoint state")
        if (
            raw.get("schema") != 1
            or raw.get("kind") != "fastconnectome-training-checkpoint"
            or not isinstance(raw.get("steps"), int)
            or isinstance(raw.get("steps"), bool)
            or int(raw["steps"]) < 0
            or not isinstance(raw.get("total_reward"), (int, float))
            or isinstance(raw.get("total_reward"), bool)
            or not isinstance(raw.get("pending_reward"), (int, float))
            or isinstance(raw.get("pending_reward"), bool)
            or not isinstance(raw.get("terminated"), bool)
            or not isinstance(raw.get("truncated"), bool)
            or not math.isfinite(float(raw["total_reward"]))
            or not math.isfinite(float(raw["pending_reward"]))
        ):
            raise ValueError("Invalid training checkpoint state")
        return {
            "schema": 1,
            "kind": "fastconnectome-training-checkpoint",
            "steps": int(raw["steps"]),
            "total_reward": float(raw["total_reward"]),
            "pending_reward": float(raw["pending_reward"]),
            "terminated": bool(raw["terminated"]),
            "truncated": bool(raw["truncated"]),
        }
