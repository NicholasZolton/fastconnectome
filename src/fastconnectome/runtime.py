"""Synchronous and latest-observation runners for agent environments."""

from collections import deque
from threading import Condition, Thread
from time import monotonic, sleep
from typing import Callable, Generic, TypeVar

from fastconnectome.agent import Agent
from fastconnectome.protocols import Environment
from fastconnectome.types import EnvironmentStep, RunResult, StepResult

ObservationT = TypeVar("ObservationT")
StimulusT = TypeVar("StimulusT")
ReinforcementT = TypeVar("ReinforcementT")
ActivityT = TypeVar("ActivityT")
ActionT = TypeVar("ActionT")


def run_episode(
    agent: Agent[ObservationT, StimulusT, ReinforcementT, ActivityT, ActionT],
    environment: Environment[ObservationT, ActionT],
    *,
    initial_action: ActionT,
    max_steps: int,
) -> RunResult:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    observation = environment.reset()
    reward = 0.0
    action = initial_action
    total_reward = 0.0
    terminated = False
    truncated = False
    for step_index in range(max_steps):
        result = agent.step(observation, reward=reward)
        action = result.action
        transition = environment.step(action)
        observation = transition.observation
        reward = transition.reward
        total_reward += reward
        terminated = transition.terminated
        truncated = transition.truncated
        if terminated or truncated:
            return RunResult(
                step_index + 1, step_index + 1, total_reward, terminated, truncated
            )
    return RunResult(max_steps, max_steps, total_reward, terminated, truncated)


class _AgentWorker(Generic[ObservationT, StimulusT, ReinforcementT, ActivityT, ActionT]):
    def __init__(
        self,
        agent: Agent[ObservationT, StimulusT, ReinforcementT, ActivityT, ActionT],
    ) -> None:
        self._agent = agent
        self._condition = Condition()
        self._observation: ObservationT | None = None
        self._observation_version = 0
        self._rewards: deque[float] = deque()
        self._result: StepResult[ActionT] | None = None
        self._result_version = 0
        self._stopping = False
        self._failure: BaseException | None = None
        self._thread = Thread(target=self._run, name="fastconnectome-agent", daemon=True)
        self._thread.start()

    def submit(self, observation: ObservationT, reward: float) -> None:
        with self._condition:
            self._observation = observation
            self._observation_version += 1
            if reward != 0:
                self._rewards.append(reward)
            self._condition.notify()

    def latest(self) -> tuple[int, StepResult[ActionT] | None]:
        with self._condition:
            if self._failure is not None:
                raise RuntimeError("Agent worker failed") from self._failure
            return self._result_version, self._result

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify()
        self._thread.join()
        self.latest()

    def _run(self) -> None:
        consumed_version = 0
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stopping
                        or self._observation_version > consumed_version
                    )
                    if self._stopping:
                        return
                    observation = self._observation
                    consumed_version = self._observation_version
                    reward = self._rewards.popleft() if self._rewards else 0.0
                if observation is None:
                    continue
                result = self._agent.step(observation, reward=reward)
                with self._condition:
                    self._result = result
                    self._result_version += 1
        except BaseException as error:
            with self._condition:
                self._failure = error


StepCallback = Callable[[int, EnvironmentStep[ObservationT], StepResult[ActionT] | None], bool]


class RealtimeRunner(Generic[ObservationT, StimulusT, ReinforcementT, ActivityT, ActionT]):
    """Run environment time independently while holding the latest agent action."""

    def __init__(
        self,
        agent: Agent[ObservationT, StimulusT, ReinforcementT, ActivityT, ActionT],
        environment: Environment[ObservationT, ActionT],
        *,
        initial_action: ActionT,
        environment_hz: float = 60.0,
    ) -> None:
        if environment_hz <= 0:
            raise ValueError("environment_hz must be positive")
        self.agent = agent
        self.environment = environment
        self.initial_action = initial_action
        self.environment_hz = environment_hz

    def run(
        self,
        *,
        max_steps: int,
        on_step: StepCallback[ObservationT, ActionT] | None = None,
    ) -> RunResult:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        worker = _AgentWorker(self.agent)
        observation = self.environment.reset()
        worker.submit(observation, 0.0)
        action = self.initial_action
        latest_result: StepResult[ActionT] | None = None
        result_version = 0
        agent_steps = 0
        total_reward = 0.0
        terminated = False
        truncated = False
        deadline = monotonic()
        completed_steps = 0
        try:
            for step_index in range(max_steps):
                latest_version, candidate = worker.latest()
                if candidate is not None and latest_version != result_version:
                    latest_result = candidate
                    action = candidate.action
                    agent_steps += latest_version - result_version
                    result_version = latest_version
                transition = self.environment.step(action)
                completed_steps = step_index + 1
                total_reward += transition.reward
                terminated = transition.terminated
                truncated = transition.truncated
                worker.submit(transition.observation, transition.reward)
                if on_step is not None and not on_step(
                    step_index, transition, latest_result
                ):
                    break
                if terminated or truncated:
                    break
                deadline += 1.0 / self.environment_hz
                sleep(max(0.0, deadline - monotonic()))
        finally:
            worker.stop()
        return RunResult(
            completed_steps,
            agent_steps,
            total_reward,
            terminated,
            truncated,
        )
