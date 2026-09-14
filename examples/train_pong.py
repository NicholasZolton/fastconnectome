"""Train, checkpoint, export, and deploy a MaleCNS Pong agent."""

import argparse
from collections import Counter
from dataclasses import dataclass
import gc
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from fastconnectome import Agent, TrainingSession
from fastconnectome.models.malecns import CurrentPulse, NeuralActivity, Turn
from pong import Pong, PongReward

RGBFrame = NDArray[np.uint8]
PongSession = TrainingSession[
    RGBFrame,
    RGBFrame,
    CurrentPulse | None,
    NeuralActivity,
    Turn,
]


@dataclass(frozen=True, slots=True)
class PhaseResult:
    steps: int
    reward: float
    hits: int
    misses: int
    actions: Counter[Turn]
    changed_connections: int
    aligned_steps: int


def run_phase(
    session: PongSession,
    game: Pong,
    steps: int,
) -> PhaseResult:
    actions: Counter[Turn] = Counter()
    reward_before = session.total_reward
    changed_connections = 0
    aligned_steps = 0
    for _ in range(steps):
        current = session.step()
        actions[current.agent.action] += 1
        changed_connections = current.agent.learning.changed_connections
        aligned_steps += int(current.transition.info.get("aligned", False))
    return PhaseResult(
        steps=steps,
        reward=session.total_reward - reward_before,
        hits=game.state.hits,
        misses=game.state.misses,
        actions=actions,
        changed_connections=changed_connections,
        aligned_steps=aligned_steps,
    )


def print_phase(name: str, result: PhaseResult) -> None:
    action_text = ", ".join(
        f"{action.value}={result.actions[action]}" for action in Turn
    )
    print(
        f"{name:>10}: reward={result.reward:+.0f} "
        f"hits={result.hits} misses={result.misses} "
        f"aligned={result.aligned_steps}/{result.steps} "
        f"changed={result.changed_connections} [{action_text}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a full MaleCNS Pong policy and deploy it frozen"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pong-demo"))
    parser.add_argument("--evaluation-steps", type=int, default=250)
    parser.add_argument("--training-steps", type=int, default=500)
    parser.add_argument(
        "--reward",
        type=PongReward,
        choices=list(PongReward),
        default=PongReward.SPARSE,
    )
    parser.add_argument("--tracking-tolerance", type=float, default=26.0)
    args = parser.parse_args()
    if args.evaluation_steps <= 0 or args.training_steps < 2:
        parser.error("evaluation steps must be positive and training steps must be at least 2")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "training.fccheckpoint"
    policy = args.output_dir / "pong.fcmodel"
    started = perf_counter()

    print("Loading frozen baseline...")
    baseline_agent = Agent.from_preset(
        "malecns-visual-turning",
        data_dir=args.data_dir,
        learning=False,
    )
    baseline_game = Pong(
        seed=7,
        reward=args.reward,
        tracking_tolerance=args.tracking_tolerance,
    )
    baseline_session = TrainingSession(baseline_agent, baseline_game)
    baseline = run_phase(baseline_session, baseline_game, args.evaluation_steps)
    print_phase("baseline", baseline)
    del baseline_session, baseline_agent, baseline_game
    gc.collect()

    print("Loading training agent...")
    training_agent = Agent.from_preset(
        "malecns-visual-turning",
        data_dir=args.data_dir,
        learning=True,
    )
    training_game = Pong(
        seed=23,
        reward=args.reward,
        tracking_tolerance=args.tracking_tolerance,
    )
    training_session = TrainingSession(training_agent, training_game)
    first_steps = args.training_steps // 2
    first = run_phase(training_session, training_game, first_steps)
    training_session.save_checkpoint(checkpoint)

    probe = training_session.step()
    training_session.restore_checkpoint(checkpoint)
    replay = training_session.step()
    exact_resume = (
        probe.agent.action == replay.agent.action
        and probe.agent.activity == replay.agent.activity
        and probe.agent.learning == replay.agent.learning
        and probe.agent.metrics == replay.agent.metrics
        and probe.transition.reward == replay.transition.reward
        and np.array_equal(
            probe.transition.observation,
            replay.transition.observation,
        )
    )
    if not exact_resume:
        raise RuntimeError("Checkpoint replay diverged")

    remaining_steps = args.training_steps - first_steps - 1
    second = run_phase(training_session, training_game, remaining_steps)
    training = PhaseResult(
        steps=args.training_steps,
        reward=first.reward + probe.transition.reward + second.reward,
        hits=training_game.state.hits,
        misses=training_game.state.misses,
        actions=first.actions + Counter([replay.agent.action]) + second.actions,
        changed_connections=(
            second.changed_connections
            if remaining_steps
            else replay.agent.learning.changed_connections
        ),
        aligned_steps=first.aligned_steps
        + int(probe.transition.info.get("aligned", False))
        + second.aligned_steps,
    )
    print_phase("training", training)
    print(f"checkpoint: exact replay verified ({checkpoint})")
    training_agent.export_policy(policy)
    del training_session, training_agent, training_game
    gc.collect()

    print("Loading fresh frozen deployment...")
    deployed_agent = Agent.load_policy(
        policy,
        data_dir=args.data_dir,
    )
    deployed_game = Pong(
        seed=7,
        reward=args.reward,
        tracking_tolerance=args.tracking_tolerance,
    )
    deployed_session = TrainingSession(deployed_agent, deployed_game)
    deployed = run_phase(deployed_session, deployed_game, args.evaluation_steps)
    print_phase("deployed", deployed)

    elapsed = perf_counter() - started
    print(f"policy: {policy} ({policy.stat().st_size / 1024:.1f} KiB, frozen)")
    print(f"elapsed: {elapsed:.1f} seconds")
    reward_delta = deployed.reward - baseline.reward
    print(f"evaluation reward delta: {reward_delta:+.0f}")
    if args.reward == PongReward.TRACKING:
        alignment_delta = deployed.aligned_steps - baseline.aligned_steps
        print(f"evaluation aligned-step delta: {alignment_delta:+d}")
    if reward_delta <= 0:
        print("Result: plasticity was saved and deployed, but Pong did not improve.")
    else:
        print("Result: this run improved; repeat across seeds before claiming learning.")


if __name__ == "__main__":
    main()
