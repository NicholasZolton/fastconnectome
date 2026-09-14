"""Smallest complete native KC→MBON learning example."""

import argparse
from pathlib import Path

from fastconnectome import Agent, ApproachChoice, KCCue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--backend", choices=["auto", "cpu", "metal"], default="auto")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("runs/kc-mbon.fcmodel"),
    )
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("trials must be positive")

    with Agent.from_preset(
        "malecns-kc-conditioning",
        data_dir=args.data_dir,
        backend=args.backend,
        learning=False,
    ) as baseline:
        before = baseline.step(KCCue.A)

    with Agent.from_preset(
        "malecns-kc-conditioning",
        data_dir=args.data_dir,
        backend=args.backend,
        learning=True,
    ) as training:
        trained = training.step(KCCue.A, reward=1.0)
        for _ in range(args.trials - 1):
            training.reset(keep_learning=True)
            trained = training.step(KCCue.A, reward=1.0)
        training.export_policy(args.policy)

    with Agent.load_policy(
        args.policy,
        data_dir=args.data_dir,
        backend=args.backend,
        preset="malecns-kc-conditioning",
    ) as deployed:
        paired = deployed.step(KCCue.A)
        deployed.reset()
        unpaired = deployed.step(KCCue.B)

    print(
        f"before:   cue A → {before.action.value} "
        f"({before.metrics['mbon_spikes']} MBON07 spikes)"
    )
    print(
        f"after:    cue A → {paired.action.value} "
        f"({paired.metrics['mbon_spikes']} MBON07 spikes)"
    )
    print(
        f"control:  cue B → {unpaired.action.value} "
        f"({unpaired.metrics['mbon_spikes']} MBON07 spikes)"
    )
    print(f"changed:  {trained.learning.changed_connections:,} KC→MBON connections")
    print(f"policy:   {args.policy}")

    if (
        before.action is not ApproachChoice.AVOID
        or paired.action is not ApproachChoice.APPROACH
        or unpaired.action is not ApproachChoice.AVOID
    ):
        raise RuntimeError("KC→MBON quickstart did not pass")


if __name__ == "__main__":
    main()
