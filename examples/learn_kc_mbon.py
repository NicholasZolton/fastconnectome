"""Controlled KC→MBON conditioning through the complete MaleCNS graph."""

import argparse
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from fastconnectome.models.malecns import (
    CurrentPulse,
    MaleCNS,
    MaleCNSStimulus,
    MetalMaleCNS,
    PopulationCurrent,
)
from fastconnectome.presets import resolve_backend

CONDITIONED_CUE = "KCab-m"
UNRELATED_CUE = "KCab-s"
CUE_CURRENT = 40.0
TRIAL_MS = 1000.0
REWARD_MS = 200.0
REWARD_CURRENT = 20.0
APPROACH_THRESHOLD = 480
BLACK_FRAME = np.zeros((180, 320, 3), dtype=np.uint8)


class Choice(StrEnum):
    APPROACH = "approach"
    AVOID = "avoid"


@dataclass(frozen=True, slots=True)
class CueResponse:
    cue: str
    kc_spikes: int
    mbon07_spikes: int
    action: Choice


def cue_stimulus(population: str) -> MaleCNSStimulus:
    return MaleCNSStimulus(
        BLACK_FRAME,
        (PopulationCurrent(population, CUE_CURRENT),),
    )


def decode_choice(mbon07_spikes: int) -> Choice:
    """Apply the fixed decoder used unchanged by every experimental arm."""

    if mbon07_spikes >= APPROACH_THRESHOLD:
        return Choice.APPROACH
    return Choice.AVOID


def make_simulator(
    data_dir: Path,
    backend: str,
    *,
    learning: bool,
) -> MaleCNS | MetalMaleCNS:
    selected = resolve_backend(backend)
    simulator_type = MetalMaleCNS if selected == "metal" else MaleCNS
    return simulator_type(data_dir, learning=learning)


def close_simulator(simulator: MaleCNS | MetalMaleCNS) -> None:
    if isinstance(simulator, MetalMaleCNS):
        simulator.close()


def evaluate(simulator: MaleCNS | MetalMaleCNS, cue: str) -> CueResponse:
    simulator.reset(keep_learning=True)
    result = simulator.advance(cue_stimulus(cue), None, TRIAL_MS)
    types = result.activity.populations.cell_types
    counts = result.activity.counts
    kc_spikes = int(counts[types == cue].sum())
    mbon07_spikes = int(counts[types == "MBON07"].sum())
    return CueResponse(cue, kc_spikes, mbon07_spikes, decode_choice(mbon07_spikes))


def train(
    simulator: MaleCNS | MetalMaleCNS,
    cue: str,
    trials: int,
    reinforcement: CurrentPulse | None,
) -> int:
    changed_connections = 0
    for _ in range(trials):
        simulator.reset(keep_learning=True)
        result = simulator.advance(
            cue_stimulus(cue),
            reinforcement,
            TRIAL_MS,
        )
        changed_connections = result.learning_summary.changed_connections
    return changed_connections


def manifest(
    simulator: MaleCNS | MetalMaleCNS,
    training_cue: str,
) -> dict[str, object]:
    return {
        "schema": 1,
        "kind": "fastconnectome-kc-mbon-conditioning",
        "model": asdict(simulator.info),
        "cue": {
            "conditioned_population": CONDITIONED_CUE,
            "unrelated_population": UNRELATED_CUE,
            "current": CUE_CURRENT,
            "duration_ms": TRIAL_MS,
            "training_population": training_cue,
        },
        "reinforcement": {
            "population": "PAM11",
            "current": REWARD_CURRENT,
            "duration_ms": REWARD_MS,
        },
        "decoder": {
            "population": "MBON07",
            "approach_threshold_spikes": APPROACH_THRESHOLD,
        },
    }


def deploy_policy(
    source: MaleCNS | MetalMaleCNS,
    path: Path,
    data_dir: Path,
    backend: str,
    training_cue: str,
) -> MaleCNS | MetalMaleCNS:
    source.export_policy(path, manifest(source, training_cue))
    deployed = make_simulator(data_dir, backend, learning=False)
    try:
        deployed.import_policy(path, manifest(deployed, training_cue))
    except Exception:
        close_simulator(deployed)
        raise
    return deployed


def print_response(label: str, response: CueResponse) -> None:
    print(
        f"{label:>18}: cue={response.cue:<7} "
        f"KC={response.kc_spikes:>5} MBON07={response.mbon07_spikes:>3} "
        f"action={response.action.value}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a controlled native KC-to-MBON conditioning task"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/kc-mbon-conditioning"),
    )
    parser.add_argument("--backend", choices=["auto", "cpu", "metal"], default="auto")
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("trials must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reinforcement = CurrentPulse(
        "PAM11",
        REWARD_MS,
        REWARD_CURRENT,
        "positive",
    )

    baseline = make_simulator(args.data_dir, args.backend, learning=False)
    try:
        baseline_conditioned = evaluate(baseline, CONDITIONED_CUE)
        baseline_unrelated = evaluate(baseline, UNRELATED_CUE)
        print_response("baseline", baseline_conditioned)
        print_response("baseline control", baseline_unrelated)
    finally:
        close_simulator(baseline)
    del baseline

    paired_training = make_simulator(args.data_dir, args.backend, learning=True)
    try:
        paired_changed = train(
            paired_training,
            CONDITIONED_CUE,
            args.trials,
            reinforcement,
        )
        paired_deployment = deploy_policy(
            paired_training,
            args.output_dir / "paired.npz",
            args.data_dir,
            args.backend,
            CONDITIONED_CUE,
        )
    finally:
        close_simulator(paired_training)
    del paired_training
    try:
        paired_conditioned = evaluate(paired_deployment, CONDITIONED_CUE)
        paired_unrelated = evaluate(paired_deployment, UNRELATED_CUE)
        print_response("paired", paired_conditioned)
        print_response("unrelated cue", paired_unrelated)
    finally:
        close_simulator(paired_deployment)
    del paired_deployment

    alternate_training = make_simulator(args.data_dir, args.backend, learning=True)
    try:
        alternate_changed = train(
            alternate_training,
            UNRELATED_CUE,
            args.trials,
            reinforcement,
        )
        alternate_deployment = deploy_policy(
            alternate_training,
            args.output_dir / "alternate-paired.npz",
            args.data_dir,
            args.backend,
            UNRELATED_CUE,
        )
    finally:
        close_simulator(alternate_training)
    del alternate_training
    try:
        alternate_conditioned = evaluate(alternate_deployment, CONDITIONED_CUE)
        alternate_unrelated = evaluate(alternate_deployment, UNRELATED_CUE)
        print_response("alternate control", alternate_conditioned)
        print_response("alternate paired", alternate_unrelated)
    finally:
        close_simulator(alternate_deployment)
    del alternate_deployment

    no_reward_training = make_simulator(args.data_dir, args.backend, learning=True)
    try:
        no_reward_changed = train(
            no_reward_training,
            CONDITIONED_CUE,
            args.trials,
            None,
        )
        no_reward_deployment = deploy_policy(
            no_reward_training,
            args.output_dir / "no-reward.npz",
            args.data_dir,
            args.backend,
            CONDITIONED_CUE,
        )
    finally:
        close_simulator(no_reward_training)
    del no_reward_training
    try:
        no_reward_response = evaluate(no_reward_deployment, CONDITIONED_CUE)
        print_response("no reward", no_reward_response)
    finally:
        close_simulator(no_reward_deployment)
    del no_reward_deployment

    frozen = make_simulator(args.data_dir, args.backend, learning=False)
    try:
        frozen_changed = train(
            frozen,
            CONDITIONED_CUE,
            args.trials,
            reinforcement,
        )
        frozen_response = evaluate(frozen, CONDITIONED_CUE)
        print_response("plasticity frozen", frozen_response)
    finally:
        close_simulator(frozen)
    del frozen

    erased = make_simulator(args.data_dir, args.backend, learning=False)
    try:
        erased_response = evaluate(erased, CONDITIONED_CUE)
        print_response("memory erased", erased_response)
    finally:
        close_simulator(erased)
    del erased

    passed = (
        baseline_conditioned.action == Choice.AVOID
        and baseline_unrelated.action == Choice.AVOID
        and paired_conditioned.action == Choice.APPROACH
        and paired_unrelated.action == Choice.AVOID
        and alternate_conditioned.action == Choice.AVOID
        and alternate_unrelated.action == Choice.APPROACH
        and no_reward_response.action == Choice.AVOID
        and frozen_response.action == Choice.AVOID
        and erased_response == baseline_conditioned
        and paired_changed > 0
        and alternate_changed > 0
        and no_reward_changed > 0
        and frozen_changed == 0
    )
    print(
        f"changed connections: paired={paired_changed:,}, "
        f"alternate={alternate_changed:,}, "
        f"no-reward={no_reward_changed:,}, frozen={frozen_changed:,}"
    )
    if not passed:
        raise RuntimeError("KC→MBON conditioning controls did not pass")
    print("PASS: paired PAM11 reinforcement changed the fixed MBON07 behavior.")
    print("PASS: alternate-cue, no-reward, frozen, and memory-erasure controls held.")
    print(
        "Scope: direct population currents are a controlled mechanism assay, "
        "not natural sensory learning."
    )


if __name__ == "__main__":
    main()
