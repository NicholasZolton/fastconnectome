# FastConnectome

**Run connectomes as agents.**

FastConnectome provides a small, typed boundary between an environment and a
stateful connectome simulation. Models, dynamics, sensory encoding, action
decoding, reinforcement, and execution backends remain explicit so making a
demo convenient does not silently change its scientific assumptions.

The initial release includes one experimental preset: the complete MaleCNS
v1.0 graph using Stonkfly dynamics, an RGB retinal adapter, bilateral DNp20
turning readout, and candidate dopamine-gated plasticity. Apple hosts use Metal
propagation by default when a Metal device and Swift toolchain are available;
other hosts use the native CPU kernel.

## Install and prepare MaleCNS

Python 3.11 and a C++17 compiler are required. The Metal backend additionally
requires the Swift compiler supplied by Apple developer tools. The optional
MaleCNS preparation downloads roughly 1.1 GB and uses several additional GB
for derived data.

```sh
uv sync --all-extras
uv run fastconnectome prepare malecns-v1
```

## Minimal agent loop

```python
from fastconnectome import Agent

fly = Agent.from_preset("malecns-visual-turning")
reward = 0.0

while running:
    result = fly.step(frame, reward=reward)
    transition = environment.step(result.action)
    frame = transition.observation
    reward = transition.reward
```

`reward` describes the outcome of the previous action. Positive values schedule
a PAM11 pulse, negative values schedule a PPL101 pulse, and zero schedules
nothing. The preset advances 20 ms of neural time per call and decodes DNp20
activity over a rolling 200 ms window.

## Compose an agent

```python
from fastconnectome import Agent
from fastconnectome.models.malecns import (
    BilateralTurn,
    CompoundEye,
    DopamineValence,
    MaleCNS,
)

fly = Agent(
    simulator=MaleCNS("data", learning=True),
    observation=CompoundEye(),
    action=BilateralTurn(
        cell_type="DNp20",
        window_ms=200,
        deadband_hz=2,
    ),
    reinforcement=DopamineValence(
        positive="PAM11",
        negative="PPL101",
        duration_ms=50,
        current=20,
    ),
    neural_ms=20,
)
```

The public roles are deliberately separate:

```text
ObservationEncoder: environment observation -> simulator stimulus
Simulator:          stimulus + reinforcement -> neural activity
ActionDecoder:      neural activity -> environment action
Reinforcement:      numeric reward -> simulator reinforcement
```

## Lifecycle

```python
fly.save("runs/agent.fccheckpoint")
fly.restore("runs/agent.fccheckpoint")

fly.reset()                    # training agent: reset state and learned weights
fly.reset(keep_learning=True)  # training agent: retain learned weights
```

An agent checkpoint contains neural state, plasticity traces, pending dopamine
pulse, learned weights, and decoder history. For exact synchronous training
continuation, checkpoint the environment and pending reward as well:

```python
from fastconnectome import TrainingSession

session = TrainingSession(fly, environment)
session.run(500)
session.save_checkpoint("runs/training.fccheckpoint")
session.restore_checkpoint("runs/training.fccheckpoint")
```

The environment implements `save()` and `restore()` to preserve its own state
and random generator. Real-time sessions intentionally are not claimed to resume
exactly because thread scheduling can change which observations are superseded.

## Train and deploy

A deployment policy is a learned-weight overlay, not a paused experiment:

```python
fly.export_policy("models/pong.fcmodel")

deployed = Agent.load_policy(
    "models/pong.fcmodel",
    data_dir="data",
)
```

Loading validates the model graph, dynamics, and adapter configuration; applies
the learned KC→MBON weights to a fresh connectome; clears transient state; and
freezes plasticity. The baseline graph remains in the model cache, so the policy
artifact is only tens of KiB. Resetting a deployed agent retains its policy.

The headless Pong showcase evaluates a frozen baseline, trains synchronously,
verifies exact checkpoint replay, exports a policy, loads a fresh frozen agent,
and evaluates it on the same scenario. Defaults finish in roughly two minutes
on the reference machine:

```sh
uv run python examples/train_pong.py --data-dir data
```

It reports both behavioral scores and changed synapses. A single improved run is
not presented as evidence of learning; repeat evaluation over held-out seeds is
still required.

## Runners

`run_episode` is synchronous and deterministic. `RealtimeRunner` advances an
environment at its own frequency, gives the agent the latest observation, holds
the latest action while computation continues, and queues nonzero reward events.

The Pong example uses the real-time runner at 60 FPS:

```sh
uv run python examples/pong.py --steps 1200
```

## Timing terminology

- `neural_ms`: simulated neural time advanced by each agent call.
- `window_ms`: neural activity history used by an action decoder.
- `duration_ms`: simulated duration of a reinforcement pulse.
- Environment frequency is independent and belongs to a runner.

None of these values represent guaranteed wall-clock latency.

## Fidelity and backends

Dynamics and execution backends are separate identifiers:

```python
Agent.from_preset(
    "malecns-visual-turning",
    dynamics="stonkfly-v1",
    backend="cpu",  # Override the default automatic selection.
)
```

The default `backend="auto"` selects Metal on a compatible Apple host and CPU
elsewhere. Set `backend="cpu"` or `backend="metal"` to require one explicitly.
Metal accelerates spike propagation while the existing double-precision rate
traces and plasticity rule remain on CPU. Policies can cross these backends
because both identify the same `stonkfly-v1` dynamics; exact training
checkpoints remain backend-specific.

Metal and CPU use different parallel floating-point accumulation orders. The
current fixed-input and reinforcement conformance corpora produce matching
spike readouts and tightly bounded neural-state differences; sufficiently long
or changing trajectories may diverge at individual spike thresholds. Backend
results are therefore not promised to be bit-identical. A coarser or pruned
model must use another dynamics identifier rather than hiding the change behind
a `fast=True` option.

## Scientific status

This is a wiring-constrained simulation, not a complete biological fly. The
retinal projection, neuron equations, transmitter signs, reward assignment,
motor decoder, and plasticity rule include explicit engineering assumptions.
Changing weights does not by itself demonstrate learned behavior.

The MaleCNS implementation is currently provided by
[nftechie/stonkfly](https://github.com/nftechie/stonkfly) at commit
`78ef3e05ab0fa086032098558d893667068944a0`. Stonkfly's DOOMFLY-derived neural
code is MIT licensed. MaleCNS data is downloaded separately under its upstream
CC BY 4.0 terms.
