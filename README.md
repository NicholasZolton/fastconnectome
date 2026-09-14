# FastConnectome

**Run connectomes as agents.**

FastConnectome provides a small, typed boundary between an environment and a
stateful connectome simulation. Models, dynamics, sensory encoding, action
decoding, reinforcement, and execution backends remain explicit so making a
demo convenient does not silently change its scientific assumptions.

The initial release includes one experimental preset: the complete MaleCNS
v1.0 graph using Stonkfly's CPU dynamics, RGB retinal adapter, bilateral DNp20
turning readout, and candidate dopamine-gated plasticity.

## Install and prepare MaleCNS

Python 3.11 and a C++17 compiler are required. The optional MaleCNS preparation
downloads roughly 1.1 GB and uses several additional GB for derived data.

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
fly.save("runs/brain.npz")
fly.restore("runs/brain.npz")

fly.reset()                    # reset state and learned weights
fly.reset(keep_learning=True)  # reset activity but retain learned weights
```

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
    backend="cpu",
)
```

A future Metal implementation of the same equations would use another backend.
A coarser or pruned interactive model must use another dynamics identifier rather
than hiding the change behind a `fast=True` option.

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
