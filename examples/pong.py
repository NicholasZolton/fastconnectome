"""Minimal real-time Pong example using the MaleCNS visual-turning preset."""

import argparse
from dataclasses import asdict, dataclass
from enum import StrEnum
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fastconnectome import Agent, EnvironmentStep, RealtimeRunner, StepResult
from fastconnectome.models.malecns import Turn

RGBFrame = NDArray[np.uint8]


@dataclass(slots=True)
class PongState:
    paddle_x: float = 134
    ball_x: float = 160
    ball_y: float = 63
    ball_vx: float = 3.4
    ball_vy: float = 3.0
    hits: int = 0
    misses: int = 0


class PongReward(StrEnum):
    SPARSE = "sparse"
    TRACKING = "tracking"


class Pong:
    width = 320
    height = 180
    paddle_width = 52

    def __init__(
        self,
        seed: int = 7,
        *,
        reward: PongReward = PongReward.SPARSE,
        tracking_tolerance: float = 26.0,
    ) -> None:
        if not np.isfinite(tracking_tolerance) or tracking_tolerance < 0:
            raise ValueError("tracking_tolerance must be finite and nonnegative")
        self._initial_seed = seed & 0xFFFFFFFF
        self._rng_state = self._initial_seed
        self.reward = reward
        self.tracking_tolerance = tracking_tolerance
        self.state = PongState()

    def _direction(self) -> float:
        self._rng_state = (1664525 * self._rng_state + 1013904223) & 0xFFFFFFFF
        return -3.4 if self._rng_state < 0x80000000 else 3.4

    def reset(self) -> RGBFrame:
        self._rng_state = self._initial_seed
        self.state = PongState(ball_vx=self._direction())
        return self.render()

    def step(self, action: Turn) -> EnvironmentStep[RGBFrame]:
        state = self.state
        state.paddle_x = min(
            max(state.paddle_x + {Turn.LEFT: -16, Turn.RIGHT: 16, Turn.HOLD: 0}[action], 0),
            self.width - self.paddle_width,
        )
        state.ball_x += state.ball_vx
        state.ball_y += state.ball_vy
        if state.ball_x <= 4 or state.ball_x >= self.width - 4:
            state.ball_vx *= -1
            state.ball_x = min(max(state.ball_x, 4), self.width - 4)
        if state.ball_y <= 4:
            state.ball_vy = abs(state.ball_vy)

        reward = 0.0
        paddle_y = 162
        if (
            state.ball_vy > 0
            and state.ball_y >= paddle_y - 4
            and state.paddle_x <= state.ball_x <= state.paddle_x + self.paddle_width
        ):
            state.ball_y = paddle_y - 4
            state.ball_vy = -abs(state.ball_vy)
            state.hits += 1
            reward = 1.0
        elif state.ball_y > self.height + 4:
            state.misses += 1
            reward = -1.0
            state.ball_x = self.width / 2
            state.ball_y = self.height * 0.35
            state.ball_vx = self._direction()
            state.ball_vy = 3.0
        alignment_error = abs(
            state.ball_x - (state.paddle_x + self.paddle_width / 2)
        )
        aligned = alignment_error <= self.tracking_tolerance
        if self.reward == PongReward.TRACKING:
            reward = 1.0 if aligned else -1.0
        return EnvironmentStep(
            self.render(),
            reward=reward,
            info={
                "alignment_error": alignment_error,
                "aligned": aligned,
            },
        )

    def render(self) -> RGBFrame:
        state = self.state
        frame = np.full((self.height, self.width, 3), 236, dtype=np.uint8)
        frame[0:2, :] = (35, 42, 52)
        frame[:, 0:2] = (35, 42, 52)
        frame[:, -2:] = (35, 42, 52)
        paddle_x = int(state.paddle_x)
        frame[162:168, paddle_x : paddle_x + self.paddle_width] = (22, 101, 216)
        x = int(state.ball_x)
        y = int(state.ball_y)
        frame[max(0, y - 4) : y + 5, max(0, x - 4) : x + 5] = (218, 62, 68)
        return frame

    def save(self, path: Path) -> None:
        manifest = {
            "schema": 1,
            "type": "fastconnectome-pong-v1",
            "initial_seed": self._initial_seed,
            "reward": self.reward.value,
            "tracking_tolerance": self.tracking_tolerance,
            "rng_state": self._rng_state,
            "state": asdict(self.state),
        }
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                manifest=json.dumps(manifest, allow_nan=False),
            )

    def restore(self, path: Path) -> RGBFrame:
        with np.load(path, allow_pickle=False) as archive:
            raw: object = json.loads(str(archive["manifest"]))
        if not isinstance(raw, dict):
            raise ValueError("Invalid Pong checkpoint")
        state_raw = raw.get("state")
        if (
            raw.get("schema") != 1
            or raw.get("type") != "fastconnectome-pong-v1"
            or raw.get("initial_seed") != self._initial_seed
            or raw.get("reward", PongReward.SPARSE.value) != self.reward.value
            or raw.get("tracking_tolerance", 26.0) != self.tracking_tolerance
            or not isinstance(raw.get("rng_state"), int)
            or not isinstance(state_raw, dict)
        ):
            raise ValueError("Pong checkpoint configuration mismatch")
        values = [
            state_raw.get("paddle_x"),
            state_raw.get("ball_x"),
            state_raw.get("ball_y"),
            state_raw.get("ball_vx"),
            state_raw.get("ball_vy"),
        ]
        hits = state_raw.get("hits")
        misses = state_raw.get("misses")
        if (
            not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and np.isfinite(value)
                for value in values
            )
            or not isinstance(hits, int)
            or isinstance(hits, bool)
            or hits < 0
            or not isinstance(misses, int)
            or isinstance(misses, bool)
            or misses < 0
        ):
            raise ValueError("Invalid Pong checkpoint state")
        self._rng_state = int(raw["rng_state"])
        self.state = PongState(
            paddle_x=float(values[0]),
            ball_x=float(values[1]),
            ball_y=float(values[2]),
            ball_vx=float(values[3]),
            ball_vy=float(values[4]),
            hits=hits,
            misses=misses,
        )
        return self.render()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument(
        "--reward",
        type=PongReward,
        choices=list(PongReward),
        default=PongReward.SPARSE,
    )
    parser.add_argument("--tracking-tolerance", type=float, default=26.0)
    args = parser.parse_args()

    import pygame

    pygame.init()
    screen = pygame.display.set_mode((960, 540))
    pygame.display.set_caption("FastConnectome Pong")
    font = pygame.font.SysFont("monospace", 18)
    game = Pong(
        reward=args.reward,
        tracking_tolerance=args.tracking_tolerance,
    )
    fly = Agent.from_preset("malecns-visual-turning", data_dir=args.data_dir)

    def display(
        step: int,
        transition: EnvironmentStep[RGBFrame],
        result: StepResult[Turn] | None,
    ) -> bool:
        if any(event.type == pygame.QUIT for event in pygame.event.get()):
            return False
        surface = pygame.surfarray.make_surface(transition.observation.swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surface, screen.get_size()), (0, 0))
        action = result.action.value if result is not None else "warming up"
        label = font.render(
            f"action={action} hits={game.state.hits} misses={game.state.misses} step={step}",
            True,
            (15, 20, 28),
            (236, 236, 236),
        )
        screen.blit(label, (12, 12))
        pygame.display.flip()
        return True

    try:
        outcome = RealtimeRunner(
            fly,
            game,
            initial_action=Turn.HOLD,
            environment_hz=60,
        ).run(max_steps=args.steps, on_step=display)
        print(outcome)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
