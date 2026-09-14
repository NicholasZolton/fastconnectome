"""Minimal real-time Pong example using the MaleCNS visual-turning preset."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import random

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


class Pong:
    width = 320
    height = 180
    paddle_width = 52

    def __init__(self, seed: int = 7) -> None:
        self.random = random.Random(seed)
        self.state = PongState()

    def reset(self) -> RGBFrame:
        self.state = PongState(
            ball_vx=-3.4 if self.random.random() < 0.5 else 3.4
        )
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
            state.ball_vx = -3.4 if self.random.random() < 0.5 else 3.4
            state.ball_vy = 3.0
        return EnvironmentStep(self.render(), reward=reward)

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args()

    import pygame

    pygame.init()
    screen = pygame.display.set_mode((960, 540))
    pygame.display.set_caption("FastConnectome Pong")
    font = pygame.font.SysFont("monospace", 18)
    game = Pong()
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
