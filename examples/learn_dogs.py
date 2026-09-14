"""Animate a fixed dog detector feeding native KC→MBON conditioning."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from queue import Empty, Queue
from threading import Event as ThreadEvent, Thread
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from fastconnectome import Agent, ApproachChoice, KCCue
from fastconnectome.types import StepResult

if TYPE_CHECKING:
    import pygame

RGBFrame = NDArray[np.uint8]
Color = tuple[int, int, int]
BACKGROUND: Color = (225, 235, 238)
DOG_TRAIN_VARIANTS = (0, 1)
DOG_TEST_VARIANT = 2
CAT_TEST_VARIANT = 2


class DemoStage(StrEnum):
    BASELINE = "baseline"
    TRAINING = "training"
    DOG_TEST = "dog-test"
    CAT_TEST = "cat-test"


@dataclass(frozen=True, slots=True)
class VisionResult:
    cue: KCCue
    dog_confidence: float


@dataclass(frozen=True, slots=True)
class DemoEvent:
    stage: DemoStage
    frame: RGBFrame
    vision: VisionResult
    response: StepResult[ApproachChoice]
    trial: int
    trials: int


def _ellipse(
    frame: RGBFrame,
    center_x: int,
    center_y: int,
    radius_x: int,
    radius_y: int,
    color: Color,
) -> None:
    rows, columns = np.ogrid[: frame.shape[0], : frame.shape[1]]
    mask = (
        ((columns - center_x) / radius_x) ** 2
        + ((rows - center_y) / radius_y) ** 2
        <= 1
    )
    frame[mask] = color


def _triangle(
    frame: RGBFrame,
    first: tuple[int, int],
    second: tuple[int, int],
    third: tuple[int, int],
    color: Color,
) -> None:
    rows, columns = np.ogrid[: frame.shape[0], : frame.shape[1]]

    def edge(
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> NDArray[np.int64]:
        return (columns - start[0]) * (end[1] - start[1]) - (
            rows - start[1]
        ) * (end[0] - start[0])

    first_edge = edge(first, second)
    second_edge = edge(second, third)
    third_edge = edge(third, first)
    mask = ((first_edge >= 0) & (second_edge >= 0) & (third_edge >= 0)) | (
        (first_edge <= 0) & (second_edge <= 0) & (third_edge <= 0)
    )
    frame[mask] = color


def _line(
    frame: RGBFrame,
    start: tuple[int, int],
    end: tuple[int, int],
    color: Color,
    width: int = 2,
) -> None:
    points = max(abs(end[0] - start[0]), abs(end[1] - start[1])) + 1
    columns = np.linspace(start[0], end[0], points).astype(np.int32)
    rows = np.linspace(start[1], end[1], points).astype(np.int32)
    for offset in range(-(width // 2), width // 2 + 1):
        selected_rows = np.clip(rows + offset, 0, frame.shape[0] - 1)
        selected_columns = np.clip(columns, 0, frame.shape[1] - 1)
        frame[selected_rows, selected_columns] = color


def render_dog(variant: int) -> RGBFrame:
    """Render one synthetic dog face for the transparent vision demonstration."""

    offsets = ((-3, 2), (3, -2), (0, 0))
    if variant < 0 or variant >= len(offsets):
        raise ValueError("Unknown dog variant")
    offset_x, offset_y = offsets[variant]
    frame = np.full((180, 180, 3), BACKGROUND, dtype=np.uint8)
    fur: Color = ((174, 104, 55), (194, 119, 61), (181, 91, 51))[variant]
    dark: Color = (67, 47, 44)
    muzzle: Color = (235, 191, 132)
    _ellipse(frame, 45 + offset_x, 79 + offset_y, 24, 43, dark)
    _ellipse(frame, 135 + offset_x, 79 + offset_y, 24, 43, dark)
    _ellipse(frame, 90 + offset_x, 90 + offset_y, 53, 49, fur)
    _ellipse(frame, 90 + offset_x, 112 + offset_y, 29, 22, muzzle)
    _ellipse(frame, 71 + offset_x, 82 + offset_y, 6, 7, dark)
    _ellipse(frame, 109 + offset_x, 82 + offset_y, 6, 7, dark)
    _ellipse(frame, 90 + offset_x, 103 + offset_y, 9, 7, dark)
    _line(
        frame,
        (90 + offset_x, 109 + offset_y),
        (90 + offset_x, 122 + offset_y),
        dark,
    )
    return frame


def render_cat(variant: int) -> RGBFrame:
    """Render one synthetic non-dog control with similar colors and facial parts."""

    offsets = ((2, 1), (-3, -1), (0, 0))
    if variant < 0 or variant >= len(offsets):
        raise ValueError("Unknown cat variant")
    offset_x, offset_y = offsets[variant]
    frame = np.full((180, 180, 3), BACKGROUND, dtype=np.uint8)
    fur: Color = ((181, 107, 57), (191, 112, 63), (176, 96, 53))[variant]
    dark: Color = (67, 47, 44)
    _triangle(
        frame,
        (44 + offset_x, 64 + offset_y),
        (59 + offset_x, 20 + offset_y),
        (79 + offset_x, 62 + offset_y),
        fur,
    )
    _triangle(
        frame,
        (101 + offset_x, 62 + offset_y),
        (121 + offset_x, 20 + offset_y),
        (136 + offset_x, 64 + offset_y),
        fur,
    )
    _ellipse(frame, 90 + offset_x, 91 + offset_y, 51, 47, fur)
    _ellipse(frame, 71 + offset_x, 83 + offset_y, 6, 7, dark)
    _ellipse(frame, 109 + offset_x, 83 + offset_y, 6, 7, dark)
    _triangle(
        frame,
        (83 + offset_x, 104 + offset_y),
        (97 + offset_x, 104 + offset_y),
        (90 + offset_x, 112 + offset_y),
        dark,
    )
    for row_offset in (-5, 3, 11):
        _line(
            frame,
            (77 + offset_x, 108 + row_offset + offset_y),
            (31 + offset_x, 102 + row_offset + offset_y),
            dark,
        )
        _line(
            frame,
            (103 + offset_x, 108 + row_offset + offset_y),
            (149 + offset_x, 102 + row_offset + offset_y),
            dark,
        )
    return frame


class TemplateDogVision:
    """Classify synthetic cards by comparing foreground pixels with two templates."""

    def __init__(self) -> None:
        self._dog = self._foreground(render_dog(0))
        self._cat = self._foreground(render_cat(0))

    @staticmethod
    def _foreground(frame: RGBFrame) -> NDArray[np.bool_]:
        if frame.shape != (180, 180, 3) or frame.dtype != np.uint8:
            raise ValueError("TemplateDogVision requires a 180×180 RGB uint8 image")
        background = frame[0, 0].astype(np.int16)
        difference = np.abs(frame.astype(np.int16) - background)
        return difference.max(axis=2) > 24

    def classify(self, frame: RGBFrame) -> VisionResult:
        foreground = self._foreground(frame)
        dog_error = float(np.mean(foreground != self._dog))
        cat_error = float(np.mean(foreground != self._cat))
        total_error = dog_error + cat_error
        dog_confidence = (
            0.5
            if total_error == 0
            else 0.5 + (cat_error - dog_error) / (2 * total_error)
        )
        cue = KCCue.A if dog_error < cat_error else KCCue.B
        return VisionResult(cue, dog_confidence)


def _demo_event(
    stage: DemoStage,
    frame: RGBFrame,
    vision: TemplateDogVision,
    response: StepResult[ApproachChoice],
    trial: int,
    trials: int,
) -> DemoEvent:
    return DemoEvent(stage, frame, vision.classify(frame), response, trial, trials)


def run_experiment(
    data_dir: Path,
    backend: str,
    trials: int,
    policy: Path,
    publish: Callable[[DemoEvent], None],
    cancelled: ThreadEvent,
) -> None:
    """Run the real conditioning experiment and publish animation snapshots."""

    vision = TemplateDogVision()
    test_dog = render_dog(DOG_TEST_VARIANT)
    dog_vision = vision.classify(test_dog)
    if dog_vision.cue is not KCCue.A:
        raise RuntimeError("Fixed vision adapter did not classify the dog card")

    with Agent.from_preset(
        "malecns-kc-conditioning",
        data_dir=data_dir,
        backend=backend,
        learning=False,
    ) as baseline:
        baseline_response = baseline.step(dog_vision.cue)
        publish(
            _demo_event(
                DemoStage.BASELINE,
                test_dog,
                vision,
                baseline_response,
                0,
                trials,
            )
        )

    with Agent.from_preset(
        "malecns-kc-conditioning",
        data_dir=data_dir,
        backend=backend,
        learning=True,
    ) as training:
        for trial in range(1, trials + 1):
            if cancelled.is_set():
                return
            variant = DOG_TRAIN_VARIANTS[(trial - 1) % len(DOG_TRAIN_VARIANTS)]
            frame = render_dog(variant)
            detected = vision.classify(frame)
            training.reset(keep_learning=True)
            response = training.step(detected.cue, reward=1.0)
            publish(
                _demo_event(
                    DemoStage.TRAINING,
                    frame,
                    vision,
                    response,
                    trial,
                    trials,
                )
            )
        training.export_policy(policy)

    if cancelled.is_set():
        return
    with Agent.load_policy(
        policy,
        data_dir=data_dir,
        backend=backend,
        preset="malecns-kc-conditioning",
    ) as deployed:
        dog_response = deployed.step(dog_vision.cue)
        publish(
            _demo_event(
                DemoStage.DOG_TEST,
                test_dog,
                vision,
                dog_response,
                trials,
                trials,
            )
        )
        deployed.reset()
        cat_frame = render_cat(CAT_TEST_VARIANT)
        cat_vision = vision.classify(cat_frame)
        cat_response = deployed.step(cat_vision.cue)
        publish(
            _demo_event(
                DemoStage.CAT_TEST,
                cat_frame,
                vision,
                cat_response,
                trials,
                trials,
            )
        )

    if (
        baseline_response.action is not ApproachChoice.AVOID
        or dog_response.action is not ApproachChoice.APPROACH
        or cat_response.action is not ApproachChoice.AVOID
    ):
        raise RuntimeError("Dog conditioning controls did not pass")


def _run_text(args: argparse.Namespace) -> None:
    events: list[DemoEvent] = []
    run_experiment(
        args.data_dir,
        args.backend,
        args.trials,
        args.policy,
        events.append,
        ThreadEvent(),
    )
    for event in events:
        if event.stage is DemoStage.TRAINING and event.trial not in {
            1,
            event.trials,
        }:
            continue
        print(
            f"{event.stage.value:<9} trial={event.trial:>2} "
            f"detector={event.vision.dog_confidence:.0%} dog "
            f"MBON07={event.response.metrics['mbon_spikes']:>3} "
            f"action={event.response.action.value}"
        )


def _run_animation(args: argparse.Namespace) -> None:
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((1180, 680))
    pygame.display.set_caption("FastConnectome — Dog conditioning")
    display_font = pygame.font.SysFont("Avenir Next", 28, bold=True)
    body_font = pygame.font.SysFont("Avenir Next", 17)
    utility_font = pygame.font.SysFont("Menlo", 14)
    clock = pygame.time.Clock()
    queue: Queue[DemoEvent | BaseException | None] = Queue()
    cancelled = ThreadEvent()

    def worker() -> None:
        try:
            run_experiment(
                args.data_dir,
                args.backend,
                args.trials,
                args.policy,
                queue.put,
                cancelled,
            )
        except BaseException as error:
            queue.put(error)
        finally:
            queue.put(None)

    thread = Thread(target=worker, name="dog-conditioning", daemon=True)
    thread.start()
    timeline: list[DemoEvent] = []
    current_index = -1
    next_event_at = 0
    worker_done = False
    failure: BaseException | None = None
    running = True

    navy: Color = (22, 29, 52)
    paper: Color = (244, 240, 228)
    cyan: Color = (79, 194, 210)
    pink: Color = (241, 82, 132)
    lime: Color = (170, 218, 94)
    amber: Color = (241, 163, 64)
    muted: Color = (121, 130, 150)

    def text(
        value: str,
        position: tuple[int, int],
        color: Color,
        font: pygame.font.Font,
    ) -> None:
        surface = font.render(value, True, color)
        screen.blit(surface, position)

    while running:
        for pygame_event in pygame.event.get():
            if pygame_event.type == pygame.QUIT or (
                pygame_event.type == pygame.KEYDOWN
                and pygame_event.key in {pygame.K_ESCAPE, pygame.K_q}
            ):
                running = False
            if (
                pygame_event.type == pygame.KEYDOWN
                and pygame_event.key == pygame.K_r
                and timeline
            ):
                current_index = 0
                next_event_at = pygame.time.get_ticks() + 500

        try:
            while True:
                item = queue.get_nowait()
                if item is None:
                    worker_done = True
                elif isinstance(item, BaseException):
                    failure = item
                else:
                    timeline.append(item)
        except Empty:
            pass

        now = pygame.time.get_ticks()
        if timeline and current_index < 0:
            current_index = 0
            next_event_at = now + 650
        elif current_index + 1 < len(timeline) and now >= next_event_at:
            current_index += 1
            delay = (
                340
                if timeline[current_index].stage is DemoStage.TRAINING
                else 1200
            )
            next_event_at = now + delay

        screen.fill(navy)
        text("GOOD DOG / BAD DOG", (42, 28), paper, display_font)
        text("a tiny associative-learning theatre", (43, 65), cyan, body_font)

        current = timeline[current_index] if current_index >= 0 else None
        pygame.draw.rect(screen, paper, (42, 112, 286, 376), border_radius=18)
        if current is None:
            text("Building the fly brain…", (72, 280), navy, body_font)
        else:
            image = pygame.surfarray.make_surface(current.frame.swapaxes(0, 1))
            scaled_image = pygame.transform.smoothscale(image, (238, 238))
            screen.blit(scaled_image, (66, 136))
            dog_percent = round(current.vision.dog_confidence * 100)
            text(
                f"PIXEL DETECTOR  {dog_percent}% DOG",
                (65, 397),
                navy,
                utility_font,
            )
            pygame.draw.rect(
                screen,
                (205, 207, 198),
                (66, 428, 238, 10),
                border_radius=5,
            )
            pygame.draw.rect(
                screen,
                amber,
                (66, 428, round(238 * current.vision.dog_confidence), 10),
                border_radius=5,
            )
            text(
                f"routes to KC cue {current.vision.cue.name}",
                (66, 453),
                muted,
                body_font,
            )

        centers = ((410, 216), (570, 216), (730, 216), (890, 216))
        labels = (
            "FIXED\nVISION",
            "KENYON\nCELLS",
            "KC > MBON\nSYNAPSES",
            "MBON07\nREADOUT",
        )
        colors = (amber, cyan, pink, lime)
        stages = zip(centers, labels, colors, strict=True)
        for index, (center, label, color) in enumerate(stages):
            if index:
                pygame.draw.line(
                    screen,
                    muted,
                    (centers[index - 1][0] + 54, 216),
                    (center[0] - 54, 216),
                    3,
                )
            pygame.draw.circle(screen, color, center, 55, width=3)
            first, second = label.split("\n")
            first_surface = utility_font.render(first, True, paper)
            second_surface = utility_font.render(second, True, paper)
            first_rect = first_surface.get_rect(center=(center[0], center[1] - 9))
            second_rect = second_surface.get_rect(center=(center[0], center[1] + 11))
            screen.blit(first_surface, first_rect)
            screen.blit(second_surface, second_rect)

        if current is not None:
            phase = (now % 1400) / 1400
            route_start = centers[0][0]
            route_end = centers[-1][0]
            particle_x = round(route_start + phase * (route_end - route_start))
            pygame.draw.circle(screen, paper, (particle_x, 216), 5)
            if current.stage is DemoStage.TRAINING:
                pygame.draw.line(screen, pink, (730, 91), (730, 151), 4)
                pygame.draw.circle(screen, pink, (730, 82), 11)
                text("PAM11 + REWARD", (668, 52), pink, utility_font)

            spikes = int(current.response.metrics["mbon_spikes"])
            threshold = int(current.response.metrics["approach_threshold_spikes"])
            meter_width = min(220, round(220 * spikes / 650))
            pygame.draw.rect(
                screen,
                (52, 61, 86),
                (410, 342, 220, 20),
                border_radius=10,
            )
            pygame.draw.rect(
                screen,
                lime,
                (410, 342, meter_width, 20),
                border_radius=10,
            )
            threshold_x = 410 + round(220 * threshold / 650)
            pygame.draw.line(screen, paper, (threshold_x, 335), (threshold_x, 369), 2)
            text(f"MBON07 {spikes} spikes", (410, 378), paper, utility_font)
            text(
                f"approach threshold {threshold}",
                (410, 401),
                muted,
                utility_font,
            )

            changed = current.response.learning.changed_connections
            text(
                f"{changed:,} changed KC > MBON connections",
                (410, 454),
                pink,
                body_font,
            )
            action_color = (
                lime
                if current.response.action is ApproachChoice.APPROACH
                else amber
            )
            text(
                current.response.action.value.upper(),
                (848, 345),
                action_color,
                display_font,
            )

            fly_x = (
                885 if current.response.action is ApproachChoice.APPROACH else 786
            )
            wing_phase = 7 + round(4 * np.sin(now / 90))
            pygame.draw.ellipse(
                screen,
                cyan,
                (fly_x - 23, 411 - wing_phase, 25, 17),
                width=2,
            )
            pygame.draw.ellipse(
                screen,
                cyan,
                (fly_x + 1, 411 - wing_phase, 25, 17),
                width=2,
            )
            pygame.draw.ellipse(screen, paper, (fly_x - 9, 405, 26, 43))
            pygame.draw.circle(screen, amber, (1025, 425), 31)
            text("DOG", (1007, 416), navy, utility_font)

            if current.stage is DemoStage.BASELINE:
                status = "Before training"
            elif current.stage is DemoStage.TRAINING:
                status = (
                    f"Pairing dog cue + reward · "
                    f"trial {current.trial}/{current.trials}"
                )
            elif current.stage is DemoStage.DOG_TEST:
                status = "Frozen test · new dog card"
            else:
                status = "Frozen control · cat card"
            text(status, (410, 504), paper, display_font)
        elif failure is not None:
            text("Experiment failed", (410, 342), pink, display_font)

        pygame.draw.line(screen, (60, 69, 93), (42, 573), (1138, 573), 1)
        text("THE FLY LEARNS", (42, 598), lime, utility_font)
        text("dog cue = positive value", (42, 622), paper, body_font)
        text("THE FIXED ADAPTER DOES", (385, 598), amber, utility_font)
        text("pixel template > cue A or B", (385, 622), paper, body_font)
        text("NOT CLAIMED", (792, 598), pink, utility_font)
        text("real-world dog recognition", (792, 622), paper, body_font)
        if worker_done and current_index + 1 >= len(timeline):
            text("R replay  ·  Q quit", (965, 38), muted, utility_font)

        pygame.display.flip()
        clock.tick(60)

    cancelled.set()
    thread.join()
    pygame.quit()
    if failure is not None:
        raise RuntimeError("Dog conditioning experiment failed") from failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--backend", choices=["auto", "cpu", "metal"], default="auto")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("runs/dog-conditioning.fcmodel"),
    )
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("trials must be positive")
    if args.no_animation:
        _run_text(args)
    else:
        _run_animation(args)


if __name__ == "__main__":
    main()
