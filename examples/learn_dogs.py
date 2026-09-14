"""Run a fixed dog detector through native KC→MBON conditioning."""

from __future__ import annotations

import argparse
import base64
import json
import struct
import webbrowser
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Event as ThreadEvent

import numpy as np
from numpy.typing import NDArray

from fastconnectome import Agent, ApproachChoice, KCCue
from fastconnectome.types import StepResult

RGBFrame = NDArray[np.uint8]
Color = tuple[int, int, int]
BACKGROUND: Color = (225, 235, 238)
DOG_TRAIN_VARIANTS = (0, 1)
DOG_TEST_VARIANT = 2
CAT_TEST_VARIANT = 2
REQUIRED_DATA_FILES = ("graph.npz", "annotations.feather")
REPORT_TEMPLATE = Path(__file__).with_name("dog_conditioning_report.html")


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


def validate_data_dir(data_dir: Path) -> Path:
    """Resolve a prepared MaleCNS directory before running the experiment."""

    resolved = data_dir.expanduser().resolve()
    missing = [name for name in REQUIRED_DATA_FILES if not (resolved / name).is_file()]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(
            f"MaleCNS data is not prepared in {resolved} (missing: {names}).\n"
            "Prepare it with:\n"
            "  uv run fastconnectome prepare malecns-v1 "
            f"--data-dir {resolved}\n"
            "Or pass --data-dir pointing to an existing prepared dataset."
        )
    return resolved


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
    """Run the conditioning experiment and publish each recorded state."""

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


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data)
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _frame_data_uri(frame: RGBFrame) -> str:
    """Encode a synthetic RGB card without adding an image dependency."""

    height, width, channels = frame.shape
    if channels != 3 or frame.dtype != np.uint8:
        raise ValueError("Report frames must be RGB uint8 images")
    scanlines = b"".join(b"\x00" + frame[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )
    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _report_event(event: DemoEvent) -> dict[str, str | int | float]:
    return {
        "stage": event.stage.value,
        "image": _frame_data_uri(event.frame),
        "confidence": event.vision.dog_confidence,
        "cue": event.vision.cue.name,
        "spikes": int(event.response.metrics["mbon_spikes"]),
        "threshold": int(event.response.metrics["approach_threshold_spikes"]),
        "changed": event.response.learning.changed_connections,
        "action": event.response.action.value,
        "trial": event.trial,
        "trials": event.trials,
    }


def write_report(events: list[DemoEvent], destination: Path) -> Path:
    """Write a self-contained browser report for one completed experiment."""

    template = REPORT_TEMPLATE.read_text()
    serialized = json.dumps(
        [_report_event(event) for event in events],
        separators=(",", ":"),
    )
    if "__FASTCONNECTOME_EVENTS__" not in template:
        raise RuntimeError("Dog conditioning report template is missing its data marker")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.replace("__FASTCONNECTOME_EVENTS__", serialized))
    return destination.resolve()


def _run_browser(args: argparse.Namespace) -> None:
    events: list[DemoEvent] = []
    print(f"Running {args.trials} conditioning trials…")
    run_experiment(
        args.data_dir,
        args.backend,
        args.trials,
        args.policy,
        events.append,
        ThreadEvent(),
    )
    report = write_report(events, args.report)
    print(f"Wrote {report}")
    if not args.no_open:
        webbrowser.open(report.as_uri())


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
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/dog-conditioning.html"),
    )
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--text", "--no-animation", dest="text", action="store_true")
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("trials must be positive")
    try:
        args.data_dir = validate_data_dir(args.data_dir)
    except FileNotFoundError as error:
        parser.error(str(error))
    if args.text:
        _run_text(args)
    else:
        _run_browser(args)


if __name__ == "__main__":
    main()
