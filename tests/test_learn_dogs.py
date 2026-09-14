from pathlib import Path

import pytest

from fastconnectome import ApproachChoice, KCCue
from fastconnectome.types import (
    ActivitySummary,
    LearningSummary,
    StepResult,
    Timing,
)
from learn_dogs import (
    DemoEvent,
    DemoStage,
    TemplateDogVision,
    VisionResult,
    render_cat,
    render_dog,
    validate_data_dir,
    write_report,
)


def response(
    action: ApproachChoice,
    spikes: int,
    changed_connections: int,
) -> StepResult[ApproachChoice]:
    return StepResult(
        action=action,
        activity=ActivitySummary(spikes),
        learning=LearningSummary(True, changed_connections),
        timing=Timing(1000, 0.1, 0.1),
        metrics={"mbon_spikes": spikes, "approach_threshold_spikes": 480},
    )


def test_template_vision_routes_synthetic_dog_variants_to_one_cue() -> None:
    vision = TemplateDogVision()

    dog_results = [vision.classify(render_dog(variant)) for variant in range(3)]
    cat_results = [vision.classify(render_cat(variant)) for variant in range(3)]

    assert all(result.cue is KCCue.A for result in dog_results)
    assert all(result.dog_confidence > 0.5 for result in dog_results)
    assert all(result.cue is KCCue.B for result in cat_results)
    assert all(result.dog_confidence < 0.5 for result in cat_results)


def test_data_directory_is_checked_before_running(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fastconnectome prepare"):
        validate_data_dir(tmp_path)

    (tmp_path / "graph.npz").touch()
    (tmp_path / "annotations.feather").touch()

    assert validate_data_dir(tmp_path) == tmp_path.resolve()


def test_browser_report_embeds_experiment_data_and_images(tmp_path: Path) -> None:
    events = [
        DemoEvent(
            DemoStage.BASELINE,
            render_dog(2),
            VisionResult(KCCue.A, 0.79),
            response(ApproachChoice.AVOID, 212, 0),
            0,
            20,
        ),
        DemoEvent(
            DemoStage.DOG_TEST,
            render_dog(2),
            VisionResult(KCCue.A, 0.79),
            response(ApproachChoice.APPROACH, 513, 4153),
            20,
            20,
        ),
        DemoEvent(
            DemoStage.CAT_TEST,
            render_cat(2),
            VisionResult(KCCue.B, 0.17),
            response(ApproachChoice.AVOID, 442, 4153),
            20,
            20,
        ),
    ]

    report = write_report(events, tmp_path / "conditioning.html")
    contents = report.read_text()

    assert "__FASTCONNECTOME_EVENTS__" not in contents
    assert '"stage":"dog-test"' in contents
    assert '"spikes":513' in contents
    assert "data:image/png;base64," in contents
