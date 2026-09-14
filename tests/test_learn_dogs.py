from fastconnectome import KCCue
from learn_dogs import TemplateDogVision, render_cat, render_dog


def test_template_vision_routes_synthetic_dog_variants_to_one_cue() -> None:
    vision = TemplateDogVision()

    dog_results = [vision.classify(render_dog(variant)) for variant in range(3)]
    cat_results = [vision.classify(render_cat(variant)) for variant in range(3)]

    assert all(result.cue is KCCue.A for result in dog_results)
    assert all(result.dog_confidence > 0.5 for result in dog_results)
    assert all(result.cue is KCCue.B for result in cat_results)
    assert all(result.dog_confidence < 0.5 for result in cat_results)
