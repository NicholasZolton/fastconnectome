from learn_kc_mbon import APPROACH_THRESHOLD, Choice, decode_choice


def test_mbon_decoder_has_a_fixed_boundary() -> None:
    assert decode_choice(APPROACH_THRESHOLD - 1) == Choice.AVOID
    assert decode_choice(APPROACH_THRESHOLD) == Choice.APPROACH
