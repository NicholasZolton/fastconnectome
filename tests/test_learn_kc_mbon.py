from pathlib import Path

import numpy as np

from fastconnectome import ApproachChoice, KCCue
from fastconnectome.models.malecns import (
    KCCueEncoder,
    MBONApproach,
    NeuralActivity,
    PopulationIndex,
)


def mbon_activity(spikes: int) -> NeuralActivity:
    return NeuralActivity(
        np.array([spikes], dtype=np.int32),
        PopulationIndex(
            ids=np.array([10], dtype=np.int64),
            cell_types=np.array(["MBON07"], dtype=np.str_),
            sides=np.array([""], dtype=np.str_),
        ),
    )


def test_kc_cue_encoder_hides_declared_population_current() -> None:
    encoder = KCCueEncoder()

    stimulus = encoder.encode(KCCue.A)

    assert stimulus.frame.shape == (180, 320, 3)
    assert stimulus.currents[0].population == "KCab-m"
    assert stimulus.currents[0].current == 40


def test_mbon_decoder_has_a_fixed_boundary(tmp_path: Path) -> None:
    decoder = MBONApproach()

    assert decoder.decode(mbon_activity(479), 1000).action is ApproachChoice.AVOID
    assert decoder.decode(mbon_activity(480), 1000).action is ApproachChoice.APPROACH

    checkpoint = tmp_path / "decoder.npz"
    decoder.save(checkpoint)
    decoder.restore(checkpoint)
