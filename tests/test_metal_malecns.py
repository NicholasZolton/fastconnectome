import os
from pathlib import Path

import numpy as np
import pytest

from fastconnectome.models.malecns import CurrentPulse, MaleCNS, MetalMaleCNS
from fastconnectome.models.malecns.metal import metal_available

DATA_DIR = os.environ.get("FASTCONNECTOME_MALECNS_DATA")


@pytest.mark.skipif(
    DATA_DIR is None or not metal_available(),
    reason="requires prepared MaleCNS data and Metal",
)
def test_metal_matches_reward_learning_and_checkpoint(
    tmp_path: Path,
) -> None:
    if DATA_DIR is None:
        pytest.fail("skip condition failed to require MaleCNS data")
    frame = np.full((120, 200, 3), 128, dtype=np.uint8)
    frame[30:70, 90:120] = (255, 20, 20)
    pulses = [
        None,
        CurrentPulse("PAM11", 50, 20, "positive"),
        None,
        CurrentPulse("PPL101", 50, 20, "negative"),
        CurrentPulse("PAM11", 50, 20, "positive"),
    ]
    cpu = MaleCNS(DATA_DIR, learning=True)
    metal = MetalMaleCNS(DATA_DIR, learning=True)
    try:
        for pulse in pulses:
            cpu_result = cpu.advance(frame, pulse, 20)
            metal_result = metal.advance(frame, pulse, 20)
            assert np.array_equal(
                cpu_result.activity.counts,
                metal_result.activity.counts,
            )
            assert cpu_result.learning_summary == metal_result.learning_summary

        assert np.array_equal(cpu._brain.memory_u, metal._brain.memory_u)
        assert np.array_equal(cpu._brain.memory_w, metal._brain.memory_w)
        assert np.array_equal(
            cpu._brain.weight[cpu._plastic_edges],
            metal._brain.weight[metal._plastic_edges],
        )

        checkpoint = tmp_path / "metal-simulator.fccheckpoint"
        metal.save(checkpoint)
        expected = metal.advance(frame, None, 20)
        metal.restore(checkpoint)
        actual = metal.advance(frame, None, 20)
        assert np.array_equal(expected.activity.counts, actual.activity.counts)
        assert expected.learning_summary == actual.learning_summary
        assert expected.metrics == actual.metrics
    finally:
        metal.close()
