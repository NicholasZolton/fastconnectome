from pathlib import Path

import pytest

from fastconnectome.presets import build_preset


def test_preset_rejects_backend_without_changing_dynamics() -> None:
    with pytest.raises(ValueError, match="backend"):
        build_preset(
            "malecns-visual-turning",
            data_dir=Path("missing"),
            dynamics="stonkfly-v1",
            backend="metal",
        )
