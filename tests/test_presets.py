from pathlib import Path

import pytest

from fastconnectome.presets import build_preset
from fastconnectome.models.malecns import MaleCNS


def test_preset_rejects_unknown_backend_without_changing_dynamics() -> None:
    with pytest.raises(ValueError, match="backend"):
        build_preset(
            "malecns-visual-turning",
            data_dir=Path("missing"),
            dynamics="stonkfly-v1",
            backend="quantum",
        )


def test_auto_backend_prefers_metal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fastconnectome.presets.metal_available", lambda: True)

    from fastconnectome.presets import resolve_backend

    assert resolve_backend("auto") == "metal"


def test_auto_backend_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fastconnectome.presets.metal_available", lambda: False)

    from fastconnectome.presets import resolve_backend

    assert resolve_backend("auto") == "cpu"


def test_explicit_metal_requires_host_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("fastconnectome.presets.metal_available", lambda: False)

    from fastconnectome.presets import resolve_backend

    with pytest.raises(RuntimeError, match="unavailable"):
        resolve_backend("metal")


def test_policy_configuration_can_cross_execution_backends() -> None:
    cpu = {"kind": "fastconnectome-policy", "model": {"backend": "cpu"}}
    metal = {"kind": "fastconnectome-policy", "model": {"backend": "metal"}}

    assert MaleCNS._policy_manifest_matches(cpu, metal)

    different_dynamics = {
        "kind": "fastconnectome-policy",
        "model": {"backend": "metal", "dynamics": "other"},
    }
    assert not MaleCNS._policy_manifest_matches(cpu, different_dynamics)
