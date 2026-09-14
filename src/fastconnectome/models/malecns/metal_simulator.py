"""MaleCNS simulator with Metal propagation and CPU learning traces."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fastconnectome.models.malecns.metal import MetalRuntime
from fastconnectome.models.malecns.simulator import MaleCNS, RGBFrame
from fastconnectome.types import ModelInfo


class MetalMaleCNS(MaleCNS):
    """Run Stonkfly-v1 spikes on Metal while retaining its CPU plasticity rule."""

    def __init__(
        self,
        data_dir: str | Path = Path("data"),
        *,
        learning: bool = True,
        neural_bin_ms: float = 10.0,
    ) -> None:
        super().__init__(data_dir, learning=learning, neural_bin_ms=neural_bin_ms)
        self._metal = MetalRuntime(
            ptr=self._brain.ptr,
            post=self._brain.post,
            weight=self._brain.weight,
            v=self._brain.v,
            g=self._brain.g,
            refractory=self._brain.refractory,
            previous_drive=self._brain.previous_drive,
            queue=self._brain.queue,
            queue_count=self._brain.queue_count,
            active_flag=self._brain.active_flag,
            last=self._brain.last,
            kc_mask=self._brain.circuit["kc_mask"],
            modulation_mask=self._brain.modulation_mask,
            rest=self._brain.rest,
            adaptation=self._brain.adaptation,
            modulation=self._brain.modulation,
            modulation_last=self._brain.modulation_last,
            clock=self._brain.cursor,
        )
        self._info = ModelInfo(
            model="malecns",
            release="v1.0",
            dynamics="stonkfly-v1",
            backend="metal",
            neurons=self._brain.n,
            connections=len(self._brain.post),
        )

    def _rgb_step(
        self,
        stimulus: RGBFrame,
        duration_ms: float,
        stimulation: tuple[NDArray[np.int32], float] | None,
    ) -> tuple[NDArray[np.int32], float]:
        from stonkfly.neural.rule import advance
        from stonkfly.neural.sensory import retinal_samples

        frame = np.asarray(stimulus)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("RGB uint8 required")
        ticks = round(duration_ms / self._brain.dt)
        if ticks < 1 or ticks > 100:
            raise ValueError("Metal neural bins must contain 1...100 ticks")

        height, width = frame.shape[:2]
        x = np.minimum(
            (self._brain.r8_uv[:, 0] * (width - 1)).astype(int), width - 1
        )
        y = np.minimum(
            (self._brain.r8_uv[:, 1] * (height - 1)).astype(int), height - 1
        )
        values = frame[y, x, self._brain.r8_channel].astype(np.float32) / 255
        values = np.where(
            values <= 0.04045,
            values / 12.92,
            ((values + 0.055) / 1.055) ** 2.4,
        ).astype(np.float32)
        visual_alpha = 1 - math.exp(-ticks * self._brain.dt / 10)
        self._brain.r8_light += visual_alpha * (values - self._brain.r8_light)

        light = retinal_samples(frame, self._brain.uv)
        self._brain.luminance += visual_alpha * (
            np.clip(light, 0, 1) - self._brain.luminance
        )
        self._brain.drive.fill(0)
        self._brain.drive[self._brain.lamina] = 12.0
        self._brain.drive[self._brain.retina] = (
            30 * self._brain.luminance / (0.02 + self._brain.luminance)
        )
        self._brain.drive += self._brain.tonic
        self._brain.drive[self._brain.r8] += (
            30 * self._brain.r8_light / (0.02 + self._brain.r8_light)
        )
        if stimulation is not None:
            indices, current = stimulation
            self._brain.drive[indices] += np.float32(current)

        counts, event_indices, event_clocks, elapsed = self._metal.advance(
            self._brain.drive, ticks
        )
        self._advance_eligibility(event_indices, event_clocks)
        seconds = duration_ms / 1000
        circuit = self._brain.circuit
        advance(
            self._brain.rate_kc,
            self._brain.rate_dan,
            self._brain.memory_u,
            self._brain.memory_w,
            counts[circuit["pre"]] / seconds,
            counts[circuit["dan"]] / seconds - self._brain.dan_baseline_hz,
            circuit["gain"],
            seconds,
            self._brain.eta,
            self.learning,
            self._brain.weights_frozen,
        )
        if not self._brain.weights_frozen:
            plastic_weights = self._brain.baseline_plastic * (
                1 + self._brain.memory_w
            )
            self._brain.weight[self._plastic_edges] = plastic_weights
            self._metal.update_weights(
                self._plastic_edges,
                np.ascontiguousarray(
                    self._brain.weight[self._plastic_edges], dtype=np.float32
                ),
            )

        self._brain.cursor += ticks
        self._brain.sim_ms = self._brain.cursor * self._brain.dt
        self._brain.total_spikes += int(counts.sum())
        self._brain.counts[:] = counts
        return counts, elapsed

    def reset(self, *, keep_learning: bool = False) -> None:
        super().reset(keep_learning=keep_learning)
        self._write_metal_state()

    def save(self, path: Path) -> None:
        self._read_metal_state()
        super().save(path)

    def restore(self, path: Path) -> None:
        super().restore(path)
        self._write_metal_state()

    def import_policy(self, path: Path, manifest: dict[str, object]) -> None:
        super().import_policy(path, manifest)
        self._write_metal_state()

    def close(self) -> None:
        """Release the backend's Metal allocations."""

        self._metal.close()

    def _advance_eligibility(
        self, indices: NDArray[np.int32], clocks: NDArray[np.int64]
    ) -> None:
        tau_ms = float(self._brain.rule_parameters["trace_kc_seconds"]) * 1000
        for raw_index, raw_clock in zip(indices, clocks, strict=True):
            index = int(raw_index)
            clock = int(raw_clock)
            elapsed = self._brain.dt * (
                clock - int(self._brain.eligibility_last[index])
            )
            self._brain.eligibility[index] *= math.exp(-elapsed / tau_ms)
            self._brain.eligibility[index] += 1
            self._brain.eligibility_last[index] = clock

    def _read_metal_state(self) -> None:
        self._metal.read_state(
            v=self._brain.v,
            g=self._brain.g,
            refractory=self._brain.refractory,
            previous_drive=self._brain.previous_drive,
            queue=self._brain.queue,
            queue_count=self._brain.queue_count,
            active_flag=self._brain.active_flag,
            last=self._brain.last,
            adaptation=self._brain.adaptation,
            modulation=self._brain.modulation,
            modulation_last=self._brain.modulation_last,
        )
        active = np.flatnonzero(self._brain.active_flag).astype(np.int32)
        self._brain.active.fill(0)
        self._brain.active[: len(active)] = active
        self._brain.nactive[0] = len(active)

    def _write_metal_state(self) -> None:
        self._metal.write_state(
            clock=self._brain.cursor,
            v=self._brain.v,
            g=self._brain.g,
            refractory=self._brain.refractory,
            previous_drive=self._brain.previous_drive,
            queue=self._brain.queue,
            queue_count=self._brain.queue_count,
            active_flag=self._brain.active_flag,
            last=self._brain.last,
            adaptation=self._brain.adaptation,
            modulation=self._brain.modulation,
            modulation_last=self._brain.modulation_last,
        )
        self._metal.write_weights(self._brain.weight)
