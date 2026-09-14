"""Native Metal runtime binding for MaleCNS spike propagation."""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from threading import Lock
from typing import Final
import weakref

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
Int16Array = NDArray[np.int16]
Int32Array = NDArray[np.int32]
Int64Array = NDArray[np.int64]
UInt8Array = NDArray[np.uint8]

_SOURCE: Final[Path] = Path(__file__).with_name("metal_runtime.swift")
_BUILD_LOCK: Final[Lock] = Lock()
_LIBRARY: ctypes.CDLL | None = None


class MetalUnavailableError(RuntimeError):
    """Raised when the native Metal runtime cannot be loaded on this host."""


def metal_available() -> bool:
    """Return whether this host can compile and expose a default Metal device."""

    if platform.system() != "Darwin" or shutil.which("swiftc") is None:
        return False
    try:
        library = _load_library()
    except MetalUnavailableError:
        return False
    return bool(library.fc_metal_available())


class MetalRuntime:
    """Own Metal buffers and advance the Stonkfly-v1 neural state."""

    def __init__(
        self,
        *,
        ptr: Int64Array,
        post: Int32Array,
        weight: FloatArray,
        v: FloatArray,
        g: FloatArray,
        refractory: Int16Array,
        previous_drive: FloatArray,
        queue: Int32Array,
        queue_count: Int32Array,
        active_flag: UInt8Array,
        last: Int64Array,
        kc_mask: UInt8Array,
        modulation_mask: UInt8Array,
        rest: FloatArray,
        adaptation: FloatArray,
        modulation: FloatArray,
        modulation_last: Int64Array,
        clock: int,
    ) -> None:
        self._library = _load_library()
        self._n = len(v)
        self._edge_count = len(post)
        self._slots = queue.shape[0]
        self._validate_initial_state(
            ptr=ptr,
            post=post,
            weight=weight,
            v=v,
            g=g,
            refractory=refractory,
            previous_drive=previous_drive,
            queue=queue,
            queue_count=queue_count,
            active_flag=active_flag,
            last=last,
            kc_mask=kc_mask,
            modulation_mask=modulation_mask,
            rest=rest,
            adaptation=adaptation,
            modulation=modulation,
            modulation_last=modulation_last,
        )
        handle = self._library.fc_create(
            self._n,
            self._edge_count,
            self._slots,
            clock,
            _pointer(ptr),
            _pointer(post),
            _pointer(weight),
            _pointer(v),
            _pointer(g),
            _pointer(refractory),
            _pointer(previous_drive),
            _pointer(queue),
            _pointer(queue_count),
            _pointer(active_flag),
            _pointer(last),
            _pointer(kc_mask),
            _pointer(modulation_mask),
            _pointer(rest),
            _pointer(adaptation),
            _pointer(modulation),
            _pointer(modulation_last),
        )
        if not handle:
            raise MetalUnavailableError(_last_error(self._library))
        self._handle = int(handle)
        self._finalizer = weakref.finalize(
            self, self._library.fc_destroy, self._handle
        )
        self._counts = np.empty(self._n, dtype=np.int32)
        self._event_indices = np.empty(self._n * 5, dtype=np.int32)
        self._event_clocks = np.empty(self._n * 5, dtype=np.int64)
        self._event_count = np.empty(1, dtype=np.int32)

    def advance(
        self, drive: FloatArray, ticks: int
    ) -> tuple[Int32Array, Int32Array, Int64Array, float]:
        """Advance at most 10 ms and return counts, timed KC events, and GPU time."""

        _require_array(drive, np.dtype(np.float32), (self._n,), "drive")
        if ticks < 1 or ticks > 100:
            raise ValueError("Metal advances must contain 1...100 ticks")
        elapsed = float(
            self._library.fc_advance(
                self._handle,
                _pointer(drive),
                ticks,
                _pointer(self._counts),
                _pointer(self._event_indices),
                _pointer(self._event_clocks),
                _pointer(self._event_count),
            )
        )
        if elapsed < 0:
            raise RuntimeError(_last_error(self._library))
        event_count = int(self._event_count[0])
        return (
            self._counts.copy(),
            self._event_indices[:event_count].copy(),
            self._event_clocks[:event_count].copy(),
            elapsed,
        )

    def read_state(
        self,
        *,
        v: FloatArray,
        g: FloatArray,
        refractory: Int16Array,
        previous_drive: FloatArray,
        queue: Int32Array,
        queue_count: Int32Array,
        active_flag: UInt8Array,
        last: Int64Array,
        adaptation: FloatArray,
        modulation: FloatArray,
        modulation_last: Int64Array,
    ) -> None:
        """Copy checkpointed GPU state into validated NumPy arrays."""

        self._validate_mutable_state(
            v=v,
            g=g,
            refractory=refractory,
            previous_drive=previous_drive,
            queue=queue,
            queue_count=queue_count,
            active_flag=active_flag,
            last=last,
            adaptation=adaptation,
            modulation=modulation,
            modulation_last=modulation_last,
        )
        status = self._library.fc_read_state(
            self._handle,
            _pointer(v),
            _pointer(g),
            _pointer(refractory),
            _pointer(previous_drive),
            _pointer(queue),
            _pointer(queue_count),
            _pointer(active_flag),
            _pointer(last),
            _pointer(adaptation),
            _pointer(modulation),
            _pointer(modulation_last),
        )
        if status != 0:
            raise RuntimeError(_last_error(self._library))

    def write_state(
        self,
        *,
        clock: int,
        v: FloatArray,
        g: FloatArray,
        refractory: Int16Array,
        previous_drive: FloatArray,
        queue: Int32Array,
        queue_count: Int32Array,
        active_flag: UInt8Array,
        last: Int64Array,
        adaptation: FloatArray,
        modulation: FloatArray,
        modulation_last: Int64Array,
    ) -> None:
        """Replace checkpointed GPU state from validated NumPy arrays."""

        self._validate_mutable_state(
            v=v,
            g=g,
            refractory=refractory,
            previous_drive=previous_drive,
            queue=queue,
            queue_count=queue_count,
            active_flag=active_flag,
            last=last,
            adaptation=adaptation,
            modulation=modulation,
            modulation_last=modulation_last,
        )
        status = self._library.fc_write_state(
            self._handle,
            clock,
            _pointer(v),
            _pointer(g),
            _pointer(refractory),
            _pointer(previous_drive),
            _pointer(queue),
            _pointer(queue_count),
            _pointer(active_flag),
            _pointer(last),
            _pointer(adaptation),
            _pointer(modulation),
            _pointer(modulation_last),
        )
        if status != 0:
            raise RuntimeError(_last_error(self._library))

    def write_weights(self, weights: FloatArray) -> None:
        """Replace the complete edge-weight array after reset or restore."""

        _require_array(
            weights, np.dtype(np.float32), (self._edge_count,), "weights"
        )
        status = self._library.fc_write_weights(
            self._handle, _pointer(weights), self._edge_count
        )
        if status != 0:
            raise RuntimeError(_last_error(self._library))

    def update_weights(self, edges: Int64Array, weights: FloatArray) -> None:
        """Update the small plastic-weight overlay after a learning bin."""

        _require_array(edges, np.dtype(np.int64), (len(edges),), "edges")
        _require_array(weights, np.dtype(np.float32), (len(edges),), "weights")
        status = self._library.fc_update_weights(
            self._handle, _pointer(edges), _pointer(weights), len(edges)
        )
        if status != 0:
            raise RuntimeError(_last_error(self._library))

    def close(self) -> None:
        """Release native Metal buffers immediately."""

        self._finalizer()

    def _validate_initial_state(
        self,
        *,
        ptr: Int64Array,
        post: Int32Array,
        weight: FloatArray,
        v: FloatArray,
        g: FloatArray,
        refractory: Int16Array,
        previous_drive: FloatArray,
        queue: Int32Array,
        queue_count: Int32Array,
        active_flag: UInt8Array,
        last: Int64Array,
        kc_mask: UInt8Array,
        modulation_mask: UInt8Array,
        rest: FloatArray,
        adaptation: FloatArray,
        modulation: FloatArray,
        modulation_last: Int64Array,
    ) -> None:
        _require_array(ptr, np.dtype(np.int64), (self._n + 1,), "ptr")
        _require_array(post, np.dtype(np.int32), (self._edge_count,), "post")
        _require_array(
            weight, np.dtype(np.float32), (self._edge_count,), "weight"
        )
        self._validate_mutable_state(
            v=v,
            g=g,
            refractory=refractory,
            previous_drive=previous_drive,
            queue=queue,
            queue_count=queue_count,
            active_flag=active_flag,
            last=last,
            adaptation=adaptation,
            modulation=modulation,
            modulation_last=modulation_last,
        )
        for name, value in (
            ("kc_mask", kc_mask),
            ("modulation_mask", modulation_mask),
        ):
            _require_array(value, np.dtype(np.uint8), (self._n,), name)
        _require_array(rest, np.dtype(np.float32), (self._n,), "rest")

    def _validate_mutable_state(
        self,
        *,
        v: FloatArray,
        g: FloatArray,
        refractory: Int16Array,
        previous_drive: FloatArray,
        queue: Int32Array,
        queue_count: Int32Array,
        active_flag: UInt8Array,
        last: Int64Array,
        adaptation: FloatArray,
        modulation: FloatArray,
        modulation_last: Int64Array,
    ) -> None:
        for name, value in (
            ("v", v),
            ("g", g),
            ("previous_drive", previous_drive),
            ("adaptation", adaptation),
            ("modulation", modulation),
        ):
            _require_array(value, np.dtype(np.float32), (self._n,), name)
        _require_array(
            refractory, np.dtype(np.int16), (self._n,), "refractory"
        )
        _require_array(
            queue, np.dtype(np.int32), (self._slots, self._n), "queue"
        )
        _require_array(
            queue_count, np.dtype(np.int32), (self._slots,), "queue_count"
        )
        _require_array(
            active_flag, np.dtype(np.uint8), (self._n,), "active_flag"
        )
        _require_array(last, np.dtype(np.int64), (self._n,), "last")
        _require_array(
            modulation_last,
            np.dtype(np.int64),
            (self._n,),
            "modulation_last",
        )


def _load_library() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    with _BUILD_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        if platform.system() != "Darwin":
            raise MetalUnavailableError("Metal requires macOS")
        compiler = shutil.which("swiftc")
        if compiler is None:
            raise MetalUnavailableError("Metal requires the Swift compiler")
        source = _SOURCE.read_bytes()
        identity = hashlib.sha256(
            source + platform.machine().encode() + platform.mac_ver()[0].encode()
        ).hexdigest()[:20]
        cache_root = Path(
            os.environ.get(
                "FASTCONNECTOME_CACHE",
                Path.home() / "Library" / "Caches" / "fastconnectome",
            )
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        library_path = cache_root / f"metal-{identity}.dylib"
        if not library_path.exists():
            temporary_handle = tempfile.NamedTemporaryFile(
                prefix="metal-",
                suffix=".partial",
                dir=cache_root,
                delete=False,
            )
            temporary_handle.close()
            temporary = Path(temporary_handle.name)
            command = [
                compiler,
                "-O",
                "-emit-library",
                str(_SOURCE),
                "-o",
                str(temporary),
                "-framework",
                "Metal",
                "-framework",
                "Foundation",
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if completed.returncode != 0:
                temporary.unlink(missing_ok=True)
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise MetalUnavailableError(f"Could not compile Metal runtime: {detail}")
            os.replace(temporary, library_path)
        try:
            library = ctypes.CDLL(str(library_path))
        except OSError as error:
            raise MetalUnavailableError(f"Could not load Metal runtime: {error}") from error
        _configure_library(library)
        _LIBRARY = library
        return library


def _configure_library(library: ctypes.CDLL) -> None:
    pointer = ctypes.c_void_p
    library.fc_metal_available.argtypes = []
    library.fc_metal_available.restype = ctypes.c_int32
    library.fc_last_error.argtypes = []
    library.fc_last_error.restype = ctypes.c_char_p
    library.fc_create.argtypes = [
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int64,
        *([pointer] * 17),
    ]
    library.fc_create.restype = ctypes.c_uint64
    library.fc_destroy.argtypes = [ctypes.c_uint64]
    library.fc_destroy.restype = None
    library.fc_advance.argtypes = [
        ctypes.c_uint64,
        pointer,
        ctypes.c_int32,
        pointer,
        pointer,
        pointer,
        pointer,
    ]
    library.fc_advance.restype = ctypes.c_double
    library.fc_read_state.argtypes = [ctypes.c_uint64, *([pointer] * 11)]
    library.fc_read_state.restype = ctypes.c_int32
    library.fc_write_state.argtypes = [
        ctypes.c_uint64,
        ctypes.c_int64,
        *([pointer] * 11),
    ]
    library.fc_write_state.restype = ctypes.c_int32
    library.fc_write_weights.argtypes = [
        ctypes.c_uint64,
        pointer,
        ctypes.c_int64,
    ]
    library.fc_write_weights.restype = ctypes.c_int32
    library.fc_update_weights.argtypes = [
        ctypes.c_uint64,
        pointer,
        pointer,
        ctypes.c_int32,
    ]
    library.fc_update_weights.restype = ctypes.c_int32


def _pointer(array: NDArray[np.generic]) -> ctypes.c_void_p:
    return ctypes.c_void_p(array.ctypes.data)


def _last_error(library: ctypes.CDLL) -> str:
    raw = library.fc_last_error()
    return "Unknown Metal runtime error" if raw is None else raw.decode("utf-8")


def _require_array(
    array: NDArray[np.generic],
    dtype: np.dtype[np.generic],
    shape: tuple[int, ...],
    name: str,
) -> None:
    if array.dtype != dtype or array.shape != shape or not array.flags.c_contiguous:
        raise ValueError(
            f"{name} must be a contiguous {dtype.name} array with shape {shape}"
        )
