"""Small ctypes binding for WOOF's native Metal resource and submission ABI.

The shared library and metallib are build outputs. This module never invokes a
compiler and never compiles shader source when a user imports WOOF.
"""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from dataclasses import dataclass
from collections import deque
import fcntl
from functools import wraps
import os
from pathlib import Path
import tempfile
import threading
import time
import weakref
from types import MappingProxyType

import numpy as np


_U64 = ctypes.c_uint64
_I32 = ctypes.c_int32
_U32 = ctypes.c_uint32
_HANDLE = _U64
_ABI_MAJOR = 1
_ABI_MINIMUM_MINOR = 7
_MAX_IN_FLIGHT_SUBMISSIONS = 64


class _NativeKernelBatchItem(ctypes.Structure):
    _fields_ = [
        ("entrypoint", ctypes.c_char_p),
        ("cache_identity", ctypes.c_char_p),
        ("exact_policy", _I32),
        ("resource_start", _U32),
        ("resource_count", _U32),
        ("parameter_offset", _U32),
        ("parameter_size", _U32),
        ("grid_size", _U64),
        ("threads_per_group", _U32),
        ("reserved", _U32),
    ]


@dataclass(frozen=True)
class _PreparedKernelCall:
    """Validated, immutable host description of one typed dispatch."""

    entrypoint: str
    cache_identity: str
    exact_policy: bool
    resources: tuple[tuple[int, int, int, int], ...]
    parameter_bytes: bytes
    grid_size: int
    threads_per_group: int
    owners: tuple[object, ...]


def _serialized_enqueue(*, graph: bool):
    """Fence graph submissions against custom Metal queue work."""
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            with self._enqueue_lock:
                # A deferred typed-kernel sequence must be committed before
                # any operation that is not itself captured, so mixed queue
                # APIs retain their original serial ordering.
                if method.__name__ != "submit_kernel":
                    self._flush_deferred_kernel_calls()
                # MPSGraph's queue-run API doesn't expose its committed command
                # buffer. Drain all prior work before graph submission, and
                # drain graph callbacks before returning to caller-owned queue
                # command buffers. Custom-to-custom ordering stays queue-based.
                if graph:
                    self._wait_submissions()
                else:
                    self._wait_graph_submissions()
                deferred_kernel = (
                    method.__name__ == "submit_kernel"
                    and getattr(self._kernel_batch_local, "active", None)
                    is not None)
                if deferred_kernel:
                    self._raise_deferred_error()
                else:
                    self._ensure_submission_capacity()
                self._raise_deferred_error()
                return method(self, *args, **kwargs)
        return wrapped
    return decorate


class _CalcCoefsParams(ctypes.Structure):
    _fields_ = [
        ("struct_size", _U32), ("nz", _U32), ("ny", _U32), ("nx", _U32),
        ("moist_cq", _I32), ("top_lid", _I32), ("wrf_exact", _I32),
        ("dtau", ctypes.c_float), ("epssm", ctypes.c_float),
        ("gravity", ctypes.c_float), ("gamma", ctypes.c_float),
    ]


class _NativeKernelResource(ctypes.Structure):
    _fields_ = [
        ("buffer", _HANDLE), ("byte_offset", _U64),
        ("byte_length", _U64), ("access", _U32), ("reserved", _U32),
    ]


class _AllocationSnapshot(ctypes.Structure):
    _fields_ = [
        ("struct_size", _U32), ("category_count", _U32),
        ("live_allocated_bytes", _U64), ("live_requested_bytes", _U64),
        ("live_high_water_allocated_bytes", _U64),
        ("pending_allocated_bytes", _U64),
        ("live_allocated_by_category", _U64 * 4),
        ("live_requested_by_category", _U64 * 4),
        ("device_current_allocated_bytes", _U64),
        ("fft_plan_cache_entries", _U64),
        ("fft_plan_cache_builds", _U64),
        ("fft_plan_cache_hits", _U64),
        ("fft_plan_cache_evictions", _U64),
        ("fft_graph_executions_started", _U64),
        ("fft_graph_executions_completed", _U64),
        ("fft_graph_executions_failed", _U64),
        ("fft_graph_pending", _U64),
        ("process_phys_footprint_bytes", _U64),
        ("system_available_estimate_bytes", _U64),
        ("memory_observation_validity", _U32),
        ("reserved", _U32),
    ]


_ALLOCATION_CATEGORY_NAMES = (
    "backing", "rk_argument", "fill_argument", "kernel_argument")


def _artifact_path(variable: str, packaged: Path) -> Path:
    override = os.environ.get(variable)
    path = Path(override).expanduser() if override else packaged
    if not path.is_file():
        raise RuntimeError(
            f"Metal backend artifact is missing: {path}; build the native "
            "Metal runtime from a source checkout or install a Metal-enabled "
            "WOOF wheel")
    return path.resolve()


def _acquire_gpu_slot():
    """Hold a GPU slot only when WOOF_METAL_GPU_SLOTS is set."""
    raw_slots = os.environ.get("WOOF_METAL_GPU_SLOTS")
    if raw_slots is None:
        return None
    try:
        slot_count = int(raw_slots)
    except ValueError as exc:
        raise ValueError("WOOF_METAL_GPU_SLOTS must be a positive integer") from exc
    if slot_count < 1:
        raise ValueError("WOOF_METAL_GPU_SLOTS must be a positive integer")
    configured_slot_dir = os.environ.get("WOOF_METAL_SLOT_DIR")
    slot_dir = (Path(configured_slot_dir).expanduser()
                if configured_slot_dir else
                Path(tempfile.gettempdir()) / "woof-gpu-slots")
    slot_dir.mkdir(parents=True, exist_ok=True)
    while True:
        for index in range(slot_count):
            slot_path = slot_dir / f"slot-{index:03d}.lock"
            stream = slot_path.open("a+b")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                continue
            return stream
        time.sleep(0.5)


class MetalContext:
    """A single Metal device, serial compute queue, and cached pipelines."""

    def __init__(self, *, library: str | os.PathLike | None = None,
                 metallib: str | os.PathLike | None = None,
                 exact_metallib: str | os.PathLike | None = None,
                 batch_kernel_calls: bool = True):
        if not isinstance(batch_kernel_calls, bool):
            raise TypeError("batch_kernel_calls must be a boolean")
        package_root = Path(__file__).resolve().parents[1]
        from .bundle_integrity import verify_metal_bundle
        dylib_path = _artifact_path(
            "WOOF_METAL_RUNTIME_LIB",
            Path(library) if library else package_root /
            "libexec" / "metal" / "libwoof_metal.dylib")
        shader_path = _artifact_path(
            "WOOF_METAL_SHADER_LIBRARY",
            Path(metallib) if metallib else package_root /
            "libexec" / "metal" / "woof_default.metallib")
        exact_shader_path = _artifact_path(
            "WOOF_METAL_EXACT_SHADER_LIBRARY",
            Path(exact_metallib) if exact_metallib else package_root /
            "libexec" / "metal" / "woof_exact.metallib")
        verify_metal_bundle(package_root.parent, artifact_paths={
            "libwoof_metal.dylib": dylib_path,
            "woof_default.metallib": shader_path,
            "woof_exact.metallib": exact_shader_path,
        })
        self._lib = ctypes.CDLL(str(dylib_path))
        self._bind()
        native_abi = self._lib.woof_metal_abi_version()
        native_major, native_minor = native_abi >> 16, native_abi & 0xFFFF
        if (native_major != _ABI_MAJOR
                or native_minor < _ABI_MINIMUM_MINOR):
            raise RuntimeError(
                f"WOOF Metal ABI mismatch: Python expects major {_ABI_MAJOR}, "
                f"native runtime reports {native_abi}")
        self._gpu_slot_file = _acquire_gpu_slot()
        handle = _HANDLE()
        try:
            self._check(self._lib.woof_metal_context_create(
                os.fsencode(shader_path), os.fsencode(exact_shader_path),
                ctypes.byref(handle)))
        except BaseException:
            if self._gpu_slot_file is not None:
                self._gpu_slot_file.close()
            raise
        self._handle = handle.value
        name = ctypes.create_string_buffer(256)
        registry_id = _U64()
        max_buffer = _U64()
        working_set = _U64()
        self._check(self._lib.woof_metal_context_device_info(
            self._handle, name, len(name), ctypes.byref(registry_id),
            ctypes.byref(max_buffer), ctypes.byref(working_set)))
        self.device_name = name.value.decode("utf-8")
        self.device_registry_id = registry_id.value
        self.max_buffer_length = max_buffer.value
        self.recommended_working_set_size = working_set.value
        self.batch_kernel_calls = batch_kernel_calls
        self._buffers = weakref.WeakSet()
        self._submission_lock = threading.RLock()
        self._enqueue_lock = threading.RLock()
        self._kernel_batch_local = threading.local()
        self._seen_kernel_pipelines: set[tuple[str, bool, str]] = set()
        # Keep native submission tokens alive until poll/wait retires them.
        # A weak set let fire-and-forget WOOF kernels lose their Python token
        # before the next enqueue, leaking native status records and making
        # deferred command errors unobservable.
        self._submissions: set[MetalSubmission] = set()
        self._submission_order = deque()
        self._graph_submissions: set[MetalSubmission] = set()
        self._deferred_errors: list[Exception] = []
        self._fatal_error: Exception | None = None
        self._closed = False

    def warm_pipelines(self) -> tuple[int, tuple[str, ...]]:
        """Build every kernel pipeline in the bundle.

        Returns the number built and the kernels this GPU cannot build. The
        system keeps built pipelines in its cache, so later processes skip
        the driver compile a kernel otherwise pays at first use.
        """
        if self._closed:
            raise RuntimeError("Metal context is closed")
        count = _U32()
        failed = ctypes.create_string_buffer(1 << 16)
        self._check(self._lib.woof_metal_warm_pipelines(
            self._handle, ctypes.byref(count), failed, len(failed)))
        return count.value, tuple(failed.value.decode().split())

    def allocation_snapshot(self):
        """Return a read-only snapshot of WOOF-owned Metal buffer storage.

        Allocated-byte values use MTLBuffer.allocatedSize. Requested-byte
        values use each WOOF allocation request. ``pending_allocated_bytes``
        is the unique-byte subset referenced by unfinished submissions, so it
        is already included in ``live_allocated_bytes``. Device-wide allocated
        bytes, process physical footprint and the system available-memory
        estimate are point-in-time observations. The system estimate sums
        free, inactive and speculative VM pages. MPSGraph plan counts describe
        WOOF's bounded shape-plan cache; its private working storage remains
        opaque. No field is an admission guarantee or auto-sizing input.
        """
        if self._closed:
            raise RuntimeError("Metal context is closed")
        snapshot = _AllocationSnapshot()
        snapshot.struct_size = ctypes.sizeof(_AllocationSnapshot)
        self._check(self._lib.woof_metal_context_allocation_snapshot(
            self._handle, ctypes.byref(snapshot)))
        allocated = MappingProxyType(dict(zip(
            _ALLOCATION_CATEGORY_NAMES, snapshot.live_allocated_by_category)))
        requested = MappingProxyType(dict(zip(
            _ALLOCATION_CATEGORY_NAMES, snapshot.live_requested_by_category)))
        return MappingProxyType({
            "live_allocated_bytes": snapshot.live_allocated_bytes,
            "live_requested_bytes": snapshot.live_requested_bytes,
            "live_high_water_allocated_bytes":
                snapshot.live_high_water_allocated_bytes,
            "pending_allocated_bytes": snapshot.pending_allocated_bytes,
            "live_allocated_by_category": allocated,
            "live_requested_by_category": requested,
            "device_current_allocated_bytes":
                snapshot.device_current_allocated_bytes,
            "fft_plan_cache": MappingProxyType({
                "entries": snapshot.fft_plan_cache_entries,
                "builds": snapshot.fft_plan_cache_builds,
                "hits": snapshot.fft_plan_cache_hits,
                "evictions": snapshot.fft_plan_cache_evictions,
            }),
            "fft_graph_executions": MappingProxyType({
                "started": snapshot.fft_graph_executions_started,
                "completed": snapshot.fft_graph_executions_completed,
                "failed": snapshot.fft_graph_executions_failed,
                "pending": snapshot.fft_graph_pending,
            }),
            "process_phys_footprint_bytes":
                snapshot.process_phys_footprint_bytes,
            "system_available_estimate_bytes":
                snapshot.system_available_estimate_bytes,
            "memory_observation_validity": snapshot.memory_observation_validity,
        })


    def _bind(self) -> None:
        lib = self._lib
        lib.woof_metal_last_error.restype = ctypes.c_char_p
        lib.woof_metal_abi_version.argtypes = []
        lib.woof_metal_abi_version.restype = _U32

        lib.woof_metal_context_create.argtypes = [ctypes.c_char_p,
                                                  ctypes.c_char_p,
                                                  ctypes.POINTER(_HANDLE)]
        lib.woof_metal_context_create.restype = _I32
        lib.woof_metal_context_destroy.argtypes = [_HANDLE]
        lib.woof_metal_context_destroy.restype = _I32
        lib.woof_metal_context_device_info.argtypes = [
            _HANDLE, ctypes.c_char_p, _U64, ctypes.POINTER(_U64),
            ctypes.POINTER(_U64), ctypes.POINTER(_U64)]
        lib.woof_metal_context_device_info.restype = _I32
        lib.woof_metal_context_allocation_snapshot.argtypes = [
            _HANDLE, ctypes.POINTER(_AllocationSnapshot)]
        lib.woof_metal_context_allocation_snapshot.restype = _I32

        lib.woof_metal_buffer_create.argtypes = [
            _HANDLE, _U64, ctypes.POINTER(_HANDLE)]
        lib.woof_metal_buffer_create.restype = _I32
        lib.woof_metal_buffer_destroy.argtypes = [_HANDLE]
        lib.woof_metal_buffer_destroy.restype = _I32
        lib.woof_metal_buffer_write.argtypes = [
            _HANDLE, _U64, ctypes.c_void_p, _U64]
        lib.woof_metal_buffer_write.restype = _I32
        lib.woof_metal_buffer_read.argtypes = [
            _HANDLE, _U64, ctypes.c_void_p, _U64]
        lib.woof_metal_buffer_read.restype = _I32
        lib.woof_metal_buffer_size.argtypes = [
            _HANDLE, ctypes.POINTER(_U64)]
        lib.woof_metal_buffer_size.restype = _I32

        handle_pointer = ctypes.POINTER(_HANDLE)
        offset_pointer = ctypes.POINTER(_U64)
        lib.woof_metal_submit_rk_bookkeeping.argtypes = [
            _HANDLE, _I32, _U32, handle_pointer, offset_pointer,
            handle_pointer, offset_pointer, offset_pointer,
            ctypes.POINTER(_HANDLE)]
        lib.woof_metal_submit_rk_bookkeeping.restype = _I32
        lib.woof_metal_submit_fill.argtypes = [
            _HANDLE, _HANDLE, _U64, _U64, ctypes.POINTER(_U32), _U32,
            ctypes.POINTER(_HANDLE)]
        lib.woof_metal_submit_fill.restype = _I32
        lib.woof_metal_submit_acoustic_calc_coefs.argtypes = [
            _HANDLE, handle_pointer, offset_pointer,
            ctypes.POINTER(_CalcCoefsParams), ctypes.POINTER(_HANDLE)]
        lib.woof_metal_submit_acoustic_calc_coefs.restype = _I32
        lib.woof_metal_submit_kernel.argtypes = [
            _HANDLE, ctypes.c_char_p, ctypes.c_char_p, _I32,
            ctypes.POINTER(_NativeKernelResource), _U32,
            ctypes.c_void_p, _U32,
            _U64, _U32, ctypes.POINTER(_HANDLE)]
        lib.woof_metal_submit_kernel.restype = _I32
        lib.woof_metal_submit_kernel_batch.argtypes = [
            _HANDLE, ctypes.POINTER(_NativeKernelBatchItem), _U32,
            ctypes.POINTER(_NativeKernelResource), _U32,
            ctypes.c_void_p, _U32, ctypes.POINTER(_HANDLE)]
        lib.woof_metal_submit_kernel_batch.restype = _I32
        lib.woof_metal_submit_fft2.argtypes = [
            _HANDLE, _HANDLE, _U64, ctypes.POINTER(_U64),
            _HANDLE, _U64, ctypes.POINTER(_U64), _U32, _I32,
            ctypes.POINTER(_HANDLE)]
        lib.woof_metal_submit_fft2.restype = _I32
        lib.woof_metal_warm_pipelines.argtypes = [
            _HANDLE, ctypes.POINTER(_U32), ctypes.c_char_p, _U32]
        lib.woof_metal_warm_pipelines.restype = _I32

        lib.woof_metal_submission_poll.argtypes = [
            _HANDLE, ctypes.POINTER(_I32)]
        lib.woof_metal_submission_poll.restype = _I32
        lib.woof_metal_submission_wait.argtypes = [_HANDLE]
        lib.woof_metal_submission_wait.restype = _I32
        lib.woof_metal_submission_destroy.argtypes = [_HANDLE]
        lib.woof_metal_submission_destroy.restype = _I32

    def _check(self, code: int) -> None:
        if code:
            raw = self._lib.woof_metal_last_error()
            message = raw.decode("utf-8", errors="replace") if raw else "unknown error"
            raise RuntimeError(f"Metal backend error {code}: {message}")

    def buffer(self, size: int) -> MetalBuffer:
        if self._closed:
            raise RuntimeError("Metal context is closed")
        handle = _HANDLE()
        self._check(self._lib.woof_metal_buffer_create(
            self._handle, size, ctypes.byref(handle)))
        buffer = MetalBuffer(self, handle.value, size)
        self._buffers.add(buffer)
        return buffer

    def submit_fill(self, target: MetalBuffer, word_count: int,
                    word_value: int, *, target_offset: int = 0) -> MetalSubmission:
        return self.submit_fill_pattern(
            target, word_count, (word_value,), target_offset=target_offset)

    @_serialized_enqueue(graph=False)
    def submit_fill_pattern(self, target: MetalBuffer, word_count: int,
                            pattern, *, target_offset: int = 0) -> MetalSubmission:
        if self._closed:
            raise RuntimeError("Metal context is closed")
        self._require_context(target)
        pattern = tuple(int(word) & 0xffffffff for word in pattern)
        if not 1 <= len(pattern) <= 4:
            raise ValueError("Metal fill pattern must contain 1 to 4 words")
        if word_count % len(pattern):
            raise ValueError("Metal fill range must contain whole patterns")
        self._reap_submissions()
        self._raise_deferred_error()
        pattern_words = (_U32 * len(pattern))(*pattern)
        handle = _HANDLE()
        self._check(self._lib.woof_metal_submit_fill(
            self._handle, target.handle, target_offset, word_count,
            pattern_words, len(pattern), ctypes.byref(handle)))
        return MetalSubmission(self, handle.value, (target,))

    @_serialized_enqueue(graph=False)
    def submit_bookkeeping(self, pairs, *, zero: bool = False) -> MetalSubmission:
        if self._closed:
            raise RuntimeError("Metal context is closed")
        pairs = tuple(pairs)
        if not pairs or len(pairs) > 64:
            raise ValueError("Metal RK bookkeeping needs 1 to 64 array pairs")
        self._reap_submissions()
        self._raise_deferred_error()
        src_handles = (_HANDLE * len(pairs))()
        src_offsets = (_U64 * len(pairs))()
        dst_handles = (_HANDLE * len(pairs))()
        dst_offsets = (_U64 * len(pairs))()
        word_counts = (_U64 * len(pairs))()
        owners = []
        for index, (source, target) in enumerate(pairs):
            self._require_context(target.buffer)
            self._require_context(source.buffer) if source is not None else None
            if source is not None:
                src_handles[index] = source.buffer.handle
                src_offsets[index] = source.byte_offset
                owners.append(source.buffer)
            dst_handles[index] = target.buffer.handle
            dst_offsets[index] = target.byte_offset
            word_counts[index] = target.nbytes // 4
            owners.append(target.buffer)
        handle = _HANDLE()
        source_ptr = src_handles if not zero else None
        source_offset_ptr = src_offsets if not zero else None
        self._check(self._lib.woof_metal_submit_rk_bookkeeping(
            self._handle, int(zero), len(pairs), source_ptr,
            source_offset_ptr, dst_handles, dst_offsets, word_counts,
            ctypes.byref(handle)))
        return MetalSubmission(self, handle.value, tuple(owners))

    @_serialized_enqueue(graph=False)
    def submit_acoustic_calc_coefs(self, arrays, *, nz: int, ny: int, nx: int,
                                   moist_cq: bool, top_lid: bool,
                                   wrf_exact: bool, dtau: float, epssm: float,
                                   gravity: float, gamma: float) -> MetalSubmission:
        if self._closed:
            raise RuntimeError("Metal context is closed")
        from .array import MetalArray
        arrays = tuple(arrays)
        if len(arrays) != 15 or not all(isinstance(a, MetalArray)
                                        for a in arrays):
            raise TypeError("calc_coefs requires exactly 15 Metal arrays")
        if any(a.dtype != np.dtype(np.float32)
               or not a.flags.c_contiguous for a in arrays):
            raise TypeError("calc_coefs resources must be contiguous float32 arrays")
        for array in arrays:
            self._require_context(array.buffer)
        self._reap_submissions()
        self._raise_deferred_error()
        handles = (_HANDLE * len(arrays))(
            *(array.buffer.handle for array in arrays))
        offsets = (_U64 * len(arrays))(
            *(array.byte_offset for array in arrays))
        parameters = _CalcCoefsParams(
            ctypes.sizeof(_CalcCoefsParams), nz, ny, nx,
            int(moist_cq), int(top_lid), int(wrf_exact),
            dtau, epssm, gravity, gamma)
        handle = _HANDLE()
        self._check(self._lib.woof_metal_submit_acoustic_calc_coefs(
            self._handle, handles, offsets, ctypes.byref(parameters),
            ctypes.byref(handle)))
        return MetalSubmission(
            self, handle.value, tuple(array.buffer for array in arrays))

    @_serialized_enqueue(graph=False)
    def submit_kernel(self, entrypoint: str, arrays, parameters: dict, *,
                      exact_policy: bool, grid_size: int,
                      threads_per_group: int | None = None
                      ) -> MetalSubmission | None:
        """Submit or defer one typed-argument-buffer kernel dispatch."""
        call = self.prepare_kernel_call(
            entrypoint, arrays, parameters, exact_policy=exact_policy,
            grid_size=grid_size, threads_per_group=threads_per_group)
        batch = getattr(self._kernel_batch_local, "active", None)
        if batch is None:
            return self._submit_kernel_batch((call,), label=entrypoint)
        key = self._kernel_pipeline_key(call)
        if key not in self._seen_kernel_pipelines:
            # Resolve first-use pipelines in their original order and commit
            # each before buffering warmed dispatches. This lets GPU work
            # overlap later pipeline creation on a cold context.
            self._flush_deferred_kernel_calls()
            self._ensure_submission_capacity()
            return self._submit_kernel_batch(
                (call,), label=f"{batch['label']} pipeline warm-up: {entrypoint}")
        batch["calls"].append(call)
        if len(batch["calls"]) == 64:
            self._flush_deferred_kernel_calls()
        return None

    @staticmethod
    def _kernel_pipeline_key(call: _PreparedKernelCall):
        return (call.cache_identity, call.exact_policy, call.entrypoint)

    @contextmanager
    def defer_kernel_submissions(self, label: str):
        """Batch repeated typed kernels in a scoped serial queue sequence.

        First-use pipelines submit singly so compilation can overlap later
        preparation. Warm dispatches are committed in batches of at most 64,
        matching the native descriptor budget. Other Metal submission APIs
        flush this queue before they enqueue, preserving cross-API ordering.

        Callers must discard the return values of captured kernel submissions;
        first-use submissions still return their real token.
        """
        if not isinstance(label, str) or not label.strip():
            raise ValueError("deferred Metal sequence label must be non-empty")
        with self._enqueue_lock:
            if not self.batch_kernel_calls:
                # Preserve sequence exclusion while submitting each kernel
                # separately for synchronous verification.
                yield
                return
            if getattr(self._kernel_batch_local, "active", None) is not None:
                raise RuntimeError("nested deferred Metal sequences are unsupported")
            batch = {"label": label.strip(), "calls": [], "chunk": 0}
            self._kernel_batch_local.active = batch
            try:
                yield
            finally:
                try:
                    self._flush_deferred_kernel_calls()
                finally:
                    del self._kernel_batch_local.active

    def _flush_deferred_kernel_calls(self) -> None:
        batch = getattr(self._kernel_batch_local, "active", None)
        if batch is None or not batch["calls"]:
            return
        calls = tuple(batch["calls"])
        batch["calls"].clear()
        batch["chunk"] += 1
        self._ensure_submission_capacity()
        self._submit_kernel_batch(
            calls, label=f"{batch['label']} chunk {batch['chunk']}")

    def prepare_kernel_call(self, entrypoint: str, arrays, parameters: dict, *,
                            exact_policy: bool, grid_size: int,
                            threads_per_group: int | None = None
                            ) -> _PreparedKernelCall:
        """Validate/freeze a typed dispatch without committing GPU work."""
        if self._closed:
            raise RuntimeError("Metal context is closed")
        from .generated import kernel_specs
        from .array import MetalArray

        if not isinstance(entrypoint, str) or not entrypoint:
            raise ValueError("Metal entrypoint must be a non-empty string")
        if (isinstance(grid_size, bool) or not isinstance(grid_size, int)
                or grid_size <= 0 or grid_size > 0xFFFFFFFFFFFFFFFF):
            raise ValueError("Metal dispatch grid must be a positive uint64")
        if not isinstance(exact_policy, (bool, np.bool_)):
            raise TypeError("exact_policy must be a boolean")

        prefix = entrypoint.upper()
        specs = getattr(kernel_specs, f"{prefix}_RESOURCES", None)
        access = getattr(kernel_specs, f"{prefix}_ACCESS", None)
        shape_specs = getattr(kernel_specs, f"{prefix}_SHAPES", None)
        fields = getattr(kernel_specs, f"{prefix}_PARAM_FIELDS", None)
        if specs is None or access is None or shape_specs is None or fields is None:
            raise ValueError(f"unregistered Metal kernel entry point {entrypoint!r}")
        if not isinstance(arrays, dict) or set(arrays) != {row[0] for row in specs}:
            raise ValueError(f"{entrypoint} resource names do not match its registry")
        if not isinstance(parameters, dict) or set(parameters) != {row[0] for row in fields}:
            raise ValueError(f"{entrypoint} scalar names do not match its registry")
        integer_parameters = {
            name: int(value) for name, value in parameters.items()
            if isinstance(value, (int, np.integer))}
        # Nest registrations carry their refinement ratios in the lengths
        # of the SINT offset tables, not in scalar kernel arguments.  The
        # generated resource shape contracts name those dimensions ``nri``
        # and ``nrj``; resolve them from their already typed resources so
        # runtime validation can check the manifest-bound geometry without
        # adding fields to the kernel ABI.
        for dimension, resource in (("nri", "xig_a"), ("nrj", "xjg_a")):
            if dimension not in integer_parameters and resource in arrays:
                integer_parameters[dimension] = int(arrays[resource].size)
        # RRTM's packed coefficient blob carries its lengths as typed Metal
        # resources rather than scalar ABI arguments. Resolve those shape
        # contracts directly from the corresponding resources.
        for dimension, resource, measure in (
                ("ntable_values", "tables", "size"),
                ("ntables", "table_index", "shape0"),
                # RRTMGP's standalone Planck ABI carries the flavor table,
                # but not its row count. Keep the registered shape check.
                ("nflav", "flavor", "shape0")):
            if dimension not in integer_parameters and resource in arrays:
                value = arrays[resource]
                integer_parameters[dimension] = int(
                    value.size if measure == "size" else value.shape[0])
        if "taug" in arrays:
            for dimension, axis in (("ncol", 0), ("nlay", 1)):
                if dimension not in integer_parameters:
                    integer_parameters[dimension] = int(
                        arrays["taug"].shape[axis])
        if ("count" not in integer_parameters
                and all(name in integer_parameters for name in
                        ("nlevel", "nlatitude", "nlongitude"))):
            # A flattened Global grid resource can declare its extent as
            # ``count`` while carrying the three grid dimensions in the ABI.
            # Resolve the product here so the shape contract checks the
            # actual resident-array length before dispatch.
            integer_parameters["count"] = (
                integer_parameters["nlevel"]
                * integer_parameters["nlatitude"]
                * integer_parameters["nlongitude"])

        def extent(expression):
            if isinstance(expression, int) and expression >= 0:
                return expression
            if isinstance(expression, str) and expression.isdecimal():
                return int(expression)
            if expression in integer_parameters:
                return integer_parameters[expression]
            for operator in ("+", "-"):
                if operator in expression:
                    variable, amount = expression.split(operator, 1)
                    if variable in integer_parameters:
                        delta = int(amount)
                        return (integer_parameters[variable] + delta
                                if operator == "+" else
                                integer_parameters[variable] - delta)
            raise ValueError(
                f"{entrypoint} has unsupported shape expression {expression!r}")

        shape_rules = dict(shape_specs)
        owners = []
        resource_records = []
        for index, (name, storage_type, access_mode) in enumerate(specs):
            array = arrays[name]
            storage_dtypes = {"float": np.dtype(np.float32),
                              "float32": np.dtype(np.float32),
                              "int32": np.dtype(np.int32),
                              "uint32": np.dtype(np.uint32),
                              "int64": np.dtype(np.int64),
                              "uint64": np.dtype(np.uint64),
                              # Resource-only MSL ulong view over binary64
                              # storage. This does not request FP64 arithmetic.
                              "float64_bits": np.dtype(np.float64)}
            expected_dtype = storage_dtypes.get(storage_type)
            if expected_dtype is None:
                raise TypeError(f"{entrypoint}.{name} has unsupported storage {storage_type}")
            if (not isinstance(array, MetalArray) or array.dtype != expected_dtype
                    or not array.flags.c_contiguous):
                raise TypeError(
                    f"{entrypoint}.{name} requires contiguous {expected_dtype} storage")
            expected_shapes = tuple(
                tuple(extent(expr) for expr in shape)
                for shape in shape_rules[name]
                if "any_1d" not in shape and "any_2d" not in shape)
            if (entrypoint == "nest_bdy_interp1_fragment"
                    and name in {"value_a", "tendency_a"}
                    and integer_parameters.get("side", 0) < 2):
                # The shared boundary-table contract keeps west/east as
                # (nz, nyc, sz), while the generated fragment's generic
                # nlong expression spells (nz, sz, nyc). The shader's flat
                # addressing follows the former; validate that side-specific
                # layout here without changing its scalar ABI.
                expected_shapes += ((integer_parameters["nz"],
                                     integer_parameters["nyc"],
                                     integer_parameters["sz"]),)
            matches_any_1d = any(shape == ("any_1d",)
                                 for shape in shape_rules[name])
            matches_any_2d = any(shape == ("any_2d",)
                                 for shape in shape_rules[name])
            if (array.shape not in expected_shapes
                    and not (matches_any_1d and array.ndim == 1)
                    and not (matches_any_2d and array.ndim == 2)):
                raise ValueError(
                    f"{entrypoint}.{name} shape {array.shape} is outside "
                    f"the registered shapes {expected_shapes}")
            self._require_context(array.buffer)
            resource_records.append((
                array.buffer.handle, array.byte_offset, array.nbytes,
                3 if access_mode == "read_write" else 1))
            owners.append(array.buffer)

        ctypes_types = {
            "ctypes.c_float": ctypes.c_float,
            "ctypes.c_int32": ctypes.c_int32,
        }
        parameter_type = type(
            f"_{prefix}Params", (ctypes.Structure,),
            {"_fields_": [(name, ctypes_types[type_name])
                          for name, type_name in fields]})
        packed = parameter_type(*(parameters[name] for name, _ in fields))
        parameter_size = ctypes.sizeof(parameter_type)
        expected_size = 4 * len(fields)
        if (parameter_size != expected_size or parameter_size > 4096):
            raise RuntimeError(f"{entrypoint} parameter layout has unexpected padding")
        layout_hash = getattr(kernel_specs, f"{prefix}_LAYOUT_HASH")
        shader_hash = getattr(kernel_specs, f"{prefix}_SHADER_HASH")
        policy = "wrf-exact" if exact_policy else "default"
        cache_identity = f"{shader_hash}:{layout_hash}:{policy}"
        threads = (getattr(kernel_specs, f"{prefix}_THREADS_PER_GROUP")
                   if threads_per_group is None else int(threads_per_group))
        if not 1 <= threads <= 0xFFFFFFFF:
            raise ValueError("threads_per_group must be a positive uint32")
        return _PreparedKernelCall(
            entrypoint=entrypoint, cache_identity=cache_identity,
            exact_policy=bool(exact_policy), resources=tuple(resource_records),
            parameter_bytes=ctypes.string_at(ctypes.byref(packed),
                                              parameter_size),
            grid_size=grid_size, threads_per_group=threads,
            owners=tuple(owners))

    @_serialized_enqueue(graph=False)
    def submit_kernel_batch(self, calls, *, label: str,
                            first_use_singletons=()) -> MetalSubmission:
        """Submit a short sequence, optionally isolating first-use dispatches.

        A cold Metal context creates compute pipelines synchronously. Callers
        can keep selected edge dispatches out of the initial batch so their
        pipeline creation overlaps device work from the adjacent batch. Once
        cached, those dispatches rejoin the full batch.
        """
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Metal batch label must be non-empty")
        calls = tuple(calls)
        if not calls or len(calls) > 64:
            raise ValueError("Metal kernel batch requires 1 to 64 dispatches")
        if any(not isinstance(call, _PreparedKernelCall) for call in calls):
            raise TypeError("Metal kernel batches require prepared dispatches")
        if sum(len(call.resources) for call in calls) > 4096:
            raise ValueError("Metal kernel batch exceeds 4096 resources")

        batch_label = label.strip()
        first_use_singletons = tuple(first_use_singletons)
        if any(not isinstance(call, _PreparedKernelCall)
               for call in first_use_singletons):
            raise TypeError("first-use singletons require prepared dispatches")
        isolated_keys = {
            self._kernel_pipeline_key(call) for call in first_use_singletons}
        call_keys = {self._kernel_pipeline_key(call) for call in calls}
        if not isolated_keys.issubset(call_keys):
            raise ValueError("first-use singletons must belong to this batch")
        if not isolated_keys:
            return self._submit_kernel_batch(calls, label=batch_label)

        submissions = []
        buffered = []

        def flush_buffered():
            if not buffered:
                return
            self._ensure_submission_capacity()
            submissions.append(self._submit_kernel_batch(
                tuple(buffered), label=batch_label))
            buffered.clear()

        for call in calls:
            key = self._kernel_pipeline_key(call)
            if (key not in self._seen_kernel_pipelines
                    and key in isolated_keys):
                flush_buffered()
                self._ensure_submission_capacity()
                submissions.append(self._submit_kernel_batch(
                    (call,), label=(
                        f"{batch_label} pipeline warm-up: {call.entrypoint}")))
                continue
            buffered.append(call)
        flush_buffered()

        # The final token completes after earlier command buffers on this
        # serial queue. Preserve the batch-level metadata expected by callers.
        submission = submissions[-1]
        submission.label = batch_label
        submission.entrypoints = tuple(call.entrypoint for call in calls)
        return submission

    @_serialized_enqueue(graph=False)
    def submit_prepared_kernel_call(self, call: _PreparedKernelCall, *,
                                    label: str) -> MetalSubmission:
        """Submit a frozen dispatch using the original single-kernel ABI."""
        if not isinstance(call, _PreparedKernelCall):
            raise TypeError("single dispatch requires a prepared kernel call")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Metal submission label must be non-empty")
        resources = (_NativeKernelResource * len(call.resources))(
            *(_NativeKernelResource(*row, 0) for row in call.resources))
        encoded_entrypoint = call.entrypoint.encode("utf-8")
        encoded_identity = call.cache_identity.encode("ascii")
        params = ctypes.create_string_buffer(call.parameter_bytes)
        handle = _HANDLE()
        self._check(self._lib.woof_metal_submit_kernel(
            self._handle, encoded_entrypoint, encoded_identity,
            int(call.exact_policy), resources, len(call.resources),
            ctypes.cast(params, ctypes.c_void_p), len(call.parameter_bytes),
            call.grid_size, call.threads_per_group, ctypes.byref(handle)))
        self._seen_kernel_pipelines.add(self._kernel_pipeline_key(call))
        return MetalSubmission(self, handle.value, call.owners,
                               label=label.strip(),
                               entrypoints=(call.entrypoint,))

    def _submit_kernel_batch(self, calls, *, label: str) -> MetalSubmission:
        if self._closed:
            raise RuntimeError("Metal context is closed")
        if not calls or len(calls) > 64:
            raise ValueError("Metal kernel batch requires 1 to 64 dispatches")
        if any(not isinstance(call, _PreparedKernelCall) for call in calls):
            raise TypeError("Metal kernel batches require prepared dispatches")
        items = (_NativeKernelBatchItem * len(calls))()
        all_resources = [resource for call in calls
                         for resource in call.resources]
        if len(all_resources) > 4096:
            raise ValueError("Metal kernel batch exceeds 4096 resources")
        resource_records = (_NativeKernelResource * len(all_resources))()
        parameters = bytearray()
        encoded_names = []
        cursor = 0
        for index, call in enumerate(calls):
            entrypoint = call.entrypoint.encode("utf-8")
            cache_identity = call.cache_identity.encode("ascii")
            encoded_names.extend((entrypoint, cache_identity))
            resource_start = sum(len(prev.resources) for prev in calls[:index])
            parameter_offset = len(parameters)
            if (resource_start > 0xFFFFFFFF
                    or parameter_offset > 0xFFFFFFFF
                    or len(call.parameter_bytes) > 0xFFFFFFFF):
                raise ValueError("Metal kernel batch offsets exceed uint32")
            items[index] = _NativeKernelBatchItem(
                entrypoint, cache_identity, int(call.exact_policy),
                resource_start, len(call.resources), parameter_offset,
                len(call.parameter_bytes), call.grid_size,
                call.threads_per_group, 0)
            parameters.extend(call.parameter_bytes)
        for index, row in enumerate(all_resources):
            resource_records[index] = _NativeKernelResource(*row, 0)
        parameter_buffer = ctypes.create_string_buffer(bytes(parameters))
        self._raise_deferred_error()
        handle = _HANDLE()
        self._check(self._lib.woof_metal_submit_kernel_batch(
            self._handle, items, len(calls), resource_records,
            len(all_resources), ctypes.cast(parameter_buffer, ctypes.c_void_p),
            len(parameters), ctypes.byref(handle)))
        owners = tuple(owner for call in calls for owner in call.owners)
        entrypoints = tuple(call.entrypoint for call in calls)
        submission = MetalSubmission(
            self, handle.value, owners, label=label, entrypoints=entrypoints)
        self._seen_kernel_pipelines.update(
            self._kernel_pipeline_key(call) for call in calls)
        return submission

    @_serialized_enqueue(graph=True)
    def submit_fft2(self, source, target, *, inverse: bool) -> MetalSubmission:
        """Submit a contiguous FP32/Complex64 2D FFT on the last two axes."""
        from .array import MetalArray

        if self._closed:
            raise RuntimeError("Metal context is closed")
        if not isinstance(source, MetalArray) or not isinstance(target, MetalArray):
            raise TypeError("Metal FFT inputs and outputs must be MetalArray values")
        expected_source = np.dtype(np.complex64 if inverse else np.float32)
        expected_target = np.dtype(np.float32 if inverse else np.complex64)
        if source.dtype != expected_source or target.dtype != expected_target:
            raise TypeError(
                f"{'irfft2' if inverse else 'rfft2'} requires "
                f"{expected_source} input and {expected_target} output")
        if (source.ndim < 2 or source.ndim != target.ndim
                or not source.flags.c_contiguous or not target.flags.c_contiguous):
            raise ValueError("Metal FFT requires contiguous arrays with rank at least two")
        if source.byte_offset != 0 or target.byte_offset != 0:
            raise ValueError("MPSGraph FFT requires base arrays with zero byte offsets")
        for a, b in zip(source.shape[:-2], target.shape[:-2]):
            if a != b:
                raise ValueError("FFT input/output batch dimensions do not match")
        if source.shape[-2] != target.shape[-2]:
            raise ValueError("FFT input/output row dimensions do not match")
        if inverse:
            if source.shape[-1] != target.shape[-1] // 2 + 1:
                raise ValueError("irfft2 input must have nx//2+1 half-spectrum bins")
        elif target.shape[-1] != source.shape[-1] // 2 + 1:
            raise ValueError("rfft2 output width must be nx//2+1")
        self._require_context(source.buffer)
        self._require_context(target.buffer)
        input_shape = (_U64 * source.ndim)(*source.shape)
        output_shape = (_U64 * target.ndim)(*target.shape)
        self._reap_submissions()
        self._raise_deferred_error()
        handle = _HANDLE()
        self._check(self._lib.woof_metal_submit_fft2(
            self._handle, source.buffer.handle, source.byte_offset, input_shape,
            target.buffer.handle, target.byte_offset, output_shape,
            source.ndim, int(bool(inverse)), ctypes.byref(handle)))
        return MetalSubmission(self, handle.value,
                               (source.buffer, target.buffer), graph=True)

    def _remember_submission(self, submission: MetalSubmission) -> None:
        with self._submission_lock:
            self._submissions.add(submission)
            self._submission_order.append(submission)
            if submission._graph:
                self._graph_submissions.add(submission)

    def _completed_submission(self, submission: MetalSubmission) -> None:
        with self._submission_lock:
            self._submissions.discard(submission)
            self._graph_submissions.discard(submission)
            try:
                self._submission_order.remove(submission)
            except ValueError:
                pass

    def _record_failure(self, error: Exception) -> None:
        with self._submission_lock:
            if self._fatal_error is None:
                self._fatal_error = error

    def _ensure_submission_capacity(self) -> None:
        self._reap_submissions()
        while True:
            with self._submission_lock:
                if len(self._submissions) < _MAX_IN_FLIGHT_SUBMISSIONS:
                    return
                oldest = (self._submission_order[0]
                          if self._submission_order else None)
            if oldest is None:
                raise RuntimeError(
                    "Metal submission tracking exceeded its in-flight bound")
            oldest.wait()

    def _wait_graph_submissions(self) -> None:
        with self._submission_lock:
            submissions = tuple(self._graph_submissions)
        failures = []
        for submission in submissions:
            try:
                submission.wait()
            except Exception as error:
                failures.append(error)
                self._record_failure(error)
        with self._submission_lock:
            failures.extend(self._deferred_errors)
            self._deferred_errors.clear()
        for error in failures:
            self._record_failure(error)
        if failures:
            raise RuntimeError(
                f"{len(failures)} Metal graph submission(s) failed; "
                f"first error: {failures[0]}") from failures[0]

    def _reap_submissions(self) -> None:
        # A serial queue makes every host transfer a barrier for all earlier
        # work. Reap completed tokens on every enqueue so device-only loops do
        # not accumulate finished native command buffers between readbacks.
        with self._submission_lock:
            submissions = tuple(self._submission_order)
        for submission in submissions:
            try:
                submission.poll()
            except Exception as error:
                self._record_failure(error)
                with self._submission_lock:
                    self._deferred_errors.append(error)

    def _wait_submissions(self) -> None:
        self._flush_deferred_kernel_calls()
        with self._submission_lock:
            submissions = tuple(self._submissions)
        failures = []
        for submission in submissions:
            try:
                submission.wait()
            except Exception as error:
                failures.append(error)
                self._record_failure(error)
        with self._submission_lock:
            failures.extend(self._deferred_errors)
            self._deferred_errors.clear()
        for error in failures:
            self._record_failure(error)
        if failures:
            raise RuntimeError(
                f"{len(failures)} Metal command submission(s) failed; "
                f"first error: {failures[0]}") from failures[0]

    def _raise_deferred_error(self) -> None:
        with self._submission_lock:
            error = self._fatal_error
            if error is None and self._deferred_errors:
                error = self._deferred_errors[0]
                self._fatal_error = error
        if error is not None:
            raise RuntimeError(
                f"Metal context is latched after a submission failure: {error}") from error

    def _require_context(self, buffer: MetalBuffer) -> None:
        if (not isinstance(buffer, MetalBuffer) or buffer.context is not self
                or buffer._closed):
            raise ValueError("Metal buffer belongs to a different device context")

    def close(self) -> None:
        if self._closed:
            return
        wait_error = None
        try:
            self._wait_submissions()
        except Exception as error:
            wait_error = error
        # Explicit context shutdown invalidates any surviving array views,
        # but releases every native allocation before destroying the device.
        # All context submissions are complete before buffer handles retire.
        for buffer in tuple(self._buffers):
            buffer.close()
        self._check(self._lib.woof_metal_context_destroy(self._handle))
        self._closed = True
        if self._gpu_slot_file is not None:
            self._gpu_slot_file.close()
        if wait_error is not None:
            raise wait_error

    def __enter__(self) -> MetalContext:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class MetalBuffer:
    """Owned Metal allocation; array views retain this allocation object."""

    __slots__ = ("context", "handle", "size", "_closed", "__weakref__")

    def __init__(self, context: MetalContext, handle: int, size: int):
        self.context = context
        self.handle = handle
        self.size = size
        self._closed = False

    def write(self, data: bytes | bytearray | memoryview, *, offset: int = 0) -> None:
        if self._closed:
            raise RuntimeError("Metal buffer is closed")
        self.context._wait_submissions()
        self.context._raise_deferred_error()
        raw = bytes(data)
        if offset < 0 or offset + len(raw) > self.size:
            raise ValueError("Metal buffer write is out of bounds")
        source = ctypes.create_string_buffer(raw) if raw else None
        pointer = ctypes.cast(source, ctypes.c_void_p) if source else None
        self.context._check(self.context._lib.woof_metal_buffer_write(
            self.handle, offset, pointer, len(raw)))

    def read(self, size: int | None = None, *, offset: int = 0) -> bytes:
        if self._closed:
            raise RuntimeError("Metal buffer is closed")
        self.context._wait_submissions()
        self.context._raise_deferred_error()
        size = self.size - offset if size is None else size
        if offset < 0 or size < 0 or offset + size > self.size:
            raise ValueError("Metal buffer read is out of bounds")
        if size == 0:
            return b""
        destination = ctypes.create_string_buffer(size)
        self.context._check(self.context._lib.woof_metal_buffer_read(
            self.handle, offset, destination, size))
        return destination.raw[:size]

    def close(self) -> None:
        if self._closed:
            return
        # The native submission owns a strong MTLBuffer reference until the
        # command completes. Releasing this logical handle must not turn array
        # destruction into an implicit GPU synchronization point.
        self.context._check(self.context._lib.woof_metal_buffer_destroy(self.handle))
        self._closed = True

    def __del__(self):
        if not getattr(self, "_closed", True):
            try:
                self.close()
            except Exception:
                pass


class MetalSubmission:
    """Asynchronous command submission retaining every participating buffer."""

    __slots__ = ("context", "handle", "_owners", "_complete",
                 "_state_lock", "_graph", "label", "entrypoints",
                 "_failure", "__weakref__")

    def __init__(self, context: MetalContext, handle: int, owners, *,
                 graph: bool = False, label: str = "",
                 entrypoints: tuple[str, ...] = ()):
        self.context = context
        self.handle = handle
        self._owners = tuple(weakref.ref(owner) for owner in owners)
        self._complete = False
        self._graph = bool(graph)
        self.label = label
        self.entrypoints = tuple(entrypoints)
        self._failure: Exception | None = None
        self._state_lock = threading.RLock()
        self.context._remember_submission(self)

    def _annotate_failure(self, error: Exception) -> Exception:
        if self._failure is not None:
            return self._failure
        identity = ",".join(self.entrypoints) if self.entrypoints else "unknown operation"
        label = self.label or "unlabeled submission"
        self._failure = RuntimeError(
            f"Metal submission {label!r} [{identity}] failed: {error}")
        return self._failure

    def _finish(self) -> None:
        with self._state_lock:
            if self._complete:
                return
            try:
                self.context._check(
                    self.context._lib.woof_metal_submission_destroy(self.handle))
            except Exception as error:
                failure = self._annotate_failure(error)
                self.context._record_failure(failure)
                raise failure from error
            finally:
                self._complete = True
                self.context._completed_submission(self)
                self._owners = ()

    def poll(self) -> bool:
        with self._state_lock:
            if self._complete:
                if self._failure is not None:
                    raise self._failure
                return True
            complete = _I32()
            try:
                self.context._check(self.context._lib.woof_metal_submission_poll(
                    self.handle, ctypes.byref(complete)))
            except Exception as error:
                failure = self._annotate_failure(error)
                self.context._record_failure(failure)
                try:
                    self._finish()
                except Exception:
                    pass
                raise failure from error
            if complete.value:
                self._finish()
            return bool(complete.value)

    def wait(self) -> None:
        with self._state_lock:
            if self._complete:
                if self._failure is not None:
                    raise self._failure
                return
            try:
                self.context._check(
                    self.context._lib.woof_metal_submission_wait(self.handle))
            except Exception as error:
                failure = self._annotate_failure(error)
                self.context._record_failure(failure)
                try:
                    self._finish()
                except Exception:
                    pass
                raise failure from error
            else:
                self._finish()
