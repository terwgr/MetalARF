"""Hash-bound Level-5 checkpoints for atmosphere, surface, and physics state."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace
import json
import math
import os
from pathlib import Path

import numpy as np

from .constants import (
    CHECKPOINT_SCHEMA,
    GRID_TRACERS,
    LEVEL4_CHECKPOINT_SCHEMA,
    PHYSICS_STATE_SCHEMA,
    PROGNOSTIC_FIELDS,
    SEMILAG_CHECKPOINT_SCHEMA,
    SPECTRAL_FIELDS,
    SPECTRAL_TRACER_CHECKPOINT_SCHEMA,
)
from .semilag.state import TRAJECTORY_FIELDS, TrajectoryState
from .pins import (
    DEFAULT_INTEGRATOR,
    DEFAULT_SEMI_IMPLICIT_SCHEME,
    ACCEPTED_PINS_HASHES,
    KNOWN_PINS_HASHES,
    INSPECTABLE_RETIRED_PINS_HASHES,
    RETIRED_SEMILAG_PINS_HASHES,
    SPECTRAL_TRACER_ERA_PINS_HASHES,
    accepted_pins_hashes,
    arithmetic_label,
    pins_hash,
    scheme_of_pins_hash,
)
from .spill import spilled
from .state import (
    SEEDED_SURFACE_MEMBERS,
    SURFACE_ARRAY_NAMES,
    ArwenGlobalState,
    MoistHybridState,
    PhysicsState,
    SurfaceState,
)

TRACKER_KEYS = (
    "maximum_spectral_cfl",
    "maximum_mass_fixer_log_offset",
    "maximum_global_water_fixer_kg_m2",
    "maximum_repaired_negative_mixing_ratio",
    "maximum_semi_implicit_divergence_increment_s1",
    "maximum_physics_water_repair_kg_m2",
    "maximum_native_water_residual_kg_m2",
    "maximum_native_energy_residual_j_m2",
)

# The surface inventory is the state's own table: a SurfaceState member
# added there (the 2026-09-01 static fields) is checkpointed here without
# a second hand-kept list drifting from the first.
_SURFACE_MAP = SURFACE_ARRAY_NAMES


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _array_hash(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(arr.dtype.str.encode())
    digest.update(str(arr.shape).encode())
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def _sha(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def normalize_trackers(value: dict[str, object] | None = None) -> dict[str, float]:
    if value is None:
        return {name: 0.0 for name in TRACKER_KEYS}
    if not isinstance(value, dict) or set(value) != set(TRACKER_KEYS):
        raise ValueError(f"run trackers must contain exactly {list(TRACKER_KEYS)}")
    result: dict[str, float] = {}
    for name in TRACKER_KEYS:
        if isinstance(value[name], bool):
            raise ValueError(f"tracker {name} must be a finite nonnegative number")
        number = float(value[name])
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"tracker {name} must be finite and nonnegative")
        result[name] = number
    return result


def _checkpoint_array(value, to_numpy) -> np.ndarray:
    """One checkpointed array on the host, owned by the checkpoint.

    An array the pinned host tier holds is ALREADY on the host, so the
    device read the checkpoint would have paid is not paid: the tier's
    outstanding transfers are drained and the slot is copied.  The copy
    is not optional.  The writer runs on its own thread and outlives the
    step that produced these values, while the tier's slot is written
    again by the very next physics call, so handing the writer the slot
    itself would hash a state that is half this step and half the next.
    """
    if spilled(value):
        value.tier.synchronize()
        return np.array(value.host, copy=True)
    from woof.metal.array import MetalArray

    if isinstance(value, MetalArray):
        return value.to_numpy()
    return np.asarray(to_numpy(value))


def bundle_arrays(
    bundle: ArwenGlobalState, to_numpy, trajectory=None, *,
    card_gather=None,
) -> dict[str, np.ndarray]:
    """Every checkpointed array: the five spectral fields as complex
    coefficients, the ten grid tracers as real Gaussian-grid arrays
    (schema v3), the surface and the physics namespace, and -- when the
    run's integrator carries a second time level -- its seven trajectory
    arrays under their own namespace (schema v4).

    ``trajectory`` is None for every integrator that carries no second
    level, and also for the FIRST checkpoint of a semi-Lagrangian run,
    which is written before any step has produced one; that archive is a
    v3 archive and resuming from it takes the same non-extrapolated
    start-up step the uninterrupted run takes there, so the restart is
    still the continuation of the run.

    The grid tracers, the surface and the physics namespace may live in
    the pinned host tier rather than on the card; the format, the array
    names, the per-array hashes and ``self_sha256`` do not know which,
    and a checkpoint written either way is byte-identical.
    """
    bundle.physics_state.validate()
    arrays = {
        f"atmosphere__{name}": _checkpoint_array(
            getattr(bundle.atmosphere, name), to_numpy)
        for name in PROGNOSTIC_FIELDS
    }
    arrays.update({
        f"surface__{name}": _checkpoint_array(value, to_numpy)
        for name, value in bundle.surface.arrays().items()
    })
    for name, value in sorted(bundle.physics_state.arrays.items()):
        arrays[f"physics__{name}"] = _checkpoint_array(value, to_numpy)
    if trajectory is not None:
        for name, value in trajectory.arrays().items():
            arrays[f"trajectory__{name}"] = _checkpoint_array(value, to_numpy)
    if card_gather is not None:
        # A multi-card run holds each card's own latitude rows; the
        # checkpoint is the globe, and its per-array hashes are what the
        # bit gate is defined on, so the rows come together here, before
        # anything is hashed.  Single card: the caller passes None and not
        # a byte moves.
        arrays = card_gather(arrays)
    return arrays


def _trajectory_names(arrays) -> set[str]:
    return {name for name in arrays if name.startswith("trajectory__")}


def schema_for_arrays(arrays) -> str:
    """``v4`` when the arrays carry a second time level, ``v3`` otherwise.

    The schema is derived from the arrays rather than passed in, so an
    archive can never claim a shape it does not have: v4 means the seven
    trajectory arrays are there and v3 means none of them is, and
    :func:`read_checkpoint` refuses either half of that being false.
    """
    return (
        SEMILAG_CHECKPOINT_SCHEMA if _trajectory_names(arrays)
        else CHECKPOINT_SCHEMA
    )


def write_checkpoint(
    path: str | Path,
    bundle: ArwenGlobalState,
    *,
    config_hash: str,
    to_numpy,
    trackers: dict[str, object] | None = None,
    semi_implicit_scheme: str = DEFAULT_SEMI_IMPLICIT_SCHEME,
    integrator: str = DEFAULT_INTEGRATOR,
    trajectory=None,
    card_gather=None,
) -> Path:
    """Write a hash-bound checkpoint.  ``semi_implicit_scheme`` is the
    [semi_implicit] scheme the run integrates with: it selects the
    arithmetic pin the checkpoint carries (pins.SEMI_IMPLICIT_PINS), so an
    external-scheme run writes the pin of its era and a vertical-mode run
    writes v2; ``integrator`` selects the split-era document (``ssprk3`` /
    ``rk4``) or the IMEX one.  The caller with a config passes
    ``cfg.semi_implicit_scheme`` and ``cfg.integrator``."""
    if not _sha(config_hash):
        raise ValueError("config_hash must be a SHA-256 hex digest")
    if bundle.step < 0 or not math.isfinite(bundle.time_s) or bundle.time_s < 0.0:
        raise ValueError("checkpoint step/time are invalid")
    arrays = bundle_arrays(
        bundle, to_numpy, trajectory, card_gather=card_gather)
    return write_checkpoint_arrays(
        path, arrays,
        step=int(bundle.step), time_s=float(bundle.time_s),
        physics_state_schema=bundle.physics_state.schema,
        physics_metadata=bundle.physics_state.metadata,
        config_hash=config_hash, trackers=trackers,
        semi_implicit_scheme=semi_implicit_scheme, integrator=integrator,
    )


def checkpoint_metadata(
    arrays: dict[str, np.ndarray],
    *,
    step: int,
    time_s: float,
    physics_state_schema: str,
    physics_metadata: dict,
    config_hash: str,
    trackers: dict[str, object] | None = None,
    semi_implicit_scheme: str = DEFAULT_SEMI_IMPLICIT_SCHEME,
    integrator: str = DEFAULT_INTEGRATOR,
) -> dict[str, object]:
    """The metadata a checkpoint of ``arrays`` carries, ``self_sha256``
    included: exactly the record :func:`write_checkpoint_arrays` publishes
    and :func:`read_checkpoint` verifies, computed without writing.  A
    resident state can therefore be given the identity its checkpoint
    would have (the in-process cycle door names a background by it
    without a disk round trip) and :func:`state_from_checkpoint` accepts
    the result beside the arrays."""
    if not _sha(config_hash):
        raise ValueError("config_hash must be a SHA-256 hex digest")
    bundle = SimpleNamespace(step=int(step), time_s=float(time_s))
    if bundle.step < 0 or not math.isfinite(bundle.time_s) or bundle.time_s < 0.0:
        raise ValueError("checkpoint step/time are invalid")
    for name, array in arrays.items():
        if array.dtype.hasobject:
            raise TypeError(f"checkpoint array {name} may not use object dtype")
        if not np.isfinite(array.real).all() or (
            np.iscomplexobj(array) and not np.isfinite(array.imag).all()
        ):
            raise ValueError(f"checkpoint array {name} contains non-finite values")
    metadata: dict[str, object] = {
        "schema": schema_for_arrays(arrays),
        "config_hash": config_hash,
        "pins_hash": pins_hash(semi_implicit_scheme, integrator),
        "time_s": float(bundle.time_s),
        "step": int(bundle.step),
        "run_trackers": normalize_trackers(trackers),
        "physics_state_schema": physics_state_schema,
        "physics_metadata": physics_metadata,
        "arrays": {
            name: {
                "shape": list(array.shape),
                "dtype": array.dtype.str,
                "sha256": _array_hash(array),
            }
            for name, array in arrays.items()
        },
    }
    # Enforce pure JSON scheduler/identity state before any bytes are published.
    _canonical(metadata)
    metadata["self_sha256"] = hashlib.sha256(_canonical(metadata)).hexdigest()
    return metadata


def write_checkpoint_arrays(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    *,
    step: int,
    time_s: float,
    physics_state_schema: str,
    physics_metadata: dict,
    config_hash: str,
    trackers: dict[str, object] | None = None,
    semi_implicit_scheme: str = DEFAULT_SEMI_IMPLICIT_SCHEME,
    integrator: str = DEFAULT_INTEGRATOR,
    metadata: dict[str, object] | None = None,
) -> Path:
    """The host half of :func:`write_checkpoint`: hash, serialise and
    publish arrays already brought to the host by :func:`bundle_arrays`.

    Split out so a runner can take the device-to-host copy on the model
    thread and hand this part (SHA-256 of every array, zlib compression,
    fsync and the atomic rename) to a worker thread while the next step
    integrates; the archive it writes carries the metadata and arrays
    :func:`write_checkpoint` writes for the same state (the zip
    container's own entry timestamp is the one byte that may differ).
    ``metadata`` is the record :func:`checkpoint_metadata` computed for
    these same arrays and arguments, when the caller already holds it;
    it is checked against the arrays' own identity before anything is
    written, so a stale record cannot be published over fresh arrays.
    """
    computed = checkpoint_metadata(
        arrays, step=step, time_s=time_s,
        physics_state_schema=physics_state_schema,
        physics_metadata=physics_metadata, config_hash=config_hash,
        trackers=trackers, semi_implicit_scheme=semi_implicit_scheme,
        integrator=integrator,
    )
    if metadata is None:
        metadata = computed
    elif metadata.get("self_sha256") != computed["self_sha256"]:
        raise ValueError(
            "checkpoint metadata does not describe these arrays: the "
            "record's self hash differs from the arrays' own"
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.partial-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            __metadata__=np.asarray(
                json.dumps(metadata, sort_keys=True, allow_nan=False)
            ),
            **arrays,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return target


def read_checkpoint_header(path: str | Path) -> dict:
    """One checkpoint's metadata record, without its arrays.

    :func:`read_checkpoint` materializes every array, which is the right
    thing for a reader that is about to integrate from the state and the
    wrong thing for one that only needs the step, the model time or the
    config hash: a T533 checkpoint is gigabytes and a plan query mode that
    read one would be paying a forecast's memory to answer an arithmetic
    question.  ``np.load`` on an ``.npz`` reads members on demand, so only
    the metadata member is decompressed here.

    The record is returned as it was written and is NOT verified against the
    arrays it describes (that is :func:`read_checkpoint`'s contract and needs
    the arrays); the inventory is checked, so a file that is not one of this
    model's checkpoints raises instead of returning a dict with holes in it.
    """

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if "__metadata__" not in archive:
            raise ValueError(f"checkpoint {source} has no metadata")
        metadata = json.loads(str(archive["__metadata__"].item()))
    if not isinstance(metadata, dict) or "step" not in metadata:
        raise ValueError(f"checkpoint {source} carries no step")
    return metadata


def read_checkpoint(
    path: str | Path,
    *,
    expected_config_hash: str | None = None,
    semi_implicit_scheme: str | None = None,
    integrator: str = DEFAULT_INTEGRATOR,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Read and verify a checkpoint.

    The arithmetic pin is checked against the scheme the reader integrates
    with when ``semi_implicit_scheme`` is given (the restart door and every
    door that holds a config): a checkpoint written under the other scheme
    is refused by name, because its state was advanced by a different
    gravity-wave arithmetic and the resumed run would not be the
    continuation of the archived one.  Without a scheme (inspection,
    comparison of two checkpoints of one run) the pin must be one this
    build integrates, which admits both eras: the external proxy's pin is
    exactly the pin every archive carried before the vertical-mode scheme
    existed, and that arithmetic is shipped bit-identical.
    """
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if "__metadata__" not in archive:
            raise ValueError(f"checkpoint {source} has no metadata")
        metadata = json.loads(str(archive["__metadata__"].item()))
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
            if name != "__metadata__"
        }
    required = {
        "schema", "config_hash", "pins_hash", "time_s", "step",
        "run_trackers", "physics_state_schema", "physics_metadata",
        "arrays", "self_sha256",
    }
    if isinstance(metadata, dict) and metadata.get("schema") == LEVEL4_CHECKPOINT_SCHEMA:
        # Sniffed before the inventory check: a genuine Level-4 checkpoint has a
        # smaller metadata inventory, and the corruption message would otherwise
        # bury the door the operator is meant to use.
        raise ValueError(
            "checkpoint schema mismatch; Level-4 checkpoints require the "
            "explicit migrate-level4-checkpoint command"
        )
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise ValueError("checkpoint metadata inventory mismatch")
    schema = metadata["schema"]
    if schema not in (
        CHECKPOINT_SCHEMA, SEMILAG_CHECKPOINT_SCHEMA,
        SPECTRAL_TRACER_CHECKPOINT_SCHEMA,
    ):
        raise ValueError(
            "checkpoint schema mismatch; Level-4 checkpoints require the "
            "explicit migrate-level4-checkpoint command"
        )
    spectral_tracer_era = schema == SPECTRAL_TRACER_CHECKPOINT_SCHEMA
    # The second time level, both halves of the contract.  A v4 archive
    # missing one of the seven is refused by name, and so is a v3 archive
    # carrying any of them: a checkpoint that claimed a shape it did not
    # have would resume a two-time-level integrator on a level it never
    # wrote, which reads as a start-up step in the middle of a forecast
    # and shows up nowhere except as a phase error.
    trajectory_names = _trajectory_names(arrays)
    expected_trajectory = {f"trajectory__{name}" for name in TRAJECTORY_FIELDS}
    if schema == SEMILAG_CHECKPOINT_SCHEMA:
        missing = sorted(expected_trajectory - trajectory_names)
        extra = sorted(trajectory_names - expected_trajectory)
        if missing or extra:
            raise ValueError(
                "checkpoint trajectory inventory mismatch: schema v4 carries "
                f"exactly {sorted(expected_trajectory)}"
                + (f"; missing {missing}" if missing else "")
                + (f"; unknown {extra}" if extra else "")
            )
    elif trajectory_names:
        raise ValueError(
            f"checkpoint schema {schema} carries trajectory arrays "
            f"{sorted(trajectory_names)}: only schema "
            f"{SEMILAG_CHECKPOINT_SCHEMA} holds a second time level"
        )
    written_scheme = scheme_of_pins_hash(metadata["pins_hash"])
    if semi_implicit_scheme is not None:
        # the v3 pin, or the v2 pin WOOF 1.0.0 wrote for the same arithmetic
        if metadata["pins_hash"] not in accepted_pins_hashes(semi_implicit_scheme, integrator):
            raise ValueError(
                "checkpoint arithmetic pins mismatch: the checkpoint was "
                + (
                    f"written under the {written_scheme!r} arithmetic"
                    if written_scheme is not None
                    else (
                        "written in the spectral-tracer era (schema v2, "
                        "condensate and number moments as spectral "
                        "coefficients; that arithmetic moved the condensate "
                        "out of its columns and is not shipped)"
                        if metadata["pins_hash"] in SPECTRAL_TRACER_ERA_PINS_HASHES
                        else (
                            "written under a retired semi-Lagrangian pin "
                            "(the arithmetic before 2026-09-06's whole-theta "
                            "gather; inspectable, not resumable)"
                            if metadata["pins_hash"] in RETIRED_SEMILAG_PINS_HASHES
                            else "written under a pin no shipped scheme carries"
                        )
                    )
                )
                + f" and this run integrates {arithmetic_label(semi_implicit_scheme, integrator)!r}"
            )
    elif metadata["pins_hash"] not in ACCEPTED_PINS_HASHES | INSPECTABLE_RETIRED_PINS_HASHES:
        raise ValueError(
            "checkpoint arithmetic pins mismatch: the checkpoint's pin is not "
            "the pin of any semi-implicit scheme this build integrates, nor "
            "a spectral-tracer-era or retired semi-Lagrangian pin this build "
            "can inspect"
        )
    if metadata["physics_state_schema"] != PHYSICS_STATE_SCHEMA:
        raise ValueError("checkpoint physics-state schema mismatch")
    if not isinstance(metadata["physics_metadata"], dict):
        raise ValueError("checkpoint physics metadata must be an object")
    if not _sha(metadata["config_hash"]):
        raise ValueError("checkpoint config hash is malformed")
    if expected_config_hash is not None and metadata["config_hash"] != expected_config_hash:
        raise ValueError("checkpoint config identity does not match this run")
    if (
        isinstance(metadata["step"], bool)
        or not isinstance(metadata["step"], int)
        or metadata["step"] < 0
    ):
        raise ValueError("checkpoint step must be a nonnegative integer")
    time_s = float(metadata["time_s"])
    if not math.isfinite(time_s) or time_s < 0.0:
        raise ValueError("checkpoint time_s must be finite and nonnegative")
    metadata["run_trackers"] = normalize_trackers(metadata["run_trackers"])
    self_hash = metadata.pop("self_sha256")
    if not _sha(self_hash) or self_hash != hashlib.sha256(_canonical(metadata)).hexdigest():
        raise ValueError("checkpoint metadata self-hash mismatch")
    metadata["self_sha256"] = self_hash
    expected = metadata["arrays"]
    if not isinstance(expected, dict) or set(expected) != set(arrays):
        raise ValueError("checkpoint array inventory mismatch")
    spectral_names = {f"atmosphere__{name}" for name in SPECTRAL_FIELDS}
    grid_names = {f"atmosphere__{name}" for name in GRID_TRACERS}
    if spectral_tracer_era:
        # v2 carried every prognostic field as spectral coefficients.
        spectral_names |= grid_names
        grid_names = set()
    surface_names = {f"surface__{stored}" for stored in _SURFACE_MAP.values()}
    # A checkpoint written before the cold-start seeding carries no sea-ice
    # planes; the state it holds ran ice-free, so reading it with zero ice
    # reproduces that run rather than inventing an analysis field.  The
    # absence is recorded on the metadata for the receipt.
    absent_seeded = sorted(
        f"surface__{_SURFACE_MAP[member]}" for member in SEEDED_SURFACE_MEMBERS
        if f"surface__{_SURFACE_MAP[member]}" not in arrays
    )
    if not (spectral_names | grid_names | surface_names) - set(absent_seeded) <= set(arrays):
        raise ValueError("checkpoint atmosphere/surface array set is incomplete")
    metadata["absent_seeded_surface_arrays"] = absent_seeded
    if any(
        not (
            name in spectral_names or name in grid_names
            or name in surface_names or name.startswith("physics__")
            or name in trajectory_names
        )
        for name in arrays
    ):
        raise ValueError("checkpoint has an unknown array namespace")
    for name, array in arrays.items():
        row = expected[name]
        if not isinstance(row, dict) or set(row) != {"shape", "dtype", "sha256"}:
            raise ValueError(f"checkpoint array metadata for {name} is malformed")
        if list(array.shape) != row["shape"] or array.dtype.str != row["dtype"]:
            raise ValueError(f"checkpoint array {name} shape/dtype mismatch")
        if _array_hash(array) != row["sha256"]:
            raise ValueError(f"checkpoint array {name} hash mismatch")
        if name in spectral_names and array.dtype.kind != "c":
            raise ValueError(f"spectral checkpoint array {name} must be complex")
        if name in grid_names and array.dtype.kind != "f":
            raise ValueError(
                f"grid tracer checkpoint array {name} must be a real float "
                "array on the Gaussian grid"
            )
        if array.dtype.hasobject:
            raise TypeError(f"checkpoint array {name} may not use object dtype")
        if not np.isfinite(array.real).all() or (
            np.iscomplexobj(array) and not np.isfinite(array.imag).all()
        ):
            raise ValueError(f"checkpoint array {name} contains non-finite values")
    return metadata, arrays


def state_from_checkpoint(
    metadata: dict, arrays: dict[str, np.ndarray], backend, *, transform=None,
):
    """Rebuild the bundle a checkpoint holds.

    A schema-v3 checkpoint needs no transform.  A spectral-tracer-era
    checkpoint (schema v2) holds the ten condensate and number-moment
    fields as spectral coefficients; given ``transform`` they are
    synthesized onto the Gaussian grid and floored at zero (the clip is
    the representation's own ringing, measured and left to the
    caller's receipt), and without one the caller is refused by name
    rather than handed a state whose tracers are complex coefficients
    nothing can transport.
    """
    spectral_tracer_era = (
        metadata.get("schema") == SPECTRAL_TRACER_CHECKPOINT_SCHEMA
    )
    tracers = {}
    for name in GRID_TRACERS:
        value = arrays[f"atmosphere__{name}"]
        if spectral_tracer_era:
            if transform is None:
                raise ValueError(
                    "a spectral-tracer-era checkpoint (schema v2) carries "
                    f"{name} as spectral coefficients; pass the run's "
                    "transform to synthesize it onto the grid"
                )
            grid = transform.inverse(
                backend.asarray(value, dtype=backend.complex_dtype)
            )
            tracers[name] = backend.xp.maximum(grid, 0.0)
        else:
            tracers[name] = backend.asarray(value, dtype=backend.float_dtype)
    atmosphere = MoistHybridState(
        **{
            name: backend.asarray(
                arrays[f"atmosphere__{name}"], dtype=backend.complex_dtype
            )
            for name in SPECTRAL_FIELDS
        },
        **tracers,
        time_s=float(metadata["time_s"]),
        step=int(metadata["step"]),
    )
    surface = SurfaceState(**{
        member: backend.asarray(
            arrays[f"surface__{stored}"], dtype=backend.float_dtype
        )
        for member, stored in _SURFACE_MAP.items()
        # Pre-seeding checkpoints: the members default to zero planes
        # (state.SurfaceState.__post_init__); read_checkpoint recorded
        # which ones were absent.
        if f"surface__{stored}" in arrays
    })
    physics = PhysicsState(
        schema=metadata["physics_state_schema"],
        arrays={
            name.removeprefix("physics__"): backend.asarray(
                value,
                dtype=(
                    backend.complex_dtype
                    if np.iscomplexobj(value)
                    else backend.float_dtype
                ),
            )
            for name, value in arrays.items()
            if name.startswith("physics__")
        },
        metadata=dict(metadata["physics_metadata"]),
    )
    physics.validate()
    return ArwenGlobalState(atmosphere, surface, physics)


def trajectory_from_checkpoint(metadata: dict, arrays: dict, backend):
    """The second time level a v4 checkpoint holds, or None.

    A run whose integrator carries no second level gets None from every
    archive, and a semi-Lagrangian run gets None from the step-0 archive
    of its own lineage, which is exactly the state the uninterrupted run
    is in at step 0.
    """
    if metadata.get("schema") != SEMILAG_CHECKPOINT_SCHEMA:
        return None
    return TrajectoryState.from_arrays({
        name: backend.asarray(
            arrays[f"trajectory__{name}"], dtype=backend.float_dtype
        )
        for name in TRAJECTORY_FIELDS
    })


__all__ = [
    "TRACKER_KEYS",
    "bundle_arrays",
    "checkpoint_metadata",
    "normalize_trackers",
    "read_checkpoint",
    "read_checkpoint_header",
    "schema_for_arrays",
    "state_from_checkpoint",
    "trajectory_from_checkpoint",
    "write_checkpoint",
]
