"""Target-device qualification battery for Level-5 native Arwen physics.

Source tests prove the transaction and adapter contracts; this module produces
the separate evidence required before a device-pending adapter can be promoted.
A failed qualification is itself durable evidence and never becomes a candidate.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from woof.globe.physics.registry import get_global_physics_adapter

from .checkpoint import read_checkpoint
from .config import ArwenGlobalConfig
from .constants import (
    NATIVE_CONTRACT_CANDIDATE_SCHEMA,
    NATIVE_DEVICE_EVIDENCE_SCHEMA,
)
from .physics.builtin_adapters import ensure_builtin_global_physics_adapters
from .pins import ACCEPTED_PINS_HASHES, pins_hash
from .runner import CHECKPOINT_PREFIX, RECEIPT_NAME, run


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(_json_safe(k)): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    row = dict(payload)
    row["self_sha256"] = hashlib.sha256(_canonical(row)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(
        json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _read_json(path: str | Path, schema: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != schema or payload.get("pins_hash") not in ACCEPTED_PINS_HASHES:
        raise ValueError("native qualification receipt schema/pins mismatch")
    self_hash = payload.pop("self_sha256", None)
    if self_hash != hashlib.sha256(_canonical(payload)).hexdigest():
        raise ValueError("native qualification receipt self-hash mismatch")
    payload["self_sha256"] = self_hash
    return payload


def read_native_device_evidence(path: str | Path) -> dict[str, object]:
    return _read_json(path, NATIVE_DEVICE_EVIDENCE_SCHEMA)


def read_native_contract_candidate(path: str | Path) -> dict[str, object]:
    payload = _read_json(path, NATIVE_CONTRACT_CANDIDATE_SCHEMA)
    if payload.get("status") != "experimental-candidate":
        raise ValueError("native contract candidate is not experimental-candidate")
    return payload


def _terminal_checkpoint(outdir: Path) -> Path:
    candidates = sorted(outdir.glob(f"{CHECKPOINT_PREFIX}*.npz"))
    if not candidates:
        raise ValueError(f"qualification run {outdir} emitted no checkpoints")
    return candidates[-1]


def _midpoint_checkpoint(outdir: Path, total_steps: int) -> Path:
    candidates = sorted(outdir.glob(f"{CHECKPOINT_PREFIX}*.npz"))
    eligible = []
    for path in candidates:
        metadata, _ = read_checkpoint(path)
        step = int(metadata["step"])
        if 0 < step < total_steps:
            eligible.append((abs(step - total_steps / 2.0), path))
    if not eligible:
        raise ValueError(
            "native qualification requires an intermediate checkpoint; set "
            "output_interval_s no greater than half duration_s"
        )
    return min(eligible, key=lambda item: item[0])[1]


def _compare_checkpoints(
    first: Path, second: Path, *, expected_config_hash: str
) -> dict[str, object]:
    meta_a, arrays_a = read_checkpoint(first)
    meta_b, arrays_b = read_checkpoint(second)
    for path, metadata in ((first, meta_a), (second, meta_b)):
        if metadata["config_hash"] != expected_config_hash:
            raise ValueError(
                f"qualification checkpoint {path} was written for config "
                f"{metadata['config_hash']}, not the qualified config "
                f"{expected_config_hash}; the comparison would compare two "
                "different experiments"
            )
    if set(arrays_a) != set(arrays_b):
        raise ValueError("qualification restart changed checkpoint inventory")
    rows = {}
    bit_exact = True
    for name in sorted(arrays_a):
        a = arrays_a[name]
        b = arrays_b[name]
        exact = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
        bit_exact &= exact
        if np.iscomplexobj(a):
            maximum = float(np.max(np.abs(a - b)))
        else:
            maximum = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        rows[name] = {"bit_exact": bool(exact), "maximum_absolute_difference": maximum}
    trackers_exact = meta_a["run_trackers"] == meta_b["run_trackers"]
    bit_exact &= trackers_exact
    return {
        "bit_exact": bool(bit_exact),
        "trackers_exact": bool(trackers_exact),
        "arrays": rows,
        "first_checkpoint_self_sha256": meta_a["self_sha256"],
        "second_checkpoint_self_sha256": meta_b["self_sha256"],
    }


def _source_hashes() -> dict[str, str]:
    # Enumerated, not listed: a hand-written list silently drops a new or
    # renamed adapter file out of the evidence, and the evidence then fails to
    # identify the code that produced it.
    root = Path(__file__).resolve().parent
    files = [root / "native_qualification.py"]
    files.extend(sorted((root / "physics").glob("*.py")))
    # POSIX keys so the evidence record is comparable across platforms.
    return {
        path.relative_to(root.parent.parent).as_posix(): _file_hash(path)
        for path in files
    }


EVIDENCE_NAME = "native-device-evidence.json"
CANDIDATE_NAME = "native-contract-candidate.json"
CONTINUOUS_DIR = "continuous"
RESUMED_DIR = "resumed"


def _owned_paths(output: Path) -> list[Path]:
    return [
        output / EVIDENCE_NAME,
        output / CANDIDATE_NAME,
        output / CONTINUOUS_DIR,
        output / RESUMED_DIR,
    ]


def _prepare_output(output: Path, overwrite: bool) -> None:
    existing = [path for path in _owned_paths(output) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"native qualification output exists in {output} "
            f"({', '.join(path.name for path in existing)}); pass overwrite=True "
            "to replace the artifacts this door owns"
        )
    if overwrite:
        # Only the four owned artifacts.  A recursive wipe of the whole outdir
        # destroys operator notes, prior evidence and plots that this door
        # never wrote and cannot restore.
        import shutil

        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)


def qualify_native_adapter(
    cfg: ArwenGlobalConfig,
    outdir: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path | None]:
    """Run uninterrupted/restarted CUDA campaigns and emit device evidence."""
    output = Path(outdir)
    evidence_path = output / EVIDENCE_NAME
    candidate_path = output / CANDIDATE_NAME
    _prepare_output(output, overwrite)

    ensure_builtin_global_physics_adapters()
    if cfg.physics_mode != "arwen-native":
        raise ValueError("native qualification requires physics.mode='arwen-native'")
    if cfg.backend != "cupy" or cfg.precision != "float32":
        raise ValueError("native qualification requires cupy/float32")
    registration = get_global_physics_adapter(cfg.native_adapter_name or "")
    if registration.contract["admission_status"] != "device-pending":
        raise ValueError("this qualification door is for device-pending adapters")

    started = time.time()
    base: dict[str, object] = {
        "schema": NATIVE_DEVICE_EVIDENCE_SCHEMA,
        "pins_hash": pins_hash(cfg.semi_implicit_scheme, cfg.integrator),
        "config_hash": cfg.config_hash,
        "adapter_name": registration.name,
        "adapter_contract_hash": registration.contract_hash,
        "adapter_arithmetic_sha256": registration.contract["arithmetic_sha256"],
        "source_hashes": _source_hashes(),
        "host": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
        },
        "started_unix_s": started,
    }
    from woof.local_gpu import NO_LOCAL_GPU_ENV, no_local_gpu

    if no_local_gpu():
        raise RuntimeError(
            f"native qualification refused: {NO_LOCAL_GPU_ENV} is set, so this "
            "process may not read the local CUDA device for evidence; run the "
            "battery on a node where the variable is unset."
        )
    try:
        import cupy as cp  # type: ignore

        device = int(cp.cuda.runtime.getDevice())
        properties = _json_safe(cp.cuda.runtime.getDeviceProperties(device))
        driver_version = int(cp.cuda.runtime.driverGetVersion())
        runtime_version = int(cp.cuda.runtime.runtimeGetVersion())
        base["device"] = {
            "ordinal": device,
            "properties": properties,
            "driver_version": driver_version,
            "runtime_version": runtime_version,
            "cupy": cp.__version__,
        }
        cp.cuda.runtime.deviceSynchronize()
        continuous_dir = output / CONTINUOUS_DIR
        resumed_dir = output / RESUMED_DIR
        continuous_result = run(cfg, continuous_dir, overwrite=True)
        total_steps = int(round(cfg.duration_s / cfg.dt_s))
        midpoint = _midpoint_checkpoint(continuous_dir, total_steps)
        resumed_result = run(cfg, resumed_dir, restart=midpoint, overwrite=True)
        # runner.run() does not raise on a failed gate; it records status='fail'
        # and returns.  Without this the battery emits pass evidence and a
        # promotion candidate for two campaigns that failed their own receipts.
        campaign_gates = {}
        for label, result in (
            ("continuous", continuous_result), ("resumed", resumed_result),
        ):
            campaign_gates[label] = {
                "status": result["status"],
                "gates": result["gates"],
                "run_trackers": result["run_trackers"],
            }
        failed = sorted(
            label for label, row in campaign_gates.items()
            if row["status"] != "pass"
        )
        if failed:
            names = {
                label: sorted(
                    gate for gate, row in campaign_gates[label]["gates"].items()
                    if not row["passed"]
                )
                for label in failed
            }
            raise FloatingPointError(
                "native qualification campaigns did not pass their receipt "
                f"gates: {json.dumps(names, sort_keys=True)}"
            )
        terminal_a = _terminal_checkpoint(continuous_dir)
        terminal_b = _terminal_checkpoint(resumed_dir)
        comparison = _compare_checkpoints(
            terminal_a, terminal_b, expected_config_hash=cfg.config_hash
        )
        if not comparison["bit_exact"]:
            raise FloatingPointError(
                "native qualification restart continuation is not bit-exact"
            )
        continuous_receipt = continuous_dir / RECEIPT_NAME
        resumed_receipt = resumed_dir / RECEIPT_NAME
        base.update({
            "status": "pass",
            "completed_unix_s": time.time(),
            "wall_seconds": time.time() - started,
            "campaign": {
                "duration_s": cfg.duration_s,
                "dt_s": cfg.dt_s,
                "total_steps": total_steps,
                "midpoint_checkpoint": str(midpoint),
                "continuous_terminal": str(terminal_a),
                "resumed_terminal": str(terminal_b),
                "continuous_receipt_sha256": _file_hash(continuous_receipt),
                "resumed_receipt_sha256": _file_hash(resumed_receipt),
                "receipt_gates": campaign_gates,
            },
            "restart_comparison": comparison,
            "admission": (
                "device evidence passed; adapter remains experimental until "
                "meteorological and scheme-parity campaigns are reviewed"
            ),
        })
        _write_json(evidence_path, base)
        evidence = read_native_device_evidence(evidence_path)
        candidate = {
            "schema": NATIVE_CONTRACT_CANDIDATE_SCHEMA,
            "pins_hash": pins_hash(cfg.semi_implicit_scheme, cfg.integrator),
            "status": "experimental-candidate",
            "adapter_name": registration.name,
            "adapter_contract_hash": registration.contract_hash,
            "device_evidence_path": str(evidence_path),
            "device_evidence_file_sha256": _file_hash(evidence_path),
            "device_evidence_self_sha256": evidence["self_sha256"],
            "proposed_admission_status": "experimental",
            "not_validated": [
                "multi-case meteorological skill",
                "per-scheme WRF column parity on this global adapter",
                "long coupled global stability",
                "global-to-regional forecast A/B",
            ],
        }
        _write_json(candidate_path, candidate)
        read_native_contract_candidate(candidate_path)
        return evidence_path, candidate_path
    except Exception as exc:
        base.update({
            "status": "error",
            "completed_unix_s": time.time(),
            "wall_seconds": time.time() - started,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "candidate_emitted": False,
        })
        _write_json(evidence_path, base)
        read_native_device_evidence(evidence_path)
        raise RuntimeError(
            f"native qualification failed; durable evidence: {evidence_path}: {exc}"
        ) from exc


__all__ = [
    "qualify_native_adapter",
    "read_native_contract_candidate",
    "read_native_device_evidence",
]
