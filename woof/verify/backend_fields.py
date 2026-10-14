"""Registered, fail-closed gates on saved backend fields.

Scores use complete stored arrays in float64. NetCDF values are decoded by
spectral_io and the Rust bridge; this module never starts a model backend.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from woof.verify.spectral_io import load_array, sha256_file

SCHEMA = "woof-backend-fields-v1"
CASES = ("gabls1", "cbl_dry", "igw", "hill2d")
FILES = {"gabls1": "gabls1_head.npz", "cbl_dry": "cbl_dry_1.npz",
         "igw": "wrfout_igw.nc", "hill2d": "wrfout_hill2d.nc"}
GABLS_FIELDS = ("u", "v", "theta", "e", "uw", "vw", "wt", "kv",
                "ust", "hfx_kin", "tstar", "theta_s")
SURFACE_FIELDS = ("ust", "hfx_kin", "tstar", "theta_s")


def _reject_nonfinite_json(value: str):
    raise ValueError(f"non-finite JSON constant {value}")


def canonical_hash(value: object) -> str:
    """SHA-256 of UTF-8 sorted compact JSON, with no non-finite numbers."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def seal_registration(spec: dict) -> dict:
    """Seal an independently declared policy; never estimate limits."""
    sealed = json.loads(json.dumps(spec, allow_nan=False))
    sealed.pop("registration_sha256", None)
    sealed["limits_sha256"] = canonical_hash(sealed["limits"])
    sealed["registration_sha256"] = canonical_hash(sealed)
    return sealed


def _relative(root: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not name:
        raise ValueError("artifact names must be relative, without '..'")
    return root / path


def _array(root: Path, name: str, variable: str | None = None,
           index: int = 0) -> np.ndarray:
    return load_array(_relative(root, name), variable=variable,
                      time_index=index)[0]


def _equal(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.shape != right.shape or not np.array_equal(left, right):
        raise ValueError(f"{label}: initial state or geometry differs")


def _configuration(reference: Path, reg: dict) -> dict | None:
    name = reg.get("reference", {}).get("config_file")
    if name is None:
        return None
    cfg = json.loads(_relative(reference, name).read_text(),
                     parse_constant=_reject_nonfinite_json)
    if not isinstance(cfg, dict):
        raise ValueError("physical configuration must be a JSON object")
    return cfg


def _size(cfg: dict, key: str) -> int:
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"configuration {key} must be a positive integer")
    return value


def _check_equal(left: np.ndarray, right: np.ndarray, label: str,
                 blockers: list[str]) -> None:
    # Retain diagnostic scores when admission fails, without a pass claim.
    try:
        _equal(left, right, label)
    except ValueError as exc:
        blockers.append(str(exc))


def _rms(value: np.ndarray) -> float:
    # Scaling avoids overflow of the square while retaining FP64 reductions.
    if not value.size:
        raise ValueError("RMS requires a nonempty array")
    largest = float(np.max(np.abs(value)))
    if largest == 0:
        return 0.0
    return largest * float(np.sqrt(np.mean(np.square(value / largest),
                                         dtype=np.float64)))


def score(candidate: np.ndarray, reference: np.ndarray) -> dict:
    """RMS(delta)/RMS(reference signal), never pooled across leads/units."""
    if candidate.shape != reference.shape or not reference.size:
        raise ValueError(f"field shapes differ or are empty: "
                         f"{candidate.shape} versus {reference.shape}")
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(reference)):
        raise ValueError("candidate/reference field contains non-finite cells")
    with np.errstate(over="raise", invalid="raise"):
        delta = candidate.astype(np.float64) - reference.astype(np.float64)
    if not np.all(np.isfinite(delta)):
        raise ValueError("field difference is non-finite")
    rms_delta, rms_reference = _rms(delta), _rms(reference)
    nrms = rms_delta / rms_reference if rms_reference else (
        0.0 if rms_delta == 0 else None)
    if nrms is not None and not math.isfinite(nrms):
        nrms = None
    return {"shape": list(reference.shape), "cells": int(reference.size),
            "rms_delta": rms_delta, "rms_reference_signal": rms_reference,
            "normalized_rms": nrms, "max_abs_delta": float(np.max(np.abs(delta)))}


def _read_registration(path: Path | None, case: str, reference: Path,
                       candidate_config: Path | None,
                       candidate_initializer: Path | None) -> tuple[dict, list[str]]:
    problems: list[str] = []
    if path is None:
        return {}, ["missing registration; scores have no acceptance policy"]
    try:
        reg = json.loads(path.read_text(), parse_constant=_reject_nonfinite_json)
        if not isinstance(reg, dict):
            raise ValueError("registration must be a JSON object")
    except (OSError, ValueError) as exc:
        return {}, [f"cannot read registration: {exc}"]
    try:
        payload = dict(reg)
        digest = payload.pop("registration_sha256", None)
        if digest != canonical_hash(payload):
            problems.append("registration canonical SHA-256 mismatch")
        if reg.get("schema") != SCHEMA or reg.get("case") != case:
            problems.append("registration schema or case mismatch")
        limits = reg.get("limits", {})
        if not isinstance(limits, dict) or not limits:
            problems.append("missing independently declared numeric limits")
        if reg.get("limits_sha256") != canonical_hash(limits):
            problems.append("limits SHA-256 mismatch")
        if not isinstance(limits, dict):
            limits = {}
        if any(not isinstance(reg.get(key), str) or not reg[key].strip()
               for key in ("declaration", "declared_by")):
            problems.append("missing independent policy declaration/provenance")
        mode = reg.get("mode")
        if mode not in ("registered_nrms", "exact_stored_output"):
            problems.append("unrecognized registration mode")
        for name, bound in limits.items():
            if not isinstance(bound, dict):
                problems.append(f"invalid limits for {name}")
                continue
            for key, value in bound.items():
                if key not in ("nrms_max", "zero_reference_rms_delta_max"):
                    problems.append(f"unknown limit {name}:{key}")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    problems.append(f"invalid numeric limit {name}:{key}")
                elif mode == "exact_stored_output" and value != 0:
                    problems.append("exact_stored_output requires all limits equal zero")
            if "nrms_max" not in bound:
                problems.append(f"missing nrms_max for {name}")
        ref = reg.get("reference", {})
        if not isinstance(ref, dict):
            raise ValueError("reference must be an object")
        if any(not isinstance(ref.get(key), str) or not ref[key].strip()
               for key in ("provenance", "producer")):
            problems.append("missing reference producer/provenance")
        if case == "gabls1" and ref.get("producer") != "numpy_cpu":
            problems.append("GABLS reference must declare numpy_cpu producer")
        if case != "gabls1" and ref.get("producer") not in ("numpy_cpu", "cpu", "woof_cuda"):
            problems.append("reference producer must be independently identified as cpu, numpy_cpu or woof_cuda")
        artifacts = ref.get("artifacts", {})
        if not isinstance(artifacts, dict) or FILES[case] not in artifacts:
            problems.append("missing required reference artifact hash")
        if isinstance(artifacts, dict):
            for name, expected in artifacts.items():
                try:
                    if sha256_file(_relative(reference, name)) != expected:
                        problems.append(f"reference artifact SHA-256 mismatch: {name}")
                except (OSError, ValueError) as exc:
                    problems.append(f"reference artifact unavailable: {name}: {exc}")
        config_name = ref.get("config_file")
        config_digest = ref.get("configuration_sha256")
        try:
            config = _configuration(reference, reg)
            if config is None:
                raise ValueError("missing physical configuration")
            actual = canonical_hash(config)
            if actual != config_digest or not config_digest:
                problems.append("reference configuration digest mismatch")
            if config_name not in artifacts:
                problems.append("reference configuration artifact is not pinned")
        except (OSError, ValueError, TypeError) as exc:
            problems.append(f"missing reference physical configuration: {exc}")
        try:
            if candidate_config is None:
                raise ValueError("supply --candidate-config")
            candidate_configuration = json.loads(candidate_config.read_text(),
                                                parse_constant=_reject_nonfinite_json)
            if not isinstance(candidate_configuration, dict):
                raise ValueError("candidate physical configuration must be a JSON object")
            if canonical_hash(candidate_configuration) != config_digest:
                problems.append("candidate physical configuration digest mismatch")
        except (OSError, ValueError) as exc:
            problems.append(f"candidate configuration admission blocked: {exc}")
        init_name = ref.get("initializer_file")
        try:
            if not ref.get("initializer_sha256") or sha256_file(_relative(reference, init_name)) != ref["initializer_sha256"]:
                problems.append("reference initializer digest mismatch")
            if init_name not in artifacts:
                problems.append("reference initializer artifact is not pinned")
        except (OSError, ValueError, TypeError) as exc:
            problems.append(f"missing reference initializer provenance: {exc}")
        try:
            if candidate_initializer is None:
                raise ValueError("supply --candidate-initializer")
            if not reg.get("candidate_initializer_sha256") or sha256_file(candidate_initializer) != reg["candidate_initializer_sha256"]:
                problems.append("candidate initializer digest mismatch")
        except (OSError, ValueError) as exc:
            problems.append(f"candidate initializer admission blocked: {exc}")
    except (ValueError, TypeError, KeyError) as exc:
        problems.append(f"invalid registration: {exc}")
    if not isinstance(reg.get("limits"), dict):
        reg["limits"] = {}
    if not isinstance(reg.get("reference"), dict):
        reg["reference"] = {}
    return reg, problems


def _base(reg: dict, reference: Path, case: str) -> np.ndarray:
    spec = reg.get("reference", {}).get("base", {})
    if not isinstance(spec, dict):
        raise ValueError("reference base specification must be an object")
    if not spec.get("file") or not spec.get("sha256"):
        raise ValueError("missing independently pinned reference initial-theta/base artifact")
    path = _relative(reference, spec["file"])
    if path.suffix != ".npy":
        raise ValueError("reference base must be a saved .npy array")
    if sha256_file(path) != spec["sha256"]:
        raise ValueError("reference initial-theta/base SHA-256 mismatch")
    # Preserve and check stored precision before load_array widens it.
    raw = np.load(path, allow_pickle=False)
    expected = np.dtype("float64" if case == "gabls1" else "float32")
    if raw.dtype != expected:
        raise ValueError(f"reference base must be stored {expected}")
    return load_array(path)[0]


def _unavailable(field: str, seconds: int, reason: str) -> dict:
    return {"field": field, "lead_seconds": seconds, "gated": True,
            "verdict": "blocked", "error": reason}


def _npz_pairs(case: str, candidate: Path, reference: Path, reg: dict,
               blockers: list[str]):
    name = FILES[case]
    cfg = _configuration(reference, reg)
    if case == "gabls1":
        for key in ("z", "z_face"):
            _check_equal(_array(candidate, name, key), _array(reference, name, key), key, blockers)
        try:
            base = _base(reg, reference, case)
            base_problem = None
        except (OSError, ValueError, TypeError) as exc:
            base, base_problem = None, str(exc)
            blockers.append(base_problem)
        z = _array(reference, name, "z")
        z_face = _array(reference, name, "z_face")
        if z.ndim != 1 or z_face.shape != (len(z) - 1,):
            raise ValueError("invalid GABLS saved profile geometry")
        if cfg is not None and z.shape != (_size(cfg, "NZ"),):
            raise ValueError("GABLS profile geometry differs from admitted NZ")
        for field in GABLS_FIELDS:
            c, r = _array(candidate, name, field), _array(reference, name, field)
            if field not in SURFACE_FIELDS:
                expected = z_face.shape if field in ("uw", "vw", "wt") else z.shape
                if c.shape != expected or r.shape != expected:
                    raise ValueError(f"{field}: profile shape differs from geometry")
            if field == "theta":
                yield field + "_raw_diagnostic", 32400, c, r, False
                if base is None or base.shape != r.shape:
                    reason = base_problem or "GABLS reference initial theta shape mismatch"
                    if reason not in blockers:
                        blockers.append(reason)
                    yield _unavailable(field, 32400, reason)
                    continue
                c, r = c - base, r - base
            if field in SURFACE_FIELDS:
                if c.shape != r.shape or c.ndim != 1 or len(c) != 3240:
                    raise ValueError(f"{field}: expected 3240 saved 10-second samples")
                if field == "theta_s":
                    initial = reg.get("reference", {}).get("surface_theta_initial")
                    if isinstance(initial, bool) or not isinstance(initial, (float, int)) or not math.isfinite(initial):
                        reason = "missing independently pinned initial surface theta"
                        blockers.append(reason)
                        yield _unavailable(field, 0, reason)
                        continue
                    _check_equal(np.asarray([initial]), r[:1], "initial surface theta", blockers)
                    c, r = c - initial, r - initial
                for step in range(len(r)):
                    yield field, step * 10, c[step:step+1], r[step:step+1], True
            else:
                yield field, 32400, c, r, True
    else:
        for key in ("z", "theta", "u", "v", "speed"):
            c, r = _array(candidate, name, "initial_" + key), _array(reference, name, "initial_" + key)
            if cfg is not None and (c.shape != (_size(cfg, "nz"),) or r.shape != (_size(cfg, "nz"),)):
                raise ValueError(f"initial_{key}: profile differs from admitted nz")
            _check_equal(c, r, "initial_" + key, blockers)
        with np.load(_relative(reference, name), allow_pickle=False) as archive:
            optional = "final_tke" in archive.files
        with np.load(_relative(candidate, name), allow_pickle=False) as archive:
            if ("final_tke" in archive.files) != optional:
                raise ValueError("candidate/reference TKE profile inventory differs")
        fields = ("u", "v", "theta", "speed", "z") + (("tke",) if optional else ())
        z = _array(reference, name, "final_z")
        if z.ndim != 1:
            raise ValueError("CBL geometry must be a saved vertical profile")
        if cfg is not None and z.shape != (_size(cfg, "nz"),):
            raise ValueError("CBL final profile differs from admitted nz")
        for field in fields:
            c, r = _array(candidate, name, "final_" + field), _array(reference, name, "final_" + field)
            if c.shape != z.shape or r.shape != z.shape:
                raise ValueError(f"{field}: profile shape differs from geometry")
            if field == "tke":
                _check_equal(_array(candidate, name, "initial_tke"), _array(reference, name, "initial_tke"), "initial_tke", blockers)
            if field == "theta":
                yield "theta_raw_diagnostic", 7200, c, r, False
                initial = _array(reference, name, "initial_theta")
                c, r = c - initial, r - initial
            yield field, 7200, c, r, True


def _netcdf_inventory(path: Path, cfg: dict | None) -> tuple[list[str], dict]:
    from woof.netcdf_bridge import open_dataset
    with open_dataset(path) as dataset:
        if "Times" not in dataset.variables:
            raise ValueError("saved NetCDF is missing Times")
        values = dataset.variables["Times"][:]
        times = [b"".join(np.asarray(row, dtype="S1").reshape(-1)).decode("ascii").rstrip("\x00 ") for row in values]
        geometry = {"dimensions": {name: len(size) for name, size in dataset.dimensions.items() if name not in ("Time", "DateStrLen")},
                    "DX": float(dataset.getncattr("DX")), "DY": float(dataset.getncattr("DY")),
                    "coordinates": {name: list(dataset.variables[name].dimensions)
                                    for name in ("ZNU", "ZNW", "P_TOP")
                                    if name in dataset.variables}}
        if not math.isfinite(geometry["DX"]) or not math.isfinite(geometry["DY"]) or geometry["DX"] <= 0 or geometry["DY"] <= 0:
            raise ValueError("DX/DY must be finite and positive")
        for field in ("U", "V", "W", "T", "PH", "MU", "PHB", "MUB", "HGT"):
            if field not in dataset.variables or not dataset.variables[field].shape or dataset.variables[field].shape[0] != len(times):
                raise ValueError(f"{field}: missing or incomplete time inventory")
        if cfg is not None:
            nx, ny, nz = (_size(cfg, key) for key in ("nx", "ny", "nz"))
            shapes = {"U": (nz, ny, nx+1), "V": (nz, ny+1, nx),
                      "W": (nz+1, ny, nx), "T": (nz, ny, nx),
                      "PH": (nz+1, ny, nx), "PHB": (nz+1, ny, nx),
                      "MU": (ny, nx), "MUB": (ny, nx), "HGT": (ny, nx)}
            for field, shape in shapes.items():
                if tuple(dataset.variables[field].shape) != (len(times), *shape):
                    raise ValueError(f"{field}: native staggered shape differs from admitted grid")
        return times, geometry


def _nc_pairs(case: str, candidate: Path, reference: Path, reg: dict,
              blockers: list[str]):
    name = FILES[case]
    cfg = _configuration(reference, reg)
    ct, cg = _netcdf_inventory(_relative(candidate, name), cfg)
    rt, rg = _netcdf_inventory(_relative(reference, name), cfg)
    if ct != rt or cg != rg:
        raise ValueError("candidate/reference Times or grid geometry differs")
    interval = 300 if case == "igw" else 3600
    instants = [datetime.strptime(item, "%Y-%m-%d_%H:%M:%S") for item in rt]
    if len(rt) != 11 or [(item - instants[0]).total_seconds() for item in instants] != list(range(0, 11 * interval, interval)):
        raise ValueError(f"expected all 11 frames at {interval}-second intervals")
    try:
        base = _base(reg, reference, case)
        base_problem = None
    except (OSError, ValueError, TypeError) as exc:
        base, base_problem = None, str(exc)
        blockers.append(base_problem)
    for index in range(11):
        for field in ("PHB", "MUB", "HGT"):
            _check_equal(_array(candidate, name, field, index), _array(reference, name, field, index), f"{field}@{index}", blockers)
        for field, dimensions in rg["coordinates"].items():
            coordinate_index = index if dimensions and dimensions[0].lower() in ("time", "times") else 0
            _check_equal(_array(candidate, name, field, coordinate_index),
                         _array(reference, name, field, coordinate_index),
                         f"{field}@{index}", blockers)
        if index == 0:
            for field in ("PH", "MU"):
                _check_equal(_array(candidate, name, field),
                             _array(reference, name, field),
                             f"initial {field}", blockers)
        for field in ("U", "V", "W", "T"):
            c, r = _array(candidate, name, field, index), _array(reference, name, field, index)
            if index == 0:
                _check_equal(c, r, f"initial {field}", blockers)
            label = "theta" if field == "T" else field
            if field == "U":
                offset = 20.0 if case == "igw" else 10.0
                c, r = c - offset, r - offset
            if field == "T":
                if r.ndim != 3:
                    raise ValueError("T must be a complete three-dimensional field")
                if base is None:
                    yield _unavailable(label, index * interval, base_problem)
                    continue
                if base.shape == r.shape:
                    background = base - 300.0
                elif base.ndim == 1 and base.size == r.shape[0]:
                    background = base[:, None, None] - 300.0
                else:
                    reason = "reference thb must be a matching vertical profile or full T-shaped array"
                    if reason not in blockers:
                        blockers.append(reason)
                    yield _unavailable(label, index * interval, reason)
                    continue
                c, r = c - background, r - background
            yield label, index * interval, c, r, True


def verify_fields(case: str, candidate_dir: Path, reference_dir: Path,
                  registration: Path | None = None,
                  candidate_config: Path | None = None,
                  candidate_initializer: Path | None = None) -> dict:
    if case not in CASES:
        raise ValueError(f"unsupported saved-field case: {case}")
    reg, blockers = _read_registration(registration, case, reference_dir,
                                       candidate_config, candidate_initializer)
    report = {"schema": SCHEMA, "case": case,
              "coverage": "saved column profiles and surface series" if case == "gabls1" else "saved horizontal-mean profiles only; no three-dimensional field claim" if case == "cbl_dry" else "all U/V/W/theta cells at all 11 stored leads",
              "registration_sha256": reg.get("registration_sha256"),
              "limits_sha256": reg.get("limits_sha256"),
              "reference_producer": reg.get("reference", {}).get("producer"),
              "configuration_sha256": reg.get("reference", {}).get("configuration_sha256"),
              "evaluator_sha256": sha256_file(Path(__file__)),
              "metric": "FP64 RMS(candidate signal-reference signal)/RMS(reference signal)",
              "reference_base_sha256": reg.get("reference", {}).get("base", {}).get("sha256")
                  if isinstance(reg.get("reference", {}).get("base", {}), dict) else None,
              "thermal_signal": "final theta-reference initial theta" if case == "gabls1"
                  else "final theta-matching initial theta" if case == "cbl_dry"
                  else "stored T-(reference FP32 thb-300)",
              "u_background_subtracted_m_s": 20.0 if case == "igw" else 10.0 if case == "hill2d" else 0.0,
              "blockers": blockers, "comparisons": []}
    try:
        report["artifacts"] = {side: {"file": FILES[case], "sha256": sha256_file(_relative(root, FILES[case]))} for side, root in (("candidate", candidate_dir), ("reference", reference_dir))}
        pairs = _npz_pairs(case, candidate_dir, reference_dir, reg, blockers) if case in ("gabls1", "cbl_dry") else _nc_pairs(case, candidate_dir, reference_dir, reg, blockers)
        for pair in pairs:
            if isinstance(pair, dict):
                report["comparisons"].append(pair)
                continue
            field, seconds, c, r, gated = pair
            row = {"field": field, "lead_seconds": seconds, "gated": gated, **score(c, r)}
            bound = reg.get("limits", {}).get(field)
            row["verdict"] = "diagnostic" if not gated else "blocked"
            if gated and isinstance(bound, dict) and not blockers:
                row["limits"] = bound
                if row["rms_reference_signal"] == 0:
                    row["verdict"] = "pass" if row["rms_delta"] <= bound.get("zero_reference_rms_delta_max", 0.0) else "fail"
                else:
                    row["verdict"] = "pass" if row["normalized_rms"] is not None and row["normalized_rms"] <= bound["nrms_max"] else "fail"
            elif gated and not isinstance(bound, dict):
                reason = f"missing independent limit for {field}"
                if reason not in blockers:
                    blockers.append(reason)
            report["comparisons"].append(row)
        report["coverage_complete"] = not any("error" in row for row in report["comparisons"])
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, FloatingPointError) as exc:
        report["coverage_complete"] = False
        blockers.append(f"saved-field coverage blocked: {exc}")
    if blockers:
        for row in report["comparisons"]:
            if row["gated"]:
                row["verdict"] = "blocked"
    report["verdict"] = "blocked" if blockers else "fail" if any(row["verdict"] == "fail" for row in report["comparisons"]) else "pass"
    report["report_sha256"] = canonical_hash(report)
    return report
