"""Validated host preparation and input fingerprints for CUDA forecasts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import hashlib

import json

from typing import Any, Mapping

import numpy as np

from .cuda_backend import CudaCapability, KernelCache, canonical_sha256

from .cuda_driver import (
    CUDA_AUTHORITY_INITIAL_SCHEMA,
    CUDA_AUTHORITY_RULER_SCHEMA,
    CUDA_IMPLEMENTED_UNLINKED_EVIDENCE,
    CUDA_LAYOUT_CONTRACT,
    CUDA_ORIGINAL_JW_BRANCH_EVIDENCE,
    CUDA_T0_EXACT_SIDECAR,
    CUDA_WHOLE_STEP_EVIDENCE,
    CUDA_V841_AUTHORITY_NONCLAIMS,
    CUDA_V841_IMPLEMENTED_UNLINKED_EVIDENCE,
    FROZEN_CUDA_SOURCE,
    V841_CUDA_SOURCE,
    CudaAuthorityRulerBinder,
    CudaDryDycoreDriver,
    _AUTHORITY_ADVECTION_FLOAT_FIELDS,
    _AUTHORITY_ADVECTION_INDEX_FIELDS,
    _AUTHORITY_MESH_FLOAT_FIELDS,
    _AUTHORITY_MESH_INDEX_FIELDS,
    _AUTHORITY_REFERENCE_FIELDS,
    _AUTHORITY_SIDECAR_FIELDS,
    _AUTHORITY_STATE_FIELDS,
    _AUTHORITY_TERRAIN_FIELDS,
    _AUTHORITY_VERTICAL_FIELDS,
    _validate_initial_fingerprint,
    cuda_authority_initial_fingerprint,
)

from .config_v841 import V841DryDycoreConfig, V841_SOURCE_RELEASE

from .damping_v841 import build_v841_vertical_velocity_damping

from .dynamics_v841 import (
    V841ReferenceWindProfiles,
    precomputed_mesh_inverse_v841,
)

from .driver import (
    DryDycoreConfig,
    DrySavedDiagnostics,
    SPLIT_FLUX_REDUCTION,
    load_mpas_initial_state,
    load_mpas_vertical_grid,
)

from .state import PrognosticState

from .transport import build_advection_coefficients

from .offcentering_v841 import build_v841_acoustic_offcentering

from .errors import ConfigurationRefusal

EXECUTION_INPUT_FINGERPRINT_SCHEMA = "mpas-port.cuda-execution-input-fingerprint/v1"

STATE_FIELDS = ("rho", "rho_theta", "rho_u", "rho_w", "scalars")

SIDECAR_FIELDS = (
    "theta_m",
    "exner",
    "density_perturbation",
    "rho_theta_perturbation",
    "pressure_perturbation",
    "normal_velocity",
    "vertical_velocity",
)

V841_CONTEXT_FIELDS = (
    "inv_area_cell",
    "inv_area_triangle",
    "inv_dc_edge",
    "inv_dv_edge",
    "etp",
    "etm",
    "ewp",
    "ewm",
    "u_init",
    "v_init",
)

class CudaDualRunError(RuntimeError):
    """A deterministic long-run or its evidence is incomplete."""

@dataclass(frozen=True, slots=True)
class PreparedCudaInputs:
    """A profile-tagged host preparation reusable by two CUDA arms."""

    profile: str
    target: str
    preparation_method: str
    mesh: Any
    state: PrognosticState
    vertical: Any
    reference: Any
    saved_diagnostics: DrySavedDiagnostics
    terrain_metrics: Any
    input_bytes: dict[str, dict[str, Any]]
    reference_wind_profiles: V841ReferenceWindProfiles | None = None
    expected_execution_fingerprint: dict[str, Any] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    advection_coefficients: Any | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @classmethod
    def validated(
        cls,
        *,
        config: DryDycoreConfig,
        profile: str,
        target: str,
        preparation_method: str,
        mesh: Any,
        state: PrognosticState,
        vertical: Any,
        reference: Any,
        saved_diagnostics: DrySavedDiagnostics,
        terrain_metrics: Any,
        input_bytes: dict[str, dict[str, Any]],
        reference_wind_profiles: V841ReferenceWindProfiles | None = None,
        allow_regional_sentinels: bool = False,
    ) -> "PreparedCudaInputs":
        """Seal the complete host execution boundary before either upload.

        ``allow_regional_sentinels`` says this mesh's outermost ring carries
        MPAS's limited-area one-cell-edge sentinels by construction, so the
        order-three advection stencil resolves them onto the garbage cell
        instead of refusing.  It is the mesh's own property, read off the
        bdyMask triple by the caller, never a knob a run chooses.
        """

        prepared = cls(
            profile=profile,
            target=target,
            preparation_method=preparation_method,
            mesh=mesh,
            state=state,
            vertical=vertical,
            reference=reference,
            saved_diagnostics=saved_diagnostics,
            terrain_metrics=terrain_metrics,
            input_bytes=input_bytes,
            reference_wind_profiles=reference_wind_profiles,
        )
        is_v841 = getattr(config, "source_release", "v8.2.3") == V841_SOURCE_RELEASE
        if is_v841:
            if not isinstance(config, V841DryDycoreConfig):
                raise CudaDualRunError(
                    "v8.4.1 prepared execution requires V841DryDycoreConfig"
                )
            try:
                CudaDryDycoreDriver._validate_config(config)
            except ConfigurationRefusal as error:
                raise CudaDualRunError(
                    f"v8.4.1 prepared execution is outside the CUDA lane: {error}"
                ) from error
            if reference_wind_profiles is None:
                raise CudaDualRunError(
                    "v8.4.1 prepared execution requires reference wind profiles"
                )
            reference_wind_profiles.validate(
                int(state.rho.shape[0]), np.dtype(np.float32)
            )
        elif reference_wind_profiles is not None:
            raise CudaDualRunError(
                "v8.2.3 prepared execution cannot carry v8.4.1 reference winds"
            )
        coefficients = build_advection_coefficients(
            mesh,
            config_scalar_adv_order=config.config_scalar_adv_order,
            n_vert_levels=state.rho.shape[0],
            source_order_v841=is_v841,
            allow_regional_sentinels=allow_regional_sentinels,
        )
        object.__setattr__(prepared, "advection_coefficients", coefficients)
        sealed = fingerprint_prepared_execution(prepared, config)
        object.__setattr__(prepared, "expected_execution_fingerprint", sealed)
        return prepared

def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        import cupy as cp
    except Exception:
        return np.asarray(value)
    if isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    return np.asarray(value)

def fingerprint_array(name: str, value: Any) -> dict[str, Any]:
    """Hash one logical array, binding bytes to its name, dtype, and shape."""

    array = np.ascontiguousarray(_to_numpy(value))
    if array.dtype.hasobject:
        raise CudaDualRunError(f"{name} has an object dtype")
    if array.dtype.kind in "fc" and not bool(np.all(np.isfinite(array))):
        raise CudaDualRunError(f"{name} contains non-finite values")
    raw_sha256 = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    core = {
        "name": str(name),
        "dtype": array.dtype.str,
        "shape": [int(value) for value in array.shape],
        "bytes": int(array.nbytes),
        "bytes_sha256": raw_sha256,
    }
    return {**core, "sha256": _json_sha256(core)}

def fingerprint_array_group(
    values: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not values:
        raise CudaDualRunError("an array fingerprint group cannot be empty")
    fields = {name: fingerprint_array(name, values[name]) for name in sorted(values)}
    core = {"metadata": dict(metadata or {}), "fields": fields}
    return {**core, "sha256": _json_sha256(core)}

def fingerprint_atmosphere(atmosphere: Any) -> dict[str, Any]:
    """Explicitly download and hash one resident state/sidecar snapshot."""

    state = atmosphere.state
    saved = atmosphere.saved
    state_record = fingerprint_array_group(
        {name: getattr(state, name) for name in STATE_FIELDS},
        metadata={"time_seconds": float(state.time_seconds)},
    )
    sidecar_record = fingerprint_array_group(
        {name: getattr(saved, name) for name in SIDECAR_FIELDS},
        metadata={"kind": "DrySavedDiagnostics"},
    )
    core = {
        "model_time_seconds": float(state.time_seconds),
        "state": state_record,
        "saved_diagnostics": sidecar_record,
    }
    return {**core, "sha256": _json_sha256(core)}

def _authority_array_record(value: Any, *, dtype: Any | None = None) -> dict[str, Any]:
    array = _to_numpy(value)
    if dtype is not None:
        array = np.asarray(array, dtype=dtype)
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype.hasobject:
        raise CudaDualRunError("execution-input arrays cannot use object dtype")
    if contiguous.dtype.kind in "fc" and not bool(np.all(np.isfinite(contiguous))):
        raise CudaDualRunError("execution-input arrays must be finite")
    raw = contiguous.tobytes(order="C")
    return {
        "dtype": contiguous.dtype.str,
        "shape": [int(extent) for extent in contiguous.shape],
        "c_bytes": len(raw),
        "c_bytes_sha256": hashlib.sha256(raw).hexdigest(),
    }

def _host_mesh_value(mesh: Any, name: str) -> Any:
    if hasattr(mesh, name):
        return getattr(mesh, name)
    arrays = getattr(mesh, "arrays", None)
    if isinstance(arrays, Mapping) and name in arrays:
        return arrays[name]
    raise CudaDualRunError(f"prepared mesh is missing {name}")

def _host_edge_sign_record(mesh: Any) -> dict[str, Any]:
    """Rebuild the sole derived DeviceMesh array in upload source order."""

    edges_on_cell = np.ascontiguousarray(
        np.asarray(_host_mesh_value(mesh, "edgesOnCell"), dtype=np.int32)
    )
    cells_on_edge = np.ascontiguousarray(
        np.asarray(_host_mesh_value(mesh, "cellsOnEdge"), dtype=np.int32)
    )
    counts = np.ascontiguousarray(
        np.asarray(_host_mesh_value(mesh, "nEdgesOnCell"), dtype=np.int32)
    )
    signs = np.zeros(edges_on_cell.shape, dtype=np.float32)
    for cell in range(edges_on_cell.shape[0]):
        for slot in range(int(counts[cell])):
            edge = int(edges_on_cell[cell, slot])
            signs[cell, slot] = (
                np.float32(1.0)
                if int(cells_on_edge[edge, 0]) == cell
                else np.float32(-1.0)
            )
    return _authority_array_record(signs)

def _execution_fingerprint_core(
    authority_initial: Mapping[str, Any],
    configuration_sha256: str,
    edge_sign_on_cell: Mapping[str, Any],
    v841_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    derived = {"mesh.edge_sign_on_cell": dict(edge_sign_on_cell)}
    if v841_context is not None:
        if set(v841_context) != set(V841_CONTEXT_FIELDS):
            raise CudaDualRunError("v8.4.1 context fingerprint inventory changed")
        derived.update(
            {
                f"v841_context.{name}": dict(v841_context[name])
                for name in V841_CONTEXT_FIELDS
            }
        )
    return {
        "schema": EXECUTION_INPUT_FINGERPRINT_SCHEMA,
        "configuration_sha256": configuration_sha256,
        "authority_initial": dict(authority_initial),
        "derived_upload_arrays": derived,
    }

def _finalize_execution_fingerprint(core: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(core), "sha256": _json_sha256(core)}

def fingerprint_prepared_execution(
    prepared: PreparedCudaInputs,
    config: DryDycoreConfig,
) -> dict[str, Any]:
    """Hash every host value that controls a generic CUDA trajectory."""

    configuration_sha256 = canonical_sha256(asdict(config))
    authority_initial = cuda_authority_initial_fingerprint(
        prepared.state,
        prepared.saved_diagnostics,
        mesh=prepared.mesh,
        vertical=prepared.vertical,
        reference=prepared.reference,
        terrain_metrics=prepared.terrain_metrics,
        config=config,
        advection_coefficients=prepared.advection_coefficients,
    )
    source_release = getattr(config, "source_release", "v8.2.3")
    v841_context = None
    if source_release == V841_SOURCE_RELEASE:
        if prepared.reference_wind_profiles is None:
            raise CudaDualRunError(
                "v8.4.1 execution fingerprint requires reference wind profiles"
            )
        selected_dss = build_v841_vertical_velocity_damping(
            np.asarray(prepared.vertical.zgrid),
            xnutr=config.config_xnutr,
            damping_start_z=config.config_zd,
        )
        vertical_group = authority_initial["execution_inputs"]["vertical"]
        vertical_group["arrays"]["dss"] = _authority_array_record(
            selected_dss, dtype=np.float32
        )
        for name in ("cf1", "cf2", "cf3"):
            vertical_group["scalars"][name] = float(
                np.float32(getattr(prepared.vertical, name))
            )
        advection_group = authority_initial["execution_inputs"]["advection"]
        for name in _AUTHORITY_ADVECTION_FLOAT_FIELDS:
            advection_group["arrays"][name] = _authority_array_record(
                getattr(prepared.advection_coefficients, name), dtype=np.float32
            )
        for name in _AUTHORITY_ADVECTION_INDEX_FIELDS:
            advection_group["arrays"][name] = _authority_array_record(
                getattr(prepared.advection_coefficients, name), dtype=np.int32
            )

        offcentering = build_v841_acoustic_offcentering(
            np.asarray(prepared.vertical.rdzw),
            minimum=config.config_epssm_minimum,
            maximum=config.config_epssm_maximum,
            transition_bottom_z=config.config_epssm_transition_bottom_z,
            transition_top_z=config.config_epssm_transition_top_z,
        )
        inverse_fields = {
            "inv_area_cell": "areaCell",
            "inv_area_triangle": "areaTriangle",
            "inv_dc_edge": "dcEdge",
            "inv_dv_edge": "dvEdge",
        }
        context_values: dict[str, Any] = {
            name: precomputed_mesh_inverse_v841(
                prepared.mesh, field_name, np.dtype(np.float32)
            )
            for name, field_name in inverse_fields.items()
        }
        context_values.update(
            {
                name: getattr(offcentering, name)
                for name in ("etp", "etm", "ewp", "ewm")
            }
        )
        context_values.update(
            {
                "u_init": prepared.reference_wind_profiles.u_init,
                "v_init": prepared.reference_wind_profiles.v_init,
            }
        )
        v841_context = {
            name: _authority_array_record(context_values[name], dtype=np.float32)
            for name in V841_CONTEXT_FIELDS
        }
    core = _execution_fingerprint_core(
        authority_initial,
        configuration_sha256,
        _host_edge_sign_record(prepared.mesh),
        v841_context,
    )
    return _finalize_execution_fingerprint(core)
