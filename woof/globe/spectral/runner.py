"""Standalone execution, output, gates, and receipts for Level 3."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np

from .checkpoint import read_checkpoint, state_from_checkpoint, write_checkpoint
from .config import GlobalSpectralRunConfig
from .constants import SECONDS_PER_DAY
from .diffusion import ExponentialHyperdiffusion
from .initial_conditions import primitive_rest_state, williamson2_state
from .primitive import HeldSuarezForcing, PrimitiveDryModel, SigmaCoordinate
from .receipt import check_receipt, write_receipt
from .shallow_water import ShallowWaterModel
from .transform import SphericalHarmonicTransform


_OWNED_OUTPUT_NAMES = {"diagnostics.jsonl", "global-spectral-receipt.json"}


def build_transform(cfg: GlobalSpectralRunConfig) -> SphericalHarmonicTransform:
    return SphericalHarmonicTransform.create(
        cfg.truncation,
        nlat=cfg.nlat,
        nlon=cfg.nlon,
        dealias_factor=cfg.dealias_factor,
        radius_m=cfg.radius_m,
        backend=cfg.backend,
        precision=cfg.precision,
    )


def _diffusion(cfg: GlobalSpectralRunConfig):
    if not cfg.diffusion_enabled:
        return None
    return ExponentialHyperdiffusion(
        order=cfg.diffusion_order,
        e_folding_time_s_at_truncation=cfg.diffusion_efold_s,
        preserve_degree=cfg.diffusion_preserve_degree,
    )


def _copy_state(state):
    return state.with_fields([field.copy() for field in state.fields()])


def build_model_and_state(
    cfg: GlobalSpectralRunConfig,
    transform: SphericalHarmonicTransform,
    restart: str | Path | None = None,
):
    """Return ``(model, live_state, cold_reference, restart_metadata)``.

    The primitive model's mass-fixer target is always initialized from the
    deterministic cold state, not from the checkpoint.  A restart therefore
    cannot silently redefine the conserved global mean merely because it was
    written after rounding accumulated.
    """
    if cfg.model == "shallow-water":
        model = ShallowWaterModel(
            transform,
            rotation_rate_s=cfg.rotation_rate_s,
            integrator=cfg.integrator,
            diffusion=_diffusion(cfg),
            divergence_diffusion_strength=cfg.divergence_diffusion_strength,
            geopotential_diffusion_strength=cfg.pressure_diffusion_strength,
            maximum_cfl=cfg.maximum_cfl,
        )
        cold = williamson2_state(
            transform,
            alpha_rad=cfg.williamson_alpha_rad,
            u0_m_s=cfg.williamson_u0_m_s,
            mean_geopotential_m2_s2=cfg.williamson_mean_geopotential,
            rotation_rate_s=cfg.rotation_rate_s,
        )
    else:
        sigma = SigmaCoordinate(np.asarray(cfg.sigma_half, dtype=np.float64))
        model = PrimitiveDryModel(
            transform,
            sigma,
            rotation_rate_s=cfg.rotation_rate_s,
            integrator=cfg.integrator,
            diffusion=_diffusion(cfg),
            divergence_diffusion_strength=cfg.divergence_diffusion_strength,
            pressure_diffusion_strength=cfg.pressure_diffusion_strength,
            held_suarez=HeldSuarezForcing(enabled=cfg.held_suarez),
            mass_fixer=cfg.mass_fixer,
            maximum_cfl=cfg.maximum_cfl,
        )
        cold = primitive_rest_state(
            transform,
            sigma.full_levels,
            surface_pressure_pa=cfg.primitive_surface_pressure_pa,
            temperature_surface_k=cfg.primitive_surface_temperature_k,
            temperature_top_k=cfg.primitive_top_temperature_k,
            perturbation_k=cfg.primitive_temperature_perturbation_k,
            zonal_wavenumber=cfg.primitive_zonal_wavenumber,
        )
        model.initialize_mass_target(cold)

    model.enforce(cold)
    state = _copy_state(cold)
    restart_metadata = None
    if restart is not None:
        restart_metadata, arrays = read_checkpoint(
            restart, expected_config_hash=cfg.config_hash
        )
        if restart_metadata["model"] != cfg.model:
            raise ValueError("restart model does not match config")
        state = state_from_checkpoint(restart_metadata, arrays, transform.backend)
        if state.step >= cfg.steps:
            raise ValueError(
                f"restart step {state.step} is at or beyond configured final step "
                f"{cfg.steps}; there is nothing left to integrate"
            )
        expected_time = state.step * cfg.dt_s
        tolerance = max(1.0e-9, 1.0e-12 * max(1.0, expected_time))
        if abs(state.time_s - expected_time) > tolerance:
            raise ValueError(
                f"restart clock mismatch: step {state.step} at {state.time_s:g} s, "
                f"expected {expected_time:g} s from dt_s={cfg.dt_s:g}"
            )
        model.enforce(state)
    return model, state, cold, restart_metadata


def _normalized_l2(grid, candidate, reference) -> float:
    error = np.asarray(candidate) - np.asarray(reference)
    num = math.sqrt(max(0.0, grid.global_mean(error * error)))
    den = math.sqrt(max(1.0e-30, grid.global_mean(np.asarray(reference) ** 2)))
    return num / den


def _vector_normalized_l2(grid, u, v, reference_u, reference_v) -> float:
    du = np.asarray(u) - np.asarray(reference_u)
    dv = np.asarray(v) - np.asarray(reference_v)
    numerator = math.sqrt(max(0.0, grid.global_mean(du * du + dv * dv)))
    denominator = math.sqrt(
        max(
            1.0e-30,
            grid.global_mean(
                np.asarray(reference_u) ** 2 + np.asarray(reference_v) ** 2
            ),
        )
    )
    return numerator / denominator


def williamson2_analytic_fields(cfg: GlobalSpectralRunConfig, grid):
    """Closed-form Williamson test-case-2 wind and geopotential on ``grid``.

    Evaluated straight from the published expressions with no model operator
    in the path -- not the transform, not the vorticity/divergence inversion,
    not the initial-condition builder -- so the gates that use it are anchored
    outside the machinery they judge.  Only the admitted alpha=0 orientation
    has this closed form.
    """
    if abs(float(cfg.williamson_alpha_rad)) > 1.0e-14:
        raise ValueError(
            "the closed-form Williamson-2 reference is written for alpha_rad=0 "
            "only; a tilted axis would be compared against the wrong analytic "
            "state and the gate would report skill the run does not have"
        )
    a = grid.radius_m
    u0 = (
        2.0 * math.pi * a / (12.0 * SECONDS_PER_DAY)
        if cfg.williamson_u0_m_s is None
        else float(cfg.williamson_u0_m_s)
    )
    lat, _lon = grid.mesh()
    sin_lat = np.sin(lat)
    eastward = u0 * np.cos(lat)
    northward = np.zeros_like(eastward)
    geopotential = cfg.williamson_mean_geopotential - (
        a * cfg.rotation_rate_s * u0 + 0.5 * u0 * u0
    ) * sin_lat * sin_lat
    return eastward, northward, geopotential


def _gate(name: str, value: float, limit: float, comparison: str = "<=") -> dict:
    passed = value <= limit if comparison == "<=" else value >= limit
    return {
        "name": name,
        "value": float(value),
        "limit": float(limit),
        "comparison": comparison,
        "passed": bool(passed),
    }


def _owned_outputs(output: Path) -> list[Path]:
    owned = [output / name for name in sorted(_OWNED_OUTPUT_NAMES)]
    owned.extend(sorted(output.glob("global_spectral_step*.npz")))
    return [path for path in owned if path.exists()]


def _prepare_output(output: Path, *, restart, overwrite: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    owned = _owned_outputs(output)
    if restart is not None:
        if overwrite:
            raise ValueError("--overwrite cannot be combined with --restart")
        if not owned:
            return
        restart_path = Path(restart).resolve()
        checkpoints = sorted(output.glob("global_spectral_step*.npz"))
        if restart_path.parent != output.resolve():
            raise FileExistsError(
                f"restart source {restart_path} is outside non-empty output "
                f"directory {output}; use an empty directory or resume in the "
                "checkpoint's own directory"
            )
        if not checkpoints or checkpoints[-1].resolve() != restart_path:
            latest = "none" if not checkpoints else str(checkpoints[-1])
            raise ValueError(
                f"restart checkpoint must be the latest Level-3 checkpoint in "
                f"its non-empty output directory; requested {restart_path}, "
                f"latest is {latest}"
            )
        return
    if owned and not overwrite:
        names = ", ".join(path.name for path in owned[:6])
        suffix = " ..." if len(owned) > 6 else ""
        raise FileExistsError(
            f"output directory {output} already contains Level-3 run output "
            f"({names}{suffix}); use a new directory or pass --overwrite"
        )
    if overwrite:
        for path in owned:
            path.unlink()


def run(
    cfg: GlobalSpectralRunConfig,
    outdir: str | Path,
    *,
    restart: str | Path | None = None,
    progress=None,
    overwrite: bool = False,
) -> dict:
    output = Path(outdir)
    _prepare_output(output, restart=restart, overwrite=overwrite)
    transform = build_transform(cfg)
    transform_check = transform.transform_check(seed=7)
    model, state, cold_reference, restart_metadata = build_model_and_state(
        cfg, transform, restart=restart
    )
    segment_initial = _copy_state(state)
    cold_diagnostics = model.diagnostics(cold_reference)
    segment_initial_diagnostics = model.diagnostics(segment_initial)

    inherited_trackers = (
        {"maximum_spectral_cfl": 0.0, "maximum_mass_fixer_log_offset": 0.0}
        if restart_metadata is None
        else dict(restart_metadata["run_trackers"])
    )
    maximum_cfl = float(inherited_trackers["maximum_spectral_cfl"])
    mass_fixer_max = float(inherited_trackers["maximum_mass_fixer_log_offset"])
    segment_maximum_cfl = 0.0
    segment_mass_fixer_max = 0.0

    initial_checkpoint = write_checkpoint(
        output / f"global_spectral_step{state.step:08d}.npz",
        state,
        model=cfg.model,
        config_hash=cfg.config_hash,
        to_numpy=transform.backend.to_numpy,
        run_trackers={
            "maximum_spectral_cfl": maximum_cfl,
            "maximum_mass_fixer_log_offset": mass_fixer_max,
        },
    )
    diagnostics_path = output / "diagnostics.jsonl"
    append = restart is not None and diagnostics_path.exists()
    mode = "a" if append else "w"
    checkpoints = [str(initial_checkpoint)]
    segment_start_time_s = float(state.time_s)
    segment_start_step = int(state.step)
    start_wall = time.perf_counter()
    try:
        with diagnostics_path.open(mode, encoding="utf-8", newline="\n") as log:
            if append:
                log.write(
                    json.dumps(
                        {
                            "event": "restart",
                            "step": segment_start_step,
                            "time_s": segment_start_time_s,
                            "checkpoint": str(Path(restart)),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            else:
                row = dict(segment_initial_diagnostics)
                row["event"] = "initial"
                log.write(json.dumps(row, sort_keys=True) + "\n")
            while state.step < cfg.steps:
                if cfg.model == "shallow-water":
                    cfl = model.cfl(state, cfg.dt_s)
                    state = model.step(state, cfg.dt_s)
                    step_meta = {
                        "spectral_cfl": cfl,
                        "mass_fixer_log_offset": 0.0,
                    }
                else:
                    state, step_meta = model.step(state, cfg.dt_s)
                step_cfl = float(step_meta["spectral_cfl"])
                step_mass_fix = abs(float(step_meta["mass_fixer_log_offset"]))
                segment_maximum_cfl = max(segment_maximum_cfl, step_cfl)
                segment_mass_fixer_max = max(segment_mass_fixer_max, step_mass_fix)
                maximum_cfl = max(maximum_cfl, step_cfl)
                mass_fixer_max = max(mass_fixer_max, step_mass_fix)
                if state.step % cfg.output_steps == 0 or state.step == cfg.steps:
                    diag = model.diagnostics(state)
                    diag.update(step_meta)
                    diag["event"] = "output"
                    log.write(json.dumps(diag, sort_keys=True) + "\n")
                    log.flush()
                    checkpoint = write_checkpoint(
                        output / f"global_spectral_step{state.step:08d}.npz",
                        state,
                        model=cfg.model,
                        config_hash=cfg.config_hash,
                        to_numpy=transform.backend.to_numpy,
                        run_trackers={
                            "maximum_spectral_cfl": maximum_cfl,
                            "maximum_mass_fixer_log_offset": mass_fixer_max,
                        },
                    )
                    checkpoints.append(str(checkpoint))
                    if progress is not None:
                        progress(diag)
    except Exception as exc:
        # A run that fails after publishing its initial checkpoint still gets
        # a durable, self-hashed account of what happened.  The last good
        # state remains in ``state`` because every model step validates before
        # assignment; the receipt therefore identifies the exact restart point
        # and carries the whole-run maxima accumulated up to the failure.
        transform.backend.synchronize()
        failure_wall = time.perf_counter() - start_wall
        try:
            failure_diagnostics = model.diagnostics(state)
        except Exception:
            failure_diagnostics = None
        failure_payload = {
            "name": cfg.name,
            "model": cfg.model,
            "research_only": True,
            "acknowledgement": cfg.acknowledgement,
            "config": cfg.canonical_dict(),
            "config_hash": cfg.config_hash,
            "backend": cfg.backend,
            "precision": cfg.precision,
            "grid": {
                "truncation": cfg.truncation,
                "nlat": transform.grid.nlat,
                "nlon": transform.grid.nlon,
                "dealias_factor": cfg.dealias_factor,
            },
            "transform_check": transform_check,
            "cold_start_diagnostics": cold_diagnostics,
            "segment_initial_diagnostics": segment_initial_diagnostics,
            "last_good_diagnostics": failure_diagnostics,
            "segment_start_step": segment_start_step,
            "segment_start_time_s": segment_start_time_s,
            "completed_step": int(state.step),
            "completed_time_s": float(state.time_s),
            "resumed_from": None if restart is None else str(Path(restart)),
            "restart_metadata_self_sha256": (
                None
                if restart_metadata is None
                else restart_metadata["self_sha256"]
            ),
            "maximum_spectral_cfl": maximum_cfl,
            "maximum_mass_fixer_log_offset": mass_fixer_max,
            "segment_maximum_spectral_cfl": segment_maximum_cfl,
            "segment_maximum_mass_fixer_log_offset": segment_mass_fixer_max,
            "run_trackers_from_restart": inherited_trackers,
            "wall_seconds": failure_wall,
            "checkpoints": checkpoints,
            "diagnostics_jsonl": str(diagnostics_path),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "status": "error",
        }
        write_receipt(output / "global-spectral-receipt.json", failure_payload)
        raise
    transform.backend.synchronize()
    wall = time.perf_counter() - start_wall
    final_diagnostics = model.diagnostics(state)

    gates = [
        _gate(
            "transform_roundtrip_relative_linf",
            transform_check["roundtrip_relative_linf"],
            cfg.gate_transform_roundtrip,
        ),
        _gate(
            "transform_parseval_relative_error",
            transform_check["parseval_relative_error"],
            cfg.gate_transform_parseval,
        ),
        _gate("maximum_spectral_cfl", maximum_cfl, cfg.maximum_cfl),
        _gate("completed_steps", float(state.step), float(cfg.steps), comparison=">="),
    ]
    if cfg.model == "shallow-water":
        phi0 = transform.backend.to_numpy(
            transform.inverse(cold_reference.geopotential)
        )
        phif = transform.backend.to_numpy(transform.inverse(state.geopotential))
        zeta0 = transform.backend.to_numpy(
            transform.inverse(cold_reference.vorticity)
        )
        zetaf = transform.backend.to_numpy(transform.inverse(state.vorticity))
        div0 = transform.backend.to_numpy(
            transform.inverse(cold_reference.divergence)
        )
        divf = transform.backend.to_numpy(transform.inverse(state.divergence))
        u0, v0 = model.vector.wind_from_vordiv(
            cold_reference.vorticity, cold_reference.divergence
        )
        uf, vf = model.vector.wind_from_vordiv(state.vorticity, state.divergence)
        u0 = transform.backend.to_numpy(u0)
        v0 = transform.backend.to_numpy(v0)
        uf = transform.backend.to_numpy(uf)
        vf = transform.backend.to_numpy(vf)
        williamson_l2 = _normalized_l2(transform.grid, phif, phi0)
        williamson_zeta_l2 = _normalized_l2(transform.grid, zetaf, zeta0)
        williamson_wind_l2 = _vector_normalized_l2(
            transform.grid, uf, vf, u0, v0
        )
        analytic_u, analytic_v, analytic_phi = williamson2_analytic_fields(
            cfg, transform.grid
        )
        williamson_analytic_l2 = _normalized_l2(
            transform.grid, phif, analytic_phi
        )
        williamson_wind_analytic_l2 = _vector_normalized_l2(
            transform.grid, uf, vf, analytic_u, analytic_v
        )
        zeta_scale = math.sqrt(
            max(1.0e-30, transform.grid.global_mean(zeta0 * zeta0))
        )
        williamson_divergence_scaled_rms = math.sqrt(
            max(
                0.0,
                transform.grid.global_mean((divf - div0) * (divf - div0)),
            )
        ) / zeta_scale
        mass_drift = abs(
            final_diagnostics["mass_kg_m2_mean"]
            - cold_diagnostics["mass_kg_m2_mean"]
        ) / abs(cold_diagnostics["mass_kg_m2_mean"])
        energy_drift = abs(
            final_diagnostics["total_energy_j_m2_mean"]
            - cold_diagnostics["total_energy_j_m2_mean"]
        ) / abs(cold_diagnostics["total_energy_j_m2_mean"])
        enstrophy_drift = abs(
            final_diagnostics["potential_enstrophy"]
            - cold_diagnostics["potential_enstrophy"]
        ) / abs(cold_diagnostics["potential_enstrophy"])
        gates.extend(
            [
                # The analytic pair is the only shallow-water gate that can
                # see an error shared by the initial state and the final
                # state; the steadiness gates below compare the run against
                # its own cold reference and stay at roundoff whatever
                # williamson2_state and wind_from_vordiv agree on.
                _gate(
                    "williamson2_geopotential_analytic_normalized_l2",
                    williamson_analytic_l2,
                    cfg.gate_williamson_l2,
                ),
                _gate(
                    "williamson2_wind_vector_analytic_normalized_l2",
                    williamson_wind_analytic_l2,
                    cfg.gate_williamson_l2,
                ),
                _gate(
                    "williamson2_geopotential_steadiness_l2",
                    williamson_l2,
                    cfg.gate_williamson_l2,
                ),
                _gate(
                    "williamson2_vorticity_steadiness_l2",
                    williamson_zeta_l2,
                    cfg.gate_williamson_l2,
                ),
                _gate(
                    "williamson2_wind_vector_steadiness_l2",
                    williamson_wind_l2,
                    cfg.gate_williamson_l2,
                ),
                _gate(
                    "williamson2_divergence_scaled_rms",
                    williamson_divergence_scaled_rms,
                    cfg.gate_williamson_l2,
                ),
                _gate(
                    "shallow_water_mass_relative_drift",
                    mass_drift,
                    cfg.gate_mass_relative_drift,
                ),
                _gate(
                    "shallow_water_energy_relative_drift",
                    energy_drift,
                    cfg.gate_mass_relative_drift,
                ),
                _gate(
                    "shallow_water_potential_enstrophy_relative_drift",
                    enstrophy_drift,
                    cfg.gate_mass_relative_drift,
                ),
            ]
        )
    else:
        williamson_zeta_l2 = None
        williamson_wind_l2 = None
        williamson_divergence_scaled_rms = None
        williamson_analytic_l2 = None
        williamson_wind_analytic_l2 = None
        energy_drift = None
        enstrophy_drift = None
        mass_drift = abs(
            final_diagnostics["global_mean_surface_pressure_pa"]
            - cold_diagnostics["global_mean_surface_pressure_pa"]
        ) / abs(cold_diagnostics["global_mean_surface_pressure_pa"])
        gates.extend(
            [
                _gate(
                    "surface_pressure_mass_relative_drift",
                    mass_drift,
                    cfg.gate_mass_relative_drift,
                ),
                # With the default mass fixer on, the drift above is whatever
                # _fix_mass forced it to be; the mass the dynamics actually
                # lost is the offset the fixer had to add back.  A single
                # step is never allowed to move more mass than the whole run
                # is allowed to drift, so the fixer cannot hide a loss the
                # drift gate would have caught with the fixer off.  Measured
                # 2.63e-12 against a 1.0e-10 limit on the shipped Held-Suarez
                # one-day smoke (T10, 288 steps).
                _gate(
                    "maximum_mass_fixer_log_offset",
                    mass_fixer_max,
                    cfg.gate_mass_relative_drift,
                ),
            ]
        )
        williamson_l2 = None

    status = "pass" if all(row["passed"] for row in gates) else "fail"
    simulated_segment = cfg.duration_s - segment_start_time_s
    receipt_payload = {
        "name": cfg.name,
        "model": cfg.model,
        "research_only": True,
        "acknowledgement": cfg.acknowledgement,
        "config": cfg.canonical_dict(),
        "config_hash": cfg.config_hash,
        "backend": cfg.backend,
        "precision": cfg.precision,
        "grid": {
            "truncation": cfg.truncation,
            "nlat": transform.grid.nlat,
            "nlon": transform.grid.nlon,
            "dealias_factor": cfg.dealias_factor,
        },
        "transform_check": transform_check,
        "cold_start_diagnostics": cold_diagnostics,
        "segment_initial_diagnostics": segment_initial_diagnostics,
        "final_diagnostics": final_diagnostics,
        "segment_start_step": segment_start_step,
        "segment_start_time_s": segment_start_time_s,
        "completed_step": int(state.step),
        "completed_time_s": float(state.time_s),
        "resumed_from": None if restart is None else str(Path(restart)),
        "restart_metadata_self_sha256": (
            None if restart_metadata is None else restart_metadata["self_sha256"]
        ),
        "maximum_spectral_cfl": maximum_cfl,
        "maximum_mass_fixer_log_offset": mass_fixer_max,
        "segment_maximum_spectral_cfl": segment_maximum_cfl,
        "segment_maximum_mass_fixer_log_offset": segment_mass_fixer_max,
        "run_trackers_from_restart": inherited_trackers,
        "mass_relative_drift": mass_drift,
        "williamson2_geopotential_analytic_normalized_l2": williamson_analytic_l2,
        "williamson2_wind_vector_analytic_normalized_l2": williamson_wind_analytic_l2,
        "williamson2_geopotential_steadiness_l2": williamson_l2,
        "williamson2_vorticity_steadiness_l2": williamson_zeta_l2,
        "williamson2_wind_vector_steadiness_l2": williamson_wind_l2,
        "williamson2_divergence_scaled_rms": williamson_divergence_scaled_rms,
        "shallow_water_energy_relative_drift": energy_drift,
        "shallow_water_potential_enstrophy_relative_drift": enstrophy_drift,
        "wall_seconds": wall,
        "simulated_seconds_per_wall_second": simulated_segment
        / max(wall, 1.0e-12),
        "checkpoints": checkpoints,
        "diagnostics_jsonl": str(diagnostics_path),
        "gates": gates,
        "status": status,
    }
    receipt_path = write_receipt(
        output / "global-spectral-receipt.json", receipt_payload
    )
    checked = check_receipt(receipt_path)
    checked["receipt_path"] = str(receipt_path)
    return checked
