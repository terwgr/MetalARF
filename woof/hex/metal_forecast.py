"""Run the public Hex forecast door on Metal.

The route runs the CUDA lane's full physics row: the resident v8.4.1 dycore
(``metal_driver_v841``) and the two-phase column physics
(``metal_arwen_physics_v841``), prepared by the same host code as the CUDA
forecast, so it needs ``--grid`` and ``--static`` as that door does, and
``--lbc-dir`` on a limited-area row.  Every forecast flag the route does not
implement is refused by name rather than dropped.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


def _write_metal_snapshot(runner, path, snapshot, static, *, valid_time, physics_identity, wind_farms=None):
    """Use the shared history layout with the executing backend's metadata."""
    from netCDF4 import Dataset

    scheduled = dict(snapshot, valid_time=_start_stamp(valid_time))
    runner.proof.write_snapshot_netcdf(path, scheduled, static, _hash_file=False)
    with Dataset(path, "r+") as dataset:
        dataset.setncattr("source", "WOOF Hex Metal port")
        if wind_farms is not None:
            for name, (value, dims, units) in wind_farms.history_fields().items():
                for dim, size in zip(dims, value.shape):
                    if dim not in dataset.dimensions:
                        dataset.createDimension(dim, size)
                var = dataset.createVariable(name, 'f4', dims)
                var[:] = value
                var.units = units
            dataset.setncattr('WIND_FARMS', wind_farms.metadata())
        dataset.setncattr("backend", "metal")
        dataset.setncattr("kernel_arithmetic", "metal_ordinary")
        dataset.setncattr("physics_mode", physics_identity["physics_mode"])
        generations = physics_identity["source_generations"]
        dataset.setncattr("solar_generation", generations["solar"])
        dataset.setncattr("ysu_version", generations["ysu"])
        dataset.setncattr("noahmp_snow_age", generations["noahmp_snow_age"])
        dataset.setncattr("physics_configuration_sha256",
                         physics_identity["configuration_sha256"])
        dataset.setncattr("physics_source_generations", json.dumps(generations, sort_keys=True))
        dataset.setncattr("gf_seam_parity_closed", "false")
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": runner.proof.sha256_file(path)}


def _initial_time(dataset: Any) -> datetime:
    raw = None
    if "initial_time" in dataset.variables:
        raw = dataset.variables["initial_time"][:]
    elif "xtime" in dataset.variables:
        raw = dataset.variables["xtime"][0]
    if raw is None:
        raise ValueError("the MPAS init has neither initial_time nor xtime")
    if isinstance(raw, np.ndarray):
        raw = b"".join(raw.reshape(-1).tolist())
    if isinstance(raw, bytes):
        value = raw.decode("ascii", errors="strict").strip("\x00 ")
    else:
        value = str(raw).strip()
    value = value.replace("_", "T")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


# Forecast flags this route reads.  ``--repo`` is accepted and unused on
# every backend.
_IMPLEMENTED = frozenset({
    "physics_mode", "physics_correct_skip",
    "mesh", "backend", "init", "init_source", "start_time", "hours",
    "history_every_minutes", "out", "receipt", "case_label", "repo",
    "horiz_mixing", "checkpoint", "restart", "lbc_dir",
    "grid", "static", "convection", "pbl_cadence",
    "dt", "time_integration_order", "acoustic_substeps", "radiation_seconds",
    "monotonic",
    "local_timestep", "local_timestep_rates", "local_timestep_buffer_rings",
    "local_timestep_classing",
    "preflight", "preset", "stop_on_refusal", "physics_backend", "source_table",
    "physics_suite", "physics", "orographic_statistics", "wind_farms",
})

# Flags that configure CUDA machinery Metal does not have, and why.
_CUDA_ONLY = {
    "scratch": "is the CUDA kernel cache; Metal runs its prebuilt library",
    "gpuwm_checkout": "names the tree the CUDA seam files run from",
    "headroom_mib": ("is a CUDA card admission term; Metal admits against "
                     "WOOF_METAL_MEMORY_FRACTION of its working set"),
    "device_fixed_mib": ("is a CUDA card admission term; Metal admits against "
                         "WOOF_METAL_MEMORY_FRACTION of its working set"),
    "device_bytes_per_cell": ("is a CUDA card admission term; Metal admits "
                              "against WOOF_METAL_MEMORY_FRACTION of its "
                              "working set"),
}


def _refuse_unimplemented(arguments: Any, refuse: Any, *, implemented=_IMPLEMENTED) -> None:
    from .forecast_door import add_forecast_arguments

    parser = argparse.ArgumentParser(add_help=False)
    add_forecast_arguments(parser)
    for action in parser._actions:
        dest = action.dest
        if dest in implemented or not hasattr(arguments, dest):
            continue
        if getattr(arguments, dest) != action.default:
            if dest in _CUDA_ONLY:
                raise refuse(f"{action.option_strings[0]} {_CUDA_ONLY[dest]}; "
                             "omit it with --backend metal")
            raise refuse(
                f"{action.option_strings[0]} is not implemented by "
                "--backend metal"
            )


def _solve_region(value: Any, n_cells: int, n_edges: int) -> Any:
    """Drop a limited-area run's padded garbage column; global arrays pass."""

    from dataclasses import fields, replace

    solve = {n_cells + 1: n_cells, n_edges + 1: n_edges}
    sliced = {}
    for field in fields(value):
        array = getattr(value, field.name)
        if isinstance(array, np.ndarray) and array.shape[-1:] and array.shape[-1] in solve:
            sliced[field.name] = np.ascontiguousarray(array[..., :solve[array.shape[-1]]])
    return replace(value, **sliced)


def _start_stamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d_%H:%M:%S")


def _restart_setup(host: Any, paths: dict[str, Path], *,
                   local_timestep: bool, rates: Any, rings: int,
                   known: dict[str, str] | None = None,
                   classing_path: Path | None = None,
                   orographic_statistics: Path | None = None) -> dict[str, Any]:
    """Bind the dynamics configuration and caller-owned static carriers.

    ``known`` holds digests already taken (the authority check), by role.
    """
    from dataclasses import asdict
    from hashlib import sha256

    def digest(path: Path) -> str:
        for role, value in (known or {}).items():
            if value and Path(paths[role]) == Path(path):
                return value
        result = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(chunk)
        return result.hexdigest()

    return {
        "config": json.loads(json.dumps(asdict(host["config"]))),
        "physics_backend": host["physics_backend"],
        "scalar_names": list(host["scalar_names"]),
        "grid_sha256": digest(paths["grid"]),
        "static_sha256": digest(paths["static"]),
        "init_sha256": digest(paths["init"]),
        **({"orographic_statistics_sha256": digest(orographic_statistics)}
           if orographic_statistics is not None else {}),
        "local_timestep": ({"rates": list(rates), "buffer_rings": int(rings),
                            **({"classing_sha256": digest(classing_path)}
                               if classing_path is not None else {})}
                           if local_timestep else None),
    }


def run_metal_forecast(arguments: Any) -> int:
    """Run the registered public forecast request from an initialized MPAS file."""
    from netCDF4 import Dataset

    from .forecast_door import (
        DRIVERS_DIR, _load_module, _refuse, _schedule, load_registry, resolve_door_physics,
    )
    from .mesh import Mesh

    from .mpasphys.forecast_v842 import native_route
    native_config, native_text = native_route(arguments, default_mode='correct')
    if native_config is not None:
        from .mpasphys.metal_door_v842 import run_forecast as run_native_metal
        return run_native_metal(arguments, native_config, native_text)
    _refuse_unimplemented(arguments, _refuse)
    from .forecast_door import _physics_mode_fields
    from . import numerics
    mode_fields = _physics_mode_fields(arguments)
    if numerics.is_strict():
        raise _refuse("Metal Hex strict numerics has no complete arithmetic contract")
    mesh_name = getattr(arguments, "mesh", None)
    if not mesh_name:
        raise _refuse("--mesh is required for --backend metal; name a registered MPAS mesh row")
    init_raw = getattr(arguments, "init", None)
    if init_raw is None:
        raise _refuse("--init is required for --backend metal; supply a real MPAS init")
    init = Path(init_raw).expanduser().resolve(strict=True)
    init_source = str(getattr(arguments, "init_source", "") or "").strip()
    if not init_source:
        raise _refuse(
            "--init-source is required for --backend metal; the run receipt "
            "must name the meteorological source of the initialization"
        )
    registry = load_registry()
    if mesh_name not in registry:
        raise _refuse(f"--mesh {mesh_name!r} is not registered")
    row = registry[mesh_name]
    physics, _, _ = resolve_door_physics(arguments, mesh_name, row.nominal_dx_m)
    from .physics_backend_admission import DEFAULT_BACKEND
    from .physics_default import arwen_suite
    if (getattr(arguments, "physics_suite", None) == arwen_suite()
            and getattr(arguments, "physics_backend", None) not in (None, DEFAULT_BACKEND)):
        raise _refuse(f"--physics-suite {arwen_suite()} and --physics-backend name different rows")
    if physics is not None:
        arguments.physics_backend = physics.backend_row
    elif getattr(arguments, "physics_backend", None) is None:
        arguments.physics_backend = DEFAULT_BACKEND
    binding_module = _load_module(
        "mpas_mesh_binding", DRIVERS_DIR / "mpas_mesh_binding.py"
    )
    binding = binding_module.MESH_BINDINGS[mesh_name]
    hours = getattr(arguments, "hours", None)
    if hours is None:
        raise _refuse("--hours is required for --backend metal")
    history_minutes = getattr(arguments, "history_every_minutes", None)
    if history_minutes is None:
        raise _refuse("--history-every-minutes is required for --backend metal")
    from . import mpas_defaults
    from .forecast_door import default_step_selectors, resolve_schedule
    from .global_door import default_step_cap

    schedule = resolve_schedule(arguments, None)
    radiation_seconds = getattr(arguments, "radiation_seconds", None)
    if radiation_seconds is None:
        radiation_seconds = mpas_defaults.DEFAULT_RADIATION_SECONDS
    try:
        dt_selection = mpas_defaults.resolve_dt(
            getattr(arguments, "dt", None),
            nominal_dx_m=(None if row.nominal_dx_m is None
                          else float(row.nominal_dx_m)),
            row_dt_seconds=float(row.dt_seconds),
            radiation_seconds=float(radiation_seconds))
        dt_selection = default_step_cap(
            dt_selection, default_step_selectors(physics, arguments),
            radiation_seconds=float(radiation_seconds))
    except ValueError as error:
        raise _refuse(str(error)) from None
    dt_selection["registered_row_dt_seconds"] = float(row.dt_seconds)
    dt_seconds = float(dt_selection["dt_seconds"])
    steps, _ = _schedule(float(hours), float(history_minutes), dt_seconds, mesh_name)
    if _memory_admission(arguments, init, _refuse):
        return 0

    out_raw = getattr(arguments, "out", None)
    if out_raw is None:
        raise _refuse("--out is required for --backend metal")
    out = Path(out_raw).expanduser().absolute()
    if not out.parent.is_dir():
        raise _refuse(f"--out parent directory does not exist: {out.parent}")
    if out.exists():
        raise _refuse(f"--out {out} exists; choose a fresh output directory")
    receipt_raw = getattr(arguments, "receipt", None)
    receipt_path = (
        Path(receipt_raw).expanduser().absolute() if receipt_raw is not None
        else out / "forecast-receipt.json"
    )
    if not receipt_path.parent.is_dir() and receipt_path.parent != out:
        raise _refuse(f"--receipt parent directory does not exist: {receipt_path.parent}")
    with Dataset(init, mode="r") as dataset:
        valid_time = _initial_time(dataset)
    start_time = getattr(arguments, "start_time", None)
    if start_time is not None and start_time != _start_stamp(valid_time):
        raise _refuse(
            f"--start-time {start_time} does not match the init's "
            f"{_start_stamp(valid_time)}"
        )

    # The init is a self-contained MPAS state file: it carries topology,
    # geography, static fields, and the native vertical metric contract.
    mesh = Mesh.from_netcdf(init, validate=True)
    actual_cells = int(mesh.dimensions.get("nCells", -1))
    actual_edges = int(mesh.dimensions.get("nEdges", -1))
    if (actual_cells != int(binding.n_cells)
            or actual_edges != int(binding.n_edges)):
        raise _refuse(
            f"--init has {actual_cells} cells and {actual_edges} edges, but "
            f"mesh row {mesh_name} declares {binding.n_cells} cells and "
            f"{binding.n_edges} edges"
        )
    horiz_mixing = str(getattr(arguments, "horiz_mixing", "2d_smagorinsky"))
    checkpoint = getattr(arguments, "checkpoint", None)
    if checkpoint is not None:
        checkpoint = Path(checkpoint).expanduser().absolute()
        if not checkpoint.parent.is_dir():
            raise _refuse(f"--checkpoint parent directory does not exist: {checkpoint.parent}")
    regional = bool(binding.regional)
    lbc_dir = getattr(arguments, "lbc_dir", None)
    if regional and lbc_dir is None:
        raise _refuse(
            f"mesh row {mesh_name} is limited-area: --lbc-dir is required, or "
            "the domain empties from its boundary rings inward")
    if not regional and lbc_dir is not None:
        raise _refuse(f"--lbc-dir is for limited-area rows; {mesh_name} is global")
    lbc_paths: list[str] | None = None
    if regional:
        lbc_paths = sorted(str(path) for path in Path(lbc_dir).glob("lbc.*.nc"))
        if not lbc_paths:
            raise _refuse(f"--lbc-dir {lbc_dir} holds no lbc.*.nc")
    return _run_physics(
        arguments, binding_module=binding_module, mesh_name=mesh_name,
        row=row, init=init, init_source=init_source,
        valid_time=valid_time, hours=float(hours),
        history_minutes=float(history_minutes), steps=steps, out=out,
        dt_selection=dt_selection, schedule=schedule,
        radiation_seconds=float(radiation_seconds),
        receipt_path=receipt_path, checkpoint=checkpoint,
        horiz_mixing=horiz_mixing, lbc_paths=lbc_paths, physics=physics,
        mode_fields=mode_fields)

def _memory_admission(arguments: Any, init: Path, refuse: Any) -> bool:
    """Refuse a mesh the device cannot hold; True when only a preflight."""
    import json

    from netCDF4 import Dataset

    from woof.metal.memory_gate import FRACTION_ENV, estimate_hex
    from woof.metal.runtime import MetalContext

    with Dataset(init, mode="r") as dataset:
        n_cells = len(dataset.dimensions["nCells"])
        n_levels = len(dataset.dimensions["nVertLevels"])
    from .drivers.run_cuda_v841_forecast import resolve_run_row
    from .physics_backend_admission import DEFAULT_BACKEND
    backend_row, _, _ = resolve_run_row(
        getattr(arguments, "physics_backend", None) or DEFAULT_BACKEND,
        getattr(arguments, "source_table", None))
    adapter = backend_row.load_metal_adapter()
    extra_bytes = adapter.additional_memory_bytes(backend_row, n_cells, n_levels)
    context = MetalContext()
    try:
        estimate = estimate_hex(n_cells, n_levels, context,
                                additional_bytes=extra_bytes)
    finally:
        context.close()
    if getattr(arguments, "preflight", False):
        print(json.dumps({
            "backend": "metal", "physics_backend": backend_row.name,
            "cells": n_cells, "levels": n_levels,
            "estimated_peak_bytes": estimate.peak_bytes,
            "limit_bytes": estimate.limit_bytes,
            "fits_device": estimate.fits_device}))
        if not estimate.fits_device:
            raise refuse("the estimated Metal memory exceeds the device budget")
        return True
    if not estimate.fits_device:
        raise refuse(
            f"a {n_cells}-cell, {n_levels}-level Hex physics forecast needs an "
            f"estimated {estimate.peak_bytes / 2**30:.1f} GiB; the device budget "
            f"is {estimate.limit_bytes / 2**30:.1f} GiB ({estimate.fraction:.0%} "
            f"of its {estimate.working_set_bytes / 2**30:.1f} GiB recommended "
            f"working set). Use a coarser mesh or override with {FRACTION_ENV}.")
    return False


def _run_physics(
    arguments: Any, *, binding_module: Any, mesh_name: str, row: Any,
    init: Path, init_source: str, valid_time: datetime, hours: float,
    history_minutes: float, steps: int, out: Path, receipt_path: Path,
    dt_selection: dict[str, Any], schedule: dict[str, Any],
    radiation_seconds: float,
    checkpoint: Path | None, horiz_mixing: str, lbc_paths: list[str] | None,
    physics: Any, mode_fields: dict[str, Any],
) -> int:
    """The full physics row, as the CUDA forecast runs it."""

    import importlib

    from .forecast_door import _parse_rates, _refuse
    from .metal_driver_lts import (
        attach_local_timestep_metal, detach_local_timestep_metal,
    )
    from .metal_arwen_physics_v841 import (
        capture_snapshot_metal_v841,
        execute_composite_step_metal_v841,
        open_metal_physics_stack_v841,
        step_health_gate_metal_v841,
    )
    from woof.metal.runtime import MetalContext

    from .checkpoint import read_checkpoint, read_checkpoint_physics, write_checkpoint
    from . import metal_v841_launch
    from .metal_driver_v841 import MetalGfDynamicsTendenciesV841

    dt_seconds = float(dt_selection["dt_seconds"])
    stride = int(round(history_minutes * 60 / dt_seconds))
    restart = getattr(arguments, "restart", None)
    restored = restored_saved = restored_physics = None
    first_step = 0
    if restart is not None:
        restored, restored_saved = read_checkpoint(restart)
        restored_physics = read_checkpoint_physics(restart)
        if restored_saved is None or restored_physics is None:
            raise _refuse(
                f"--restart {restart} carries no physics state; a physics run "
                "restarts only from a physics checkpoint")
        if "setup" not in restored_physics:
            raise _refuse(
                f"--restart {restart} records no setup binding, so its grid, "
                "static and configuration cannot be checked; write the "
                "checkpoint again with --checkpoint")
        elapsed = float(restored.time_seconds)
        first_step = int(round(elapsed / dt_seconds))
        if first_step * dt_seconds != elapsed:
            raise _refuse(
                f"--restart {restart} time {elapsed} s is not a whole number "
                f"of {dt_seconds:g} s steps")
        if first_step % stride:
            raise _refuse(
                f"--restart {restart} is at step {first_step}, which is not a "
                f"multiple of the {stride}-step history stride; history frames "
                "fall on an absolute step grid, so restart from a checkpoint "
                "taken at a history time")
    paths = {"init": init}
    for flag, dest in (("--grid", "grid"), ("--static", "static")):
        value = getattr(arguments, dest, None)
        if value is None:
            raise _refuse(
                f"{flag} is required for the physics row on --backend metal: "
                "the host preparation reads the MPAS grid and static files")
        paths[dest] = Path(value).expanduser().resolve(strict=True)
    runner = importlib.import_module("woof.hex.drivers.run_cuda_v841_forecast")
    convection = (str(getattr(arguments, "convection", "auto") or "auto")
                  if physics is None else physics.convection)
    # The preset resolves into the surface/PBL cadence, as on CUDA; an
    # explicit --pbl-cadence wins.
    from . import forecast_preset
    from .errors import ConfigurationRefusal
    try:
        preset = forecast_preset.resolve_preset(getattr(arguments, "preset", None))
    except ConfigurationRefusal as error:
        raise _refuse(str(error)) from None
    pbl_cadence = getattr(arguments, "pbl_cadence", None)
    if pbl_cadence is None:
        pbl_cadence = forecast_preset.pbl_cadence_request(preset, dt_seconds)
    try:
        binding_module.bind_mesh(
            runner.proof, mesh_name, grid=paths["grid"], static=paths["static"],
            forecast=runner, convection=convection, pbl_cadence=pbl_cadence,
            dt_seconds=dt_seconds, radiation_seconds=radiation_seconds,
            time_integration_order=schedule["order"],
            acoustic_substeps=schedule["substeps"])
    except binding_module.MeshBindingError as error:
        raise _refuse(f"the mesh bind refused: {error}") from None
    local_timestep = bool(getattr(arguments, "local_timestep", False))
    rates = _parse_rates(getattr(arguments, "local_timestep_rates", "1,3"),
                         "--local-timestep-rates")
    rings = int(getattr(arguments, "local_timestep_buffer_rings", 1))
    if rings < 1:
        raise _refuse("--local-timestep-buffer-rings must be at least 1")
    classing_raw = getattr(arguments, "local_timestep_classing", None)
    classing_path = (None if classing_raw is None
                     else Path(classing_raw).expanduser().absolute())
    if classing_path is not None:
        if not local_timestep:
            raise _refuse("--local-timestep-classing needs --local-timestep")
        if not classing_path.is_file():
            raise _refuse(f"--local-timestep-classing {classing_path} is not a file")
    authority = runner.verify_forecast_authorities(paths)
    # Taken once by the authority check; reused, not re-hashed.
    known_digests = {role: record.get("sha256")
                     for role, record in authority["files"].items()}
    oro_arg = getattr(arguments, "orographic_statistics", None)
    oro_path = None if oro_arg is None else Path(oro_arg).expanduser().resolve(strict=True)
    host = runner.prepare_forecast_host(
        paths, authority, start_time_text=getattr(arguments, "start_time", None),
        horiz_mixing=horiz_mixing, convection=convection,
        pbl_cadence=pbl_cadence or "auto", local_timestep=local_timestep,
        local_timestep_rates=rates, local_timestep_buffer_rings=rings,
        lbc_paths=lbc_paths,
        physics_backend=getattr(arguments, "physics_backend", "wsm6_column"),
        source_table=getattr(arguments, "source_table", None),
        orographic_statistics=oro_path, physics=physics,
        **mode_fields, radiation_seconds=radiation_seconds,
        time_integration_order=schedule["order"],
        acoustic_substeps=schedule["substeps"],
        monotonic=schedule["monotonic"])
    # The shared CUDA preparer omits exact-mode kwargs for its frozen ABI.
    # Metal must still resolve its RunConfig independently of new defaults.
    host["constructor_values"].update(mode_fields)
    # Hashing the grid, static and init files is only needed to bind a
    # checkpoint or check a restart against one.
    setup = (_restart_setup(host, paths, local_timestep=local_timestep,
                            rates=rates, rings=rings, known=known_digests,
                            classing_path=classing_path,
                            orographic_statistics=oro_path)
             if restored_physics is not None or checkpoint is not None else None)
    if restored_physics is not None and restored_physics["setup"] != setup:
        raise _refuse(
            "--restart configuration or static carriers differ from the "
            "checkpoint")
    static = runner.proof._static_output_fields(host)
    frames = []
    health = []
    with (MetalContext() as context, metal_v841_launch.preserve_post_launch(),
          ExitStack() as cleanup):
        # Without --stop-on-refusal a failed step ends this run, so the seam
        # keeps no rollback copy; with it, a refused step rolls back to the
        # last committed boundary, as on CUDA.
        stop_on_refusal = bool(getattr(arguments, "stop_on_refusal", False))
        stack = open_metal_physics_stack_v841(
            context, host, state=restored, saved_diagnostics=restored_saved,
            rollback=stop_on_refusal)
        driver, backend = stack["driver"], stack["backend"]
        scalar_names = stack["scalar_names"]
        # None, rebinding nothing, unless the config asks for local time stepping.
        classing = None
        if classing_path is not None:
            from .lts_v841 import classify_from_grid_file_with_rates
            classing = classify_from_grid_file_with_rates(
                str(paths["grid"]), str(classing_path), rates=rates,
                buffer_rings=rings)
        lts = attach_local_timestep_metal(
            driver, grid_path=str(paths["grid"]), classing=classing)
        # The attachment rebinds module-level dycore entry points; put them
        # back when the run ends, however it ends.
        cleanup.callback(detach_local_timestep_metal, driver)
        if restored_physics is None:
            previous = backend.accumulated_precipitation()
            forcing = None
        else:
            backend.restore_restart_state(restored_physics["backend"])
            if float(backend.seam.elapsed_seconds) != float(restored.time_seconds):
                raise _refuse(
                    f"--restart {restart}: the physics clock "
                    f"{backend.seam.elapsed_seconds} s differs from the "
                    f"dycore's {restored.time_seconds} s")
            upload = driver.arrays.asarray
            previous = {name: upload(np.ascontiguousarray(value))
                        for name, value in restored_physics["previous_surface"].items()}
            carried = restored_physics["gf_forcing"]
            forcing = None if carried is None else MetalGfDynamicsTendenciesV841(
                rthdynten=upload(carried["rthdynten"]),
                rqvdynten=upload(carried["rqvdynten"]),
                time_seconds=float(carried["time_seconds"]))
        out.mkdir()
        if getattr(arguments, 'wind_farms', None) not in (None, '', 'off'):
            from .wind_farm import attach_arwen
            host['wind_farms'] = arguments.wind_farms
            attach_arwen(backend, host, driver, output_root=out)
        if restored_physics is None:
            frames.append(_write_f000_frame(
                runner, host, stack, static, out,
                valid_time=valid_time))
        def capture(step: int, *, expect_refl10cm: bool = True) -> None:
            snapshot = capture_snapshot_metal_v841(
                driver=driver, backend=backend,
                prep_geometry=stack["prep_geometry"],
                scalar_names=scalar_names, step=step,
                expect_refl10cm=expect_refl10cm,
                solve_cells=stack["solve_cells"], solve_edges=stack["solve_edges"])
            path = out / f"history.{step:06d}.nc"
            _write_metal_snapshot(
                runner, path, snapshot, static,
                valid_time=valid_time + timedelta(seconds=step * dt_seconds),
                physics_identity=backend.physics_identity,
                wind_farms=getattr(backend, 'wind_farms', None))
            frames.append({"step": step, "file": path.name,
                           "valid_time": (valid_time + timedelta(
                               seconds=step * dt_seconds)).isoformat()})

        def refused(error: Exception, step: int, where: str,
                    last_committed: int) -> dict[str, Any]:
            # Recorded verbatim; no guard is relaxed.
            return {"refused": True, "refused_in": where, "step": step,
                    "model_seconds": step * dt_seconds,
                    "exception": type(error).__name__, "message": str(error),
                    "last_committed_step": last_committed}

        refusal = None
        last_step = first_step + steps
        for step in range(first_step + 1, first_step + steps + 1):
            due = step % stride == 0
            try:
                result = execute_composite_step_metal_v841(
                    driver=driver, backend=backend, scalar_names=scalar_names,
                    physics_geometry=stack["physics_geometry"],
                    previous_surface_updates=previous,
                    dynamics_tendencies=forcing, refl_10cm_due=due)
            except Exception as error:
                # Only a step that rolled back to its boundary is a refusal.
                if not stop_on_refusal or backend.phase != "boundary":
                    raise
                refusal = refused(error, step, "composite step", step - 1)
                last_step = step - 1
                break
            previous, forcing = result.surface_updates, result.dynamics_tendencies
            try:
                health.append(step_health_gate_metal_v841(stack, step))
            except FloatingPointError as error:
                if not stop_on_refusal:
                    raise
                refusal = refused(error, step, "step health gate", step)
                last_step = step
                break
            if due:
                try:
                    capture(step)
                except (FloatingPointError, RuntimeError, ValueError) as error:
                    if not stop_on_refusal:
                        raise
                    refusal = refused(error, step, "history capture", step)
                    last_step = step
                    break
        if (refusal is not None and last_step > first_step
                and all(frame["step"] != last_step for frame in frames)):
            # Keep the last committed state renderable when it can be.
            try:
                capture(last_step, expect_refl10cm=False)
            except Exception as error:  # noqa: BLE001 - never fatal here
                refusal["final_state_capture_failed"] = True
                refusal["final_state_capture_error"] = str(error)
        n_cells, n_edges = int(row.cells), int(static["latEdge"].size)
        final = _solve_region(driver.atmosphere.state.to_host(), n_cells, n_edges)
        if checkpoint is not None and refusal is not None:
            refusal["checkpoint_written"] = False
        if checkpoint is not None and refusal is None:
            write_checkpoint(
                checkpoint, final,
                saved_diagnostics=_solve_region(
                    driver.atmosphere.saved.to_host(), n_cells, n_edges),
                physics={
                    "setup": setup,
                    "backend": backend.restart_state(),
                    "gf_forcing": None if forcing is None else {
                        "rthdynten": forcing.rthdynten.to_numpy(),
                        "rqvdynten": forcing.rqvdynten.to_numpy(),
                        "time_seconds": float(forcing.time_seconds)},
                    "previous_surface": {name: value.to_numpy()
                                         for name, value in previous.items()},
                })
    if refusal is None:
        final.validate(n_cells=int(row.cells), n_edges=int(final.rho_u.shape[1]),
                       n_vert_levels=int(final.rho.shape[0]))
    receipt_path.write_text(json.dumps({
        "schema": "woof.hex.metal-forecast/v1",
        "backend": "metal",
        "status": "passed" if refusal is None else "truncated_by_model_refusal",
        "refusal": refusal,
        "case_label": getattr(arguments, "case_label", None),
        "physics": {
            "physics_mode": mode_fields["physics_mode"],
            "correct_mode": backend.correct_mode_receipt,
            "noahmp_statics": backend.noahmp_statics_receipt,
            "solar_generation": backend.physics_identity["source_generations"]["solar"],
            "ysu_version": backend.physics_identity["source_generations"]["ysu"],
            "noahmp_snow_age": backend.physics_identity["source_generations"]["noahmp_snow_age"],
            "configuration_sha256": backend.physics_identity["configuration_sha256"],
            "source_generations": backend.physics_identity["source_generations"],
            "numerics": "metal_ordinary",
            "row": host.get("physics_backend"),
            "resolution": host.get("physics_resolution"),
            "suite": host.get("physics_suite"),
            "orographic_statistics": host["constructor_receipt"].get("orographic_statistics"),
            "land_surface": getattr(backend, "land_surface_receipt", None),
            "scalars": list(scalar_names),
            "convection": host.get("convection"),
            "pbl_cadence": host.get("pbl_cadence"),
        },
        "preset": preset.receipt(),
        "pbl_cadence_request": pbl_cadence,
        "horiz_mixing": horiz_mixing,
        "local_timestep": None if not local_timestep else {
            "rates": list(rates), "buffer_rings": rings,
            "attached": lts is not None,
            "execution": None if lts is None else lts.receipt()},
        "limited_area": lbc_paths is not None,
        "lbc_intervals": None if lbc_paths is None else len(lbc_paths),
        "mesh": mesh_name,
        "grid": str(paths["grid"]),
        "static": str(paths["static"]),
        "init": str(init),
        "init_source": init_source,
        "initial_time": valid_time.isoformat(),
        "hours": hours,
        "dt_seconds": dt_seconds,
        "dt_selection": dt_selection,
        "schedule": schedule,
        "radiation_seconds": radiation_seconds,
        "steps": last_step - first_step,
        "first_step": first_step,
        "restart": None if restart is None else str(Path(restart).resolve()),
        "restart_setup_verified": restart is not None,
        "checkpoint": (str(checkpoint)
                       if checkpoint is not None and refusal is None else None),
        "history_every_minutes": history_minutes,
        "history": frames,
        # The CUDA door's frame index, so the swath and cascade readers
        # take a Metal run as they take a CUDA one.
        "forecast": _forecast_index(out, frames, paths, known_digests),
        "step_health": health,
        "finite": refusal is None,
        "state": final.bounds_receipt() if refusal is None else None,
    }, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "backend": "metal", "mesh": mesh_name, "steps": last_step - first_step,
        "hours": hours, "physics": True, "finite": refusal is None,
        "refused": refusal is not None,
        **({"checkpoint_written": False}
           if checkpoint is not None and refusal is not None else {}),
        "out": str(out), "receipt": str(receipt_path),
    }, sort_keys=True))
    if frames:
        print("NEXT " + " ".join([
            "woof hex", "render", "--history",
            *(str(out / frame["file"]) for frame in frames),
            "--mesh", str(paths["grid"]), "--out", str(out / "png"),
            "--simulation-start", _start_stamp(valid_time)]), flush=True)
    return 0


def _forecast_index(out: Path, frames: list[dict[str, Any]],
                    paths: dict[str, Path],
                    known: dict[str, str] | None = None,
                   classing_path: Path | None = None) -> dict[str, Any]:
    """Frame files with their SHA-256s, valid-time labels and authorities.

    ``known`` holds the authority digests the run already took, by role.
    """
    from hashlib import sha256

    def digest(path: Path) -> str:
        result = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(chunk)
        return result.hexdigest()

    return {
        "snapshot_files": {
            str(frame["step"]): {"path": str(out / frame["file"]),
                                 "sha256": digest(out / frame["file"])}
            for frame in frames},
        "history_labels": {
            str(frame["step"]): datetime.fromisoformat(
                frame["valid_time"]).strftime("%Y-%m-%d_%H.%M.%S")
            for frame in frames},
        "authority": {"files": {
            name: {"path": str(paths[name]),
                   "sha256": (known or {}).get(name) or digest(paths[name])}
            for name in ("grid", "static")}},
    }


def _write_f000_frame(runner: Any, host: Any, stack: Any, static: Any,
                      out: Path, *, valid_time: datetime) -> dict[str, Any]:
    """The start frame, completed as the CUDA forecast completes it.

    No physics has run, so the frame carries the allocated zero
    reflectivity, the init's own surface diagnostics in place of the
    seam's placeholders, and the start file's 2 m humidity and the model's
    surface pressure.
    """

    from .metal_arwen_physics_v841 import capture_snapshot_metal_v841

    snapshot = capture_snapshot_metal_v841(
        driver=stack["driver"], backend=stack["backend"],
        prep_geometry=stack["prep_geometry"], scalar_names=stack["scalar_names"],
        step=0, expect_refl10cm=False,
        solve_cells=stack["solve_cells"], solve_edges=stack["solve_edges"])
    arrays = snapshot["arrays"]
    arrays["refl10cm"] = np.zeros_like(arrays["theta"])
    receipt = snapshot["receipt"]
    receipt["arwen_v2_surface_execution"] = {"last_noahmp_census": None}
    receipt["f000_initialized_surface_diagnostics"] = (
        runner.proof.overlay_f000_initialized_surface_diagnostics(
            arrays, host["f000_surface_diagnostics"]["arrays"]))
    receipt["f000_completed_surface_diagnostics"] = (
        runner.complete_f000_surface_diagnostics(
            snapshot, host["f000_start_humidity"]))
    path = out / "history.000000.nc"
    _write_metal_snapshot(runner, path, snapshot, static, valid_time=valid_time,
                          physics_identity=stack["backend"].physics_identity,
                          wind_farms=getattr(stack["backend"], "wind_farms", None))
    return {"step": 0, "file": path.name, "valid_time": valid_time.isoformat()}
