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
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


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
    "mesh", "backend", "init", "init_source", "start_time", "hours",
    "history_every_minutes", "out", "receipt", "case_label", "repo",
    "horiz_mixing", "checkpoint", "restart", "lbc_dir",
    "grid", "static", "convection", "pbl_cadence",
    "local_timestep", "local_timestep_rates", "local_timestep_buffer_rings",
})


def _refuse_unimplemented(arguments: Any, refuse: Any) -> None:
    from .forecast_door import add_forecast_arguments

    parser = argparse.ArgumentParser(add_help=False)
    add_forecast_arguments(parser)
    for action in parser._actions:
        dest = action.dest
        if dest in _IMPLEMENTED or not hasattr(arguments, dest):
            continue
        if getattr(arguments, dest) != action.default:
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


def run_metal_forecast(arguments: Any) -> int:
    """Run the registered public forecast request from an initialized MPAS file."""
    from netCDF4 import Dataset

    from .forecast_door import (
        DRIVERS_DIR, _load_module, _refuse, _schedule, load_registry,
    )
    from .mesh import Mesh

    _refuse_unimplemented(arguments, _refuse)
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
    steps, _ = _schedule(float(hours), int(history_minutes), row.dt_seconds, mesh_name)

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
    out.mkdir()

    # The init is a self-contained MPAS state file: it carries topology,
    # geography, static fields, and the native vertical metric contract.
    mesh = Mesh.from_netcdf(init, validate=True)
    actual_cells = int(mesh.dimensions.get("nCells", -1))
    actual_edges = int(mesh.dimensions.get("nEdges", -1))
    if (actual_cells != int(binding.n_cells)
            or actual_edges != int(binding.n_edges)):
        raise _refuse(
            f"--init has {actual_cells} cells and {actual_edges} edges, but "
            f"mesh row {mesh_name} declares {row.cells} cells and "
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
        if getattr(arguments, "local_timestep", False):
            raise _refuse(
                "--local-timestep has no limited-area acoustic entrypoints on "
                "--backend metal; run it on a global row")
        lbc_paths = sorted(str(path) for path in Path(lbc_dir).glob("lbc.*.nc"))
        if not lbc_paths:
            raise _refuse(f"--lbc-dir {lbc_dir} holds no lbc.*.nc")
    return _run_physics(
        arguments, binding_module=binding_module, mesh_name=mesh_name,
        row=row, init=init, init_source=init_source,
        valid_time=valid_time, hours=float(hours),
        history_minutes=int(history_minutes), steps=steps, out=out,
        receipt_path=receipt_path, checkpoint=checkpoint,
        horiz_mixing=horiz_mixing, lbc_paths=lbc_paths)

def _run_physics(
    arguments: Any, *, binding_module: Any, mesh_name: str, row: Any,
    init: Path, init_source: str, valid_time: datetime, hours: float,
    history_minutes: int, steps: int, out: Path, receipt_path: Path,
    checkpoint: Path | None, horiz_mixing: str, lbc_paths: list[str] | None,
) -> int:
    """The full physics row, as the CUDA forecast runs it."""

    import importlib

    from .forecast_door import _parse_rates, _refuse
    from .metal_driver_lts import attach_local_timestep_metal
    from .metal_arwen_physics_v841 import (
        capture_snapshot_metal_v841,
        execute_composite_step_metal_v841,
        open_metal_physics_stack_v841,
    )
    from woof.metal.runtime import MetalContext

    from .checkpoint import read_checkpoint, read_checkpoint_physics, write_checkpoint
    from .metal_driver_v841 import MetalGfDynamicsTendenciesV841

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
        elapsed = float(restored.time_seconds)
        first_step = int(round(elapsed / float(row.dt_seconds)))
        if first_step * float(row.dt_seconds) != elapsed:
            raise _refuse(
                f"--restart {restart} time {elapsed} s is not a whole number "
                f"of {row.dt_seconds:.0f} s steps")
    paths = {"init": init}
    for flag, dest in (("--grid", "grid"), ("--static", "static")):
        value = getattr(arguments, dest, None)
        if value is None:
            raise _refuse(
                f"{flag} is required for the physics row on --backend metal: "
                "the host preparation reads the MPAS grid and static files")
        paths[dest] = Path(value).expanduser().resolve(strict=True)
    runner = importlib.import_module("woof.hex.drivers.run_cuda_v841_forecast")
    convection = str(getattr(arguments, "convection", "auto") or "auto")
    pbl_cadence = getattr(arguments, "pbl_cadence", None)
    try:
        binding_module.bind_mesh(
            runner.proof, mesh_name, grid=paths["grid"], static=paths["static"],
            forecast=runner, convection=convection, pbl_cadence=pbl_cadence)
    except binding_module.MeshBindingError as error:
        raise _refuse(f"the mesh bind refused: {error}") from None
    local_timestep = bool(getattr(arguments, "local_timestep", False))
    rates = _parse_rates(getattr(arguments, "local_timestep_rates", "1,3"),
                         "--local-timestep-rates")
    rings = int(getattr(arguments, "local_timestep_buffer_rings", 1))
    if rings < 1:
        raise _refuse("--local-timestep-buffer-rings must be at least 1")
    authority = runner.verify_forecast_authorities(paths)
    host = runner.prepare_forecast_host(
        paths, authority, start_time_text=getattr(arguments, "start_time", None),
        horiz_mixing=horiz_mixing, convection=convection,
        pbl_cadence=pbl_cadence or "auto", local_timestep=local_timestep,
        local_timestep_rates=rates, local_timestep_buffer_rings=rings,
        lbc_paths=lbc_paths)
    static = runner.proof._static_output_fields(host)
    stride = history_minutes * 60 // int(row.dt_seconds)
    frames = []
    with MetalContext() as context:
        stack = open_metal_physics_stack_v841(
            context, host, state=restored, saved_diagnostics=restored_saved)
        driver, backend = stack["driver"], stack["backend"]
        scalar_names = stack["scalar_names"]
        # None, rebinding nothing, unless the config asks for local time stepping.
        lts = attach_local_timestep_metal(driver, grid_path=str(paths["grid"]))
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
        if restored_physics is None:
            frames.append(_write_f000_frame(
                runner, host, stack, static, out,
                valid_time=valid_time))
        for step in range(first_step + 1, first_step + steps + 1):
            due = step % stride == 0
            result = execute_composite_step_metal_v841(
                driver=driver, backend=backend, scalar_names=scalar_names,
                physics_geometry=stack["physics_geometry"],
                previous_surface_updates=previous,
                dynamics_tendencies=forcing, refl_10cm_due=due)
            previous, forcing = result.surface_updates, result.dynamics_tendencies
            if due:
                snapshot = capture_snapshot_metal_v841(
                    driver=driver, backend=backend,
                    prep_geometry=stack["prep_geometry"],
                    scalar_names=scalar_names, step=step, expect_refl10cm=True,
                    solve_cells=stack["solve_cells"], solve_edges=stack["solve_edges"])
                path = out / f"history.{step:06d}.nc"
                runner.proof.write_snapshot_netcdf(
                    path, snapshot, static,
                    source="MPAS-A v8.4.1 full-physics Metal port")
                frames.append({"step": step, "file": path.name,
                               "valid_time": (valid_time + timedelta(
                                   seconds=step * float(row.dt_seconds))).isoformat()})
        n_cells, n_edges = int(row.cells), int(static["latEdge"].size)
        final = _solve_region(driver.atmosphere.state.to_host(), n_cells, n_edges)
        if checkpoint is not None:
            write_checkpoint(
                checkpoint, final,
                saved_diagnostics=_solve_region(
                    driver.atmosphere.saved.to_host(), n_cells, n_edges),
                physics={
                    "backend": backend.restart_state(),
                    "gf_forcing": None if forcing is None else {
                        "rthdynten": forcing.rthdynten.to_numpy(),
                        "rqvdynten": forcing.rqvdynten.to_numpy(),
                        "time_seconds": float(forcing.time_seconds)},
                    "previous_surface": {name: value.to_numpy()
                                         for name, value in previous.items()},
                })
        if lbc_paths is not None:
            from . import metal_v841_launch

            metal_v841_launch.POST_LAUNCH = None
    final.validate(n_cells=int(row.cells), n_edges=int(final.rho_u.shape[1]),
                   n_vert_levels=int(final.rho.shape[0]))
    receipt_path.write_text(json.dumps({
        "schema": "woof.hex.metal-forecast/v1",
        "backend": "metal",
        "case_label": getattr(arguments, "case_label", None),
        "physics": {
            "row": host.get("physics_backend"),
            "scalars": list(scalar_names),
            "convection": host.get("convection"),
            "pbl_cadence": host.get("pbl_cadence"),
        },
        "horiz_mixing": horiz_mixing,
        "local_timestep": None if not local_timestep else {
            "rates": list(rates), "buffer_rings": rings,
            "attached": lts is not None},
        "limited_area": lbc_paths is not None,
        "lbc_intervals": None if lbc_paths is None else len(lbc_paths),
        "mesh": mesh_name,
        "grid": str(paths["grid"]),
        "static": str(paths["static"]),
        "init": str(init),
        "init_source": init_source,
        "initial_time": valid_time.isoformat(),
        "hours": hours,
        "dt_seconds": float(row.dt_seconds),
        "steps": steps,
        "first_step": first_step,
        "restart": None if restart is None else str(Path(restart).resolve()),
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "history_every_minutes": history_minutes,
        "history": frames,
        "finite": True,
        "state": final.bounds_receipt(),
    }, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "backend": "metal", "mesh": mesh_name, "steps": steps,
        "hours": hours, "physics": True, "finite": True,
        "out": str(out), "receipt": str(receipt_path),
    }, sort_keys=True))
    return 0


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
    runner.proof.write_snapshot_netcdf(
        path, snapshot, static, source="MPAS-A v8.4.1 full-physics Metal port")
    return {"step": 0, "file": path.name, "valid_time": valid_time.isoformat()}
