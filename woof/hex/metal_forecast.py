"""Run the public Hex forecast door on the Metal-enabled dry dycore.

This route consumes a real MPAS init directly.  Init files carry the grid,
static fields, and vertical metrics needed by the current dycore, so no CUDA
driver or separate mesh asset lookup is needed here.

The route runs the resident v8.4.1 dycore (``metal_driver_v841``) without
physics, with or without horizontal mixing.  Every forecast flag it does not
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
    "horiz_mixing",
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


# The v8.4.1 lane's mixing settings (config_v841.py).
_MIXING = {
    "off": {"config_horiz_mixing": "off"},
    "2d_smagorinsky": {
        "config_horiz_mixing": "2d_smagorinsky",
        "config_len_disp": 0.0,
        "config_visc4_2dsmag": 0.05,
        "config_smagorinsky_coef": 0.125,
        "config_del4u_div_factor": 10.0,
    },
}


def _start_stamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d_%H:%M:%S")


def run_metal_forecast(arguments: Any) -> int:
    """Run the registered public forecast request from an initialized MPAS file."""
    from netCDF4 import Dataset

    from .driver import (
        load_mpas_initial_state,
        load_mpas_vertical_grid,
    )
    from .forecast_door import (
        DRIVERS_DIR, _load_module, _refuse, _schedule, load_registry,
    )
    from .mesh import Mesh
    from .metal_driver_v841 import MetalDycoreV841
    from .output import write_dry_state_history
    from .config_v841 import V841DryDycoreConfig
    from woof.metal.runtime import MetalContext

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
    native = load_mpas_vertical_grid(init, mesh)
    state, reference, saved = load_mpas_initial_state(
        init, mesh, native.vertical_grid, scalar_names=("qv",),
        terrain_metrics=native.terrain_metrics, return_saved_diagnostics=True,
    )
    # The dycore of the CUDA v8.4.1 lane (config_v841.py), without physics.
    config = V841DryDycoreConfig(
        config_dt=float(row.dt_seconds),
        config_time_integration_order=3,
        config_number_of_sub_steps=6,
        config_dynamics_split_steps=3,
        config_split_dynamics_transport=True,
        config_scalar_advection=True,
        config_monotonic=False,
        config_positive_definite=False,
        config_apvm_upwinding=0.5,
        config_xnutr=0.2,
        config_zd=22_000.0,
        config_smdiv=0.1,
        config_divergence_damping=True,
        config_terrain_following=True,
        config_scalar_adv_order=3,
        config_scalar_vadv_order=3,
        # Dry: qv is a passive tracer, as in the CUDA dry step.
        config_moist_physics=False,
        config_coef_3rd_order=0.25,
        **_MIXING[horiz_mixing],
    )
    with Dataset(init, mode="r") as dataset:
        wind_u = np.asarray(dataset.variables["u_init"][:], dtype=np.float32)
        wind_v = np.asarray(dataset.variables["v_init"][:], dtype=np.float32)
    from .dynamics_v841 import V841ReferenceWindProfiles

    wind = V841ReferenceWindProfiles(wind_u, wind_v)
    stride = int(history_minutes) * 60 // int(row.dt_seconds)
    with MetalContext() as context:
        driver = MetalDycoreV841.from_host(
            context, mesh, state, native.vertical_grid, reference, config,
            saved_diagnostics=saved, terrain_metrics=native.terrain_metrics,
            reference_wind_profiles=wind,
        )
        for step in range(1, steps + 1):
            driver.step_device()
            if step % stride == 0:
                state = driver.atmosphere.state.to_host()
                frame_time = valid_time + timedelta(seconds=step * row.dt_seconds)
                write_dry_state_history(
                    out / f"history.{step:06d}.nc", mesh, state, frame_time,
                    initial_time=valid_time, vertical_grid=native.vertical_grid,
                )
        state = driver.atmosphere.state.to_host()
        state.validate(
            n_cells=int(mesh.dimensions["nCells"]),
            n_edges=int(mesh.dimensions["nEdges"]),
            n_vert_levels=int(mesh.dimensions["nVertLevels"]),
        )

    receipt_path.write_text(json.dumps({
        "schema": "woof.hex.metal-forecast/v1",
        "backend": "metal",
        "case_label": getattr(arguments, "case_label", None),
        "physics": "none",
        "horiz_mixing": horiz_mixing,
        "mesh": mesh_name,
        "init": str(init),
        "init_source": init_source,
        "initial_time": valid_time.isoformat(),
        "hours": float(hours),
        "dt_seconds": float(row.dt_seconds),
        "steps": steps,
        "history_every_minutes": int(history_minutes),
        "finite": True,
        "state": state.bounds_receipt(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "backend": "metal", "mesh": mesh_name, "steps": steps,
        "hours": float(hours), "finite": True,
        "out": str(out), "receipt": str(receipt_path),
    }, sort_keys=True))
    return 0


__all__ = ["run_metal_forecast"]
