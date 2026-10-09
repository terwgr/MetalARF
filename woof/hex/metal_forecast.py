"""Run the public Hex forecast door on the Metal-enabled dry dycore.

This route consumes a real MPAS init directly.  Init files carry the grid,
static fields, and vertical metrics needed by the current dycore, so no CUDA
driver or separate mesh asset lookup is needed here.
"""

from __future__ import annotations

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


def run_metal_forecast(arguments: Any) -> int:
    """Run the registered public forecast request from an initialized MPAS file."""
    from netCDF4 import Dataset

    from .driver import (
        DryDycoreDriver,
        load_mpas_initial_state,
        load_mpas_vertical_grid,
    )
    from .forecast_door import (
        DRIVERS_DIR, _load_module, _refuse, _schedule, load_registry,
    )
    from .mesh import Mesh
    from .output import write_dry_state_history
    from .config_v841 import V841DryDycoreConfig
    from woof.metal.runtime import MetalContext

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
    native = load_mpas_vertical_grid(init, mesh)
    state, reference, saved = load_mpas_initial_state(
        init, mesh, native.vertical_grid, scalar_names=("qv",),
        terrain_metrics=native.terrain_metrics, return_saved_diagnostics=True,
    )
    config = V841DryDycoreConfig(
        config_dt=float(row.dt_seconds),
        config_time_integration_order=3,
        config_number_of_sub_steps=6,
        config_dynamics_split_steps=1,
        config_scalar_adv_order=3,
        config_scalar_vadv_order=3,
        config_split_dynamics_transport=True,
        config_moist_physics=True,
        config_horiz_mixing="off",
        config_coef_3rd_order=float(getattr(arguments, "coef_3rd_order", 0.25)),
    )
    with Dataset(init, mode="r") as dataset:
        valid_time = _initial_time(dataset)
        wind_u = np.asarray(dataset.variables["u_init"][:], dtype=np.float32)
        wind_v = np.asarray(dataset.variables["v_init"][:], dtype=np.float32)
    from .dynamics_v841 import V841ReferenceWindProfiles

    wind = V841ReferenceWindProfiles(wind_u, wind_v)
    stride = int(history_minutes) * 60 // int(row.dt_seconds)
    with MetalContext() as context:
        driver = DryDycoreDriver(
            mesh, native.vertical_grid, reference, config,
            terrain_metrics=native.terrain_metrics,
            reference_wind_profiles=wind, index_qv=0,
            metal_context=context, backend="metal",
        )
        for step in range(1, steps + 1):
            result = driver.step(state, saved_diagnostics=saved)
            state, saved = result.state, result.saved_diagnostics
            if step % stride == 0:
                frame_time = valid_time + timedelta(seconds=step * row.dt_seconds)
                write_dry_state_history(
                    out / f"history.{step:06d}.nc", mesh, state, frame_time,
                    initial_time=valid_time, vertical_grid=native.vertical_grid,
                )
        state.validate(
            n_cells=int(mesh.dimensions["nCells"]),
            n_edges=int(mesh.dimensions["nEdges"]),
            n_vert_levels=int(mesh.dimensions["nVertLevels"]),
        )

    receipt_path = out / "forecast-receipt.json"
    receipt_path.write_text(json.dumps({
        "schema": "woof.hex.metal-forecast/v1",
        "backend": "metal",
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
