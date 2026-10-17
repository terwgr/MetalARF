"""The Metal forecast worker: one process, one segment, one mesh bind.

``python -m woof.cycle.metal_worker {seed|segment} --port-config CFG --steps N
--out DIR [--anchor DIR]``, launched by :func:`woof.cycle.mpas_bridge.launch`
for the ``mpas-metal`` parent kinds.  It is the Metal counterpart of
``mpas_cycle_bridge.worker`` and writes the same segment layout, read by the
same :func:`woof.cycle.mpas_bridge.read_segment`.

The mesh bind rebinds module constants of the forecast driver and may run
once per process, so every leg is a process of its own and the anchor on disk
is the only channel between legs.

``seed``
    Build the Metal physics stack from the case init and integrate ``--steps``
    into the ``anchor`` segment; CONTINUE THE SAME STACK for another
    ``--steps`` into the ``control`` segment, the un-analysed continuation a
    zero increment must reproduce bitwise.

``segment``
    Read a committed anchor, apply the increment it carries on the host,
    open the stack on the analysed state, restore the physics restart
    payload, read the state back off the device, integrate ``--steps``.

The port config is JSON: ``mesh``, ``grid``, ``static``, ``init`` and
``start_time``, plus optional ``lbc_dir``, ``convection``, ``pbl_cadence``
and ``horiz_mixing``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mpas_cycle_bridge import BRIDGE_SCHEMA
from mpas_cycle_bridge import anchor_codec as codec
from mpas_cycle_bridge.portbind import (PROGNOSTIC_FIELDS,
                                        SAVED_DIAGNOSTIC_FIELDS,
                                        VERTICAL_METRIC_FIELDS)

ENGINE = "woof.hex metal"
REHYDRATION = ("open_metal_physics_stack_v841(state=, saved_diagnostics=) + "
               "restore_restart_state(backend, previous_surface, gf_forcing)")
SEGMENT_MANIFEST = "segment.json"


def provenance() -> dict[str, Any]:
    """Where the woof that ran this leg came from, for the spine to grade."""
    import woof

    return {"engine": ENGINE, "woof_version": woof.__version__,
            "module_path": str(Path(woof.__file__).resolve().parent)}


# --------------------------------------------------------------------------
# the restart payload, split for the anchor


def split_restart(payload: Mapping[str, Any]
                  ) -> tuple[Any, dict[str, np.ndarray]]:
    """JSON skeleton plus arrays: the checkpoint's own flattening."""
    from woof.hex.checkpoint import _flatten_physics

    arrays: dict[str, np.ndarray] = {}
    return _flatten_physics(payload, "", arrays), arrays


def join_restart(skeleton: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    from woof.hex.checkpoint import _rebuild_physics

    return _rebuild_physics(skeleton, arrays)


# --------------------------------------------------------------------------
# segment output


def _write_segment(out_dir: Path, *, role: str, boundary: Mapping[str, Any],
                   receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Emit one boundary as a segment directory, in the anchor's format."""
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = codec.SUFFIX[codec.ARRAY_FORMAT]
    skeleton, seam = split_restart(boundary["restart"])
    written: dict[str, str] = {}

    def emit(stem: str, mapping: Mapping[str, np.ndarray]) -> str:
        target = out_dir / f"{stem}{suffix}"
        codec.write_arrays(target, mapping)
        codec.fsync_file(target)
        written[target.name] = codec.file_sha256(target)
        return target.name

    manifest = {
        "schema": BRIDGE_SCHEMA,
        "role": str(role),
        "array_format": codec.ARRAY_FORMAT,
        "prognostic_path": emit(codec.PROGNOSTIC_STEM, boundary["prognostic"]),
        "derived_path": emit(codec.DERIVED_STEM, boundary["derived"]),
        "seam_path": emit(codec.SEAM_STEM, seam),
        "metric_path": emit(codec.METRIC_STEM, boundary["metric"]),
        "backend_restart": skeleton,
        "prognostic_sha256": codec.prognostic_sha256(boundary["prognostic"]),
        "model_time_seconds": float(boundary["model_time_seconds"]),
        "files": written,
        **dict(receipt),
    }
    path = out_dir / SEGMENT_MANIFEST
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n",
                    encoding="utf-8")
    codec.fsync_file(path)
    return manifest


# --------------------------------------------------------------------------
# the stack


class Leg:
    """The bound mesh, the host preparation and the stack of one process."""

    def __init__(self, config: Mapping[str, Any], context: Any) -> None:
        from woof.hex.forecast_door import (DRIVERS_DIR, _load_module,
                                            load_registry)

        self.mesh = str(config["mesh"])
        paths = {role: Path(config[role]).expanduser().absolute()
                 for role in ("grid", "static", "init")}
        binding_module = _load_module("mpas_mesh_binding",
                                      DRIVERS_DIR / "mpas_mesh_binding.py")
        binding = binding_module.MESH_BINDINGS[self.mesh]
        self.dt = float(load_registry()[self.mesh].dt_seconds)
        self.n_cells, self.n_edges = int(binding.n_cells), int(binding.n_edges)
        lbc_paths = None
        if bool(binding.regional) != bool(config.get("lbc_dir")):
            raise ValueError(
                f"mesh {self.mesh} is "
                f"{'limited-area' if binding.regional else 'global'}: "
                f"lbc_dir is {'required' if binding.regional else 'for limited-area meshes only'}")
        if binding.regional:
            lbc_paths = sorted(str(item) for item in
                               Path(config["lbc_dir"]).glob("lbc.*.nc"))
            if not lbc_paths:
                raise ValueError(f"lbc_dir {config['lbc_dir']} holds no lbc.*.nc")
        convection = str(config.get("convection", "auto"))
        pbl_cadence = str(config.get("pbl_cadence", "auto"))
        runner = importlib.import_module(
            "woof.hex.drivers.run_cuda_v841_forecast")
        binding_module.bind_mesh(
            runner.proof, self.mesh, grid=paths["grid"], static=paths["static"],
            forecast=runner, convection=convection, pbl_cadence=pbl_cadence)
        authority = runner.verify_forecast_authorities(paths)
        self.host = runner.prepare_forecast_host(
            paths, authority, start_time_text=config.get("start_time"),
            horiz_mixing=str(config.get("horiz_mixing", "2d_smagorinsky")),
            convection=convection, pbl_cadence=pbl_cadence,
            lbc_paths=lbc_paths)
        # The forecast door's restart binding: dynamics configuration and
        # the grid, static and init hashes, so an anchor from another case
        # is refused rather than resumed.
        from woof.hex.metal_forecast import _restart_setup
        self.setup = _restart_setup(
            self.host, paths, local_timestep=False, rates=None, rings=0,
            known={role: record.get("sha256")
                   for role, record in authority["files"].items()})
        self.context = context
        self.stack: dict[str, Any] | None = None
        self.previous: dict[str, Any] = {}
        self.forcing: Any = None
        self.steps_run = 0

    def open(self, state: Any = None, saved: Any = None,
             restart: Mapping[str, Any] | None = None) -> None:
        from woof.hex.metal_arwen_physics_v841 import \
            open_metal_physics_stack_v841
        from woof.hex.metal_driver_v841 import MetalGfDynamicsTendenciesV841

        self.stack = open_metal_physics_stack_v841(
            self.context, self.host, state=state, saved_diagnostics=saved,
            rollback=False)
        backend = self.stack["backend"]
        if restart is None:
            self.previous = backend.accumulated_precipitation()
            return
        if "setup" in restart and restart["setup"] != self.setup:
            raise ValueError(
                "the anchor's restart was written for another configuration "
                "or other grid, static or init files")
        backend.restore_restart_state(restart["backend"])
        if float(backend.seam.elapsed_seconds) != float(state.time_seconds):
            raise ValueError(
                f"the physics clock {backend.seam.elapsed_seconds} s differs "
                f"from the dycore's {state.time_seconds} s")
        upload = self.stack["driver"].arrays.asarray
        self.previous = {name: upload(np.ascontiguousarray(value))
                         for name, value in restart["previous_surface"].items()}
        carried = restart["gf_forcing"]
        self.forcing = None if carried is None else MetalGfDynamicsTendenciesV841(
            rthdynten=upload(carried["rthdynten"]),
            rqvdynten=upload(carried["rqvdynten"]),
            time_seconds=float(carried["time_seconds"]))

    def advance(self, steps: int) -> int:
        from woof.hex.metal_arwen_physics_v841 import (
            execute_composite_step_metal_v841, step_health_gate_metal_v841)

        driver, backend = self.stack["driver"], self.stack["backend"]
        first = int(round(float(driver.atmosphere.state.time_seconds) / self.dt))
        for step in range(first + 1, first + int(steps) + 1):
            result = execute_composite_step_metal_v841(
                driver=driver, backend=backend,
                scalar_names=self.stack["scalar_names"],
                physics_geometry=self.stack["physics_geometry"],
                previous_surface_updates=self.previous,
                dynamics_tendencies=self.forcing, refl_10cm_due=False)
            self.previous = result.surface_updates
            self.forcing = result.dynamics_tendencies
            step_health_gate_metal_v841(self.stack, step)
        return int(steps)

    def download(self) -> dict[str, Any]:
        """The committed boundary off the device, in the anchor's shape."""
        from woof.hex.metal_forecast import _solve_region

        driver, backend = self.stack["driver"], self.stack["backend"]
        state = _solve_region(driver.atmosphere.state.to_host(),
                              self.n_cells, self.n_edges)
        saved = _solve_region(driver.atmosphere.saved.to_host(),
                              self.n_cells, self.n_edges)
        prognostic = {name: np.ascontiguousarray(getattr(state, name))
                      for name in PROGNOSTIC_FIELDS}
        prognostic["time_seconds"] = np.asarray(float(state.time_seconds))
        metric = {}
        for name in VERTICAL_METRIC_FIELDS:
            value = np.ascontiguousarray(
                getattr(driver.atmosphere.vertical, name).to_numpy())
            metric[name] = value[..., :self.n_cells]
        forcing = self.forcing
        return {
            "prognostic": prognostic,
            "derived": {name: np.ascontiguousarray(getattr(saved, name))
                        for name in SAVED_DIAGNOSTIC_FIELDS},
            "metric": {name: np.ascontiguousarray(value)
                       for name, value in metric.items()},
            "restart": {
                "setup": self.setup,
                "backend": backend.restart_state(),
                "gf_forcing": None if forcing is None else {
                    "rthdynten": forcing.rthdynten.to_numpy(),
                    "rqvdynten": forcing.rqvdynten.to_numpy(),
                    "time_seconds": float(forcing.time_seconds)},
                "previous_surface": {name: value.to_numpy()
                                     for name, value in self.previous.items()},
            },
            "model_time_seconds": float(state.time_seconds),
        }


# --------------------------------------------------------------------------
# phases


def _config(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def seed(args: argparse.Namespace) -> int:
    from woof.hex import metal_v841_launch
    from woof.metal.runtime import MetalContext

    config = _config(args.port_config)
    out = Path(args.out).expanduser().absolute()
    with MetalContext() as context, metal_v841_launch.preserve_post_launch():
        leg = Leg(config, context)
        leg.open()
        common = {"dt_seconds": leg.dt, "steps_requested": int(args.steps),
                  "provenance": provenance(),
                  "stamp_basis": "the analysis re-enters the dycore through "
                                 "open_metal_physics_stack_v841's state seam"}
        first = leg.advance(args.steps)
        anchor = _write_segment(
            out / "anchor", role="anchor", boundary=leg.download(),
            receipt={**common, "steps_executed": first,
                     "rehydrated_sha256": None,
                     "rehydration": "none: seed constructs from the case init"})
        second = leg.advance(args.steps)
        control = _write_segment(
            out / "control", role="control", boundary=leg.download(),
            receipt={**common, "steps_executed": second,
                     "rehydrated_sha256": None,
                     "rehydration": "none: continued in-process from the "
                                    "anchor boundary without a download/"
                                    "upload round trip -- this is the control"})
    print(json.dumps({"anchor": anchor["prognostic_sha256"],
                      "control": control["prognostic_sha256"],
                      "anchor_steps": first, "control_steps": second},
                     indent=2))
    return 0


def segment(args: argparse.Namespace) -> int:
    from woof.hex import metal_v841_launch
    from woof.hex.driver import DrySavedDiagnostics
    from woof.hex.state import PrognosticState
    from woof.metal.runtime import MetalContext

    anchor_dir = Path(args.anchor).expanduser().absolute()
    manifest = codec.read_manifest(anchor_dir)
    prognostic = codec.read_member(anchor_dir, manifest["parent"]["path"])
    derived = codec.read_member(anchor_dir, manifest["derived"]["path"])
    seam = codec.read_member(anchor_dir, (manifest.get("seam") or {}).get("path"))
    increment = codec.read_member(
        anchor_dir, (manifest.get("analysis") or {}).get("path"))

    background_sha = codec.prognostic_sha256(prognostic)
    recorded = manifest["parent"]["prognostic_sha256"]
    if background_sha != recorded:
        raise codec.CodecRefusal(
            "anchor prognostics do not hash to what the manifest recorded",
            anchor=str(anchor_dir), recorded=recorded, actual=background_sha)

    # The analysis is applied on the host and never touched again: what the
    # device is handed is the analysed state, and the gate downstream asks
    # whether the device gave that exact state back.  Only cells the
    # increment actually moves are written, so a zero increment is bitwise
    # inert (x + 0.0 would turn a -0.0 into +0.0).
    applied: list[str] = []
    for name, delta in sorted((increment or {}).items()):
        if name not in prognostic:
            raise codec.CodecRefusal(
                "analysis increment names a field the parent state does not "
                "carry", field=name, present=sorted(prognostic))
        base, delta = prognostic[name], np.asarray(delta)
        if base.shape != delta.shape:
            raise codec.CodecRefusal(
                "analysis increment does not match the field it increments",
                field=name, state_shape=list(base.shape),
                increment_shape=list(delta.shape))
        delta = delta.astype(base.dtype, copy=False)
        prognostic[name] = codec.contiguous(
            np.where(delta != 0, base + delta, base).astype(base.dtype,
                                                            copy=False))
        applied.append(name)
    analysed_sha = codec.prognostic_sha256(prognostic)

    skeleton = manifest.get("backend_restart")
    if skeleton is None or not seam:
        raise codec.CodecRefusal(
            "anchor carries no physics restart payload; a Metal leg cannot "
            "resume from it", anchor=str(anchor_dir))
    restart = join_restart(skeleton, seam)
    state = PrognosticState(
        **{name: np.ascontiguousarray(prognostic[name])
           for name in PROGNOSTIC_FIELDS},
        time_seconds=float(np.asarray(prognostic["time_seconds"]).reshape(-1)[0]))
    saved = DrySavedDiagnostics(
        **{name: np.ascontiguousarray(derived[name])
           for name in SAVED_DIAGNOSTIC_FIELDS})

    config = _config(args.port_config)
    with MetalContext() as context, metal_v841_launch.preserve_post_launch():
        leg = Leg(config, context)
        leg.open(state=state, saved=saved, restart=restart)
        # THE evidence that the analysis reached the dycore: the state read
        # back off the device before any step, hashed.
        rehydrated_sha = codec.prognostic_sha256(leg.download()["prognostic"])
        executed = leg.advance(args.steps)
        written = _write_segment(
            Path(args.out).expanduser().absolute(), role="segment",
            boundary=leg.download(),
            receipt={
                "dt_seconds": leg.dt,
                "steps_requested": int(args.steps),
                "steps_executed": executed,
                "provenance": provenance(),
                "source_anchor": str(anchor_dir),
                "background_sha256": background_sha,
                "analysed_sha256": analysed_sha,
                "rehydrated_sha256": rehydrated_sha,
                "increment_fields": applied,
                "increment_sha256": (None if not increment
                                     else codec.prognostic_sha256(increment)),
                "increment_nonzero_cells": (
                    0 if not increment else int(sum(
                        int(np.count_nonzero(value))
                        for value in increment.values()))),
                "backend_restart_restored": True,
                "rehydration": REHYDRATION,
            })
    print(json.dumps({"prognostic_sha256": written["prognostic_sha256"],
                      "rehydrated_sha256": rehydrated_sha,
                      "analysed_sha256": analysed_sha,
                      "steps_executed": executed}, indent=2))
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="woof.cycle.metal_worker",
                                     description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    for name, func in (("seed", seed), ("segment", segment)):
        phase = sub.add_parser(name)
        phase.add_argument("--port-config", required=True)
        phase.add_argument("--steps", type=int, required=True)
        phase.add_argument("--out", required=True)
        if name == "segment":
            phase.add_argument("--anchor", required=True)
        phase.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    status = Path(args.out).expanduser().absolute() / "worker-status.json"
    try:
        code = int(args.func(args))
        payload = {"ok": True, "returncode": code, "phase": args.phase}
    except BaseException as error:  # noqa: BLE001 - recorded, then re-raised
        payload = {"ok": False, "returncode": 1, "phase": args.phase,
                   "error": f"{type(error).__name__}: {error}",
                   "traceback": traceback.format_exc()}
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        raise
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
