"""Follow condensate through WOOF global, stage by stage, on named columns.

Diagnostic dump script for the grid-scale precipitation defect (handoff
2026-09-02, section 4 item 1; the finding it produced retired the
spectral representation of the hydrometeors, and the tool now follows
the grid tracers through the transport stage as well).  It builds the
model exactly as the run door does, optionally plants a known rain
column and a known snow column on the grid (a Gaussian blob), then
advances the model one step at a time through a copy
of ``MoistHybridModel.step`` that records, after EVERY stage, the column
water of every species on the target columns and the global mean of
each species.  Inside the physics call every scheme of the native
runtime is wrapped the same way, and the Morrison launch itself is
wrapped so the kernel-side profiles (mixing ratios, dz, density, the
per-call surface fallout) are read on the target columns before and
after the kernel.  A third column receives the same rain plant directly
in the kernel batch on the first physics call, bypassing the spectral
path, so the kernel's own sedimentation is measured on its own.

Usage:
  python tools/arwen_global_precip_trace.py CONFIG --out DIR [--steps N]
      [--restart CKPT] [--no-plant] [--kernel-plant] [--rain-kg-kg 1.5e-3]
      [--snow-kg-kg 1.0e-3] [--sigma-cells 4]

Reads nothing it does not verify: checkpoints go through read_checkpoint
(hash-verified); the model is the run door's model.  Writes DIR/trace.json
and DIR/trace.txt.  No model code is changed.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from woof.globe.checkpoint import read_checkpoint, state_from_checkpoint
from woof.globe.config import load_config
from woof.globe.constants import (
    GRAVITY_M_S2,
    GRID_TRACERS,
    NUMBER_MOMENTS,
    WATER_SPECIES,
)
from woof.globe.physics import native_runtime as nr
from woof.globe.runner import build_model_and_cold_state, build_transform
from woof.globe.state import ArwenGlobalState
from woof.globe.spectral.timestep import step_with_scheme
import woof.globe.core.morrison as morrison_mod

CONDENSATE = ("qc", "qr", "qi", "qs", "qg")
PHYSICS_STAGES = (
    "_radiation_step", "_surface_layer_step", "_land_step",
    "_pbl_step", "_cumulus_step", "_microphysics_step",
)
RAIN_DROP_DIAMETER_M = 1.0e-3
SNOW_PARTICLE_DIAMETER_M = 1.0e-3
RHO_WATER = 1000.0
RHO_SNOW = 100.0


class Trace:
    """Everything recorded, plus the helpers that read a state."""

    def __init__(self, model, targets: dict[str, tuple[int, int]]):
        self.model = model
        self.xp = model.transform.backend.xp
        self.host = model.transform.backend.to_numpy
        self.grid = model.transform.grid
        self.targets = targets
        self.rows: list[dict] = []
        self.physics_rows: list[dict] = []
        self.morrison_rows: list[dict] = []
        self.exchange_rows: list[dict] = []
        self.step_index = 0
        self.physics_call = 0
        self.morrison_call = 0
        self.kernel_plant = None
        self.kernel_plant_done = False
        self.morrison_global_precip_mm = 0.0
        self.log_lines: list[str] = []
        # Planted blobs: label -> (species, levels, disc mask on the grid).
        self.blobs: dict[str, tuple[str, list[int], object]] = {}
        self.scale_rows: list[dict] = []
        self.pending_scale_stage = ""
        w = np.asarray(self.grid.quadrature_weights, dtype=np.float64)
        self.cell_weight = self.xp.asarray((w / (2.0 * len(self.grid.longitude_deg)))[:, None])

    def log(self, text: str) -> None:
        self.log_lines.append(text)
        print(text, flush=True)

    # ---- model-state readings ------------------------------------------
    def state_reading(self, bundle: ArwenGlobalState) -> dict:
        model = self.model
        g = model.grid_state(
            bundle.atmosphere,
            only=(*WATER_SPECIES, *NUMBER_MOMENTS, "dp", "p_full", "temperature"),
        )
        xp = self.xp
        dp_g = xp.asarray(g["dp"], dtype=xp.float64) / GRAVITY_M_S2
        out = {"columns": {}, "global_mean_kg_m2": {}, "global_min": {}, "global_max_column_kg_m2": {}}
        for name in WATER_SPECIES:
            q = xp.asarray(g[name], dtype=xp.float64)
            column = xp.sum(q * dp_g, axis=0)
            column_np = self.host(column)
            out["global_mean_kg_m2"][name] = float(self.grid.global_mean(column_np))
            out["global_min"][name] = float(self.host(xp.min(q)))
            out["global_max_column_kg_m2"][name] = float(np.max(column_np))
        surface = bundle.surface
        physics = bundle.physics_state.arrays
        for label, (j, i) in self.targets.items():
            col = {"j": int(j), "i": int(i)}
            for name in WATER_SPECIES:
                q = xp.asarray(g[name][:, j, i], dtype=xp.float64)
                col[f"{name}_kg_m2"] = float(self.host(xp.sum(q * dp_g[:, j, i])))
                col[f"{name}_profile"] = [float(v) for v in self.host(q)]
            for name in NUMBER_MOMENTS:
                col[f"{name}_profile"] = [float(v) for v in self.host(g[name][:, j, i])]
            col["p_full_pa"] = [float(v) for v in self.host(g["p_full"][:, j, i])]
            col["temperature_k"] = [float(v) for v in self.host(g["temperature"][:, j, i])]
            col["condensate_kg_m2"] = sum(col[f"{n}_kg_m2"] for n in CONDENSATE)
            col["surface_rain_kg_m2"] = float(self.host(surface.accumulated_rain_kg_m2[j, i]))
            col["surface_snow_kg_m2"] = float(self.host(surface.accumulated_snow_kg_m2[j, i]))
            col["surface_graupel_kg_m2"] = float(self.host(surface.accumulated_graupel_kg_m2[j, i]))
            col["reservoir_kg_m2"] = float(self.host(surface.water_kg_m2[j, i]))
            for name in ("rainnc", "rainc", "land_rainbl"):
                if name in physics:
                    col[f"physics_{name}"] = float(self.host(physics[name][j, i]))
            out["columns"][label] = col
        for name in ("rainnc", "rainc"):
            if name in physics:
                out["global_mean_kg_m2"][f"physics_{name}"] = float(
                    self.grid.global_mean(self.host(physics[name]))
                )
        out["global_mean_kg_m2"]["surface_rain"] = float(
            self.grid.global_mean(self.host(surface.accumulated_rain_kg_m2))
        )
        out["global_mean_kg_m2"]["surface_snow"] = float(
            self.grid.global_mean(self.host(surface.accumulated_snow_kg_m2))
        )
        # Blob partition: per planted level, the species' mass (global-mean
        # kg/m2 units, i.e. area-weighted) inside the blob disc, the
        # positive mass outside it, and the negative (ringing) mass.
        partition = {}
        for label, (species, levels, mask) in self.blobs.items():
            q = xp.asarray(g[species], dtype=xp.float64)
            rows = {}
            for k in levels:
                layer = q[k] * dp_g[k] * self.cell_weight
                inside = float(self.host(xp.sum(xp.where(mask, layer, 0.0))))
                pos_out = float(self.host(xp.sum(xp.where(~mask, xp.maximum(layer, 0.0), 0.0))))
                neg = float(self.host(xp.sum(xp.minimum(layer, 0.0))))
                rows[int(k)] = {"inside": inside, "outside_positive": pos_out, "negative": neg}
            partition[label] = {"species": species, "levels": rows,
                                "inside": sum(r["inside"] for r in rows.values()),
                                "outside_positive": sum(r["outside_positive"] for r in rows.values()),
                                "negative": sum(r["negative"] for r in rows.values())}
        out["blob_partition"] = partition
        del g
        return out

    def probe(self, stage: str, bundle: ArwenGlobalState) -> None:
        reading = self.state_reading(bundle)
        reading["stage"] = stage
        reading["step"] = self.step_index
        self.rows.append(reading)
        parts = [f"step {self.step_index:2d} {stage:<12s}"]
        for label, col in reading["columns"].items():
            parts.append(
                f"{label}: qv {col['qv_kg_m2']:7.3f} qc {col['qc_kg_m2']:.4f} "
                f"qr {col['qr_kg_m2']:.4f} qi {col['qi_kg_m2']:.4f} qs {col['qs_kg_m2']:.4f} "
                f"qg {col['qg_kg_m2']:.4f} rain {col['surface_rain_kg_m2']:.4f} "
                f"snow {col['surface_snow_kg_m2']:.4f}"
            )
        gm = reading["global_mean_kg_m2"]
        parts.append(
            "GLOBAL: qv %.4f qc %.6f qr %.6f qi %.6f qs %.6f qg %.6f rainnc %.6f rainc %.6f"
            % (gm["qv"], gm["qc"], gm["qr"], gm["qi"], gm["qs"], gm["qg"],
               gm.get("physics_rainnc", float("nan")), gm.get("physics_rainc", float("nan")))
        )
        self.log(" | ".join(parts))
        if reading["blob_partition"]:
            bits = []
            for label, part in reading["blob_partition"].items():
                bits.append(
                    f"{label}/{part['species']} planted levels: inside {part['inside']*1e3:.5f} "
                    f"outside+ {part['outside_positive']*1e3:.5f} neg {part['negative']*1e3:.5f} (e-3 kg/m2 global-mean)"
                )
            self.log(f"        blob partition {stage:<12s} " + " | ".join(bits))

    # ---- batch (physics-side) readings ---------------------------------
    def batch_columns(self, batch) -> dict:
        xp = batch.xp
        out = {}
        for label, (j, i) in self.targets.items():
            rv = xp.asarray(batch.arrays["qv"][:, j, i], dtype=xp.float64)
            moist = 1.0 + rv
            dp = xp.asarray(batch.arrays["dp"][:, j, i], dtype=xp.float64)
            col = {}
            for name in WATER_SPECIES:
                r = xp.asarray(batch.arrays[name][:, j, i], dtype=xp.float64)
                col[name] = float(self.host(xp.sum(r / moist * dp / GRAVITY_M_S2)))
            col["condensate"] = sum(col[n] for n in CONDENSATE)
            out[label] = col
        rv = xp.asarray(batch.arrays["qv"], dtype=xp.float64)
        moist = 1.0 + rv
        dp = xp.asarray(batch.arrays["dp"], dtype=xp.float64)
        col = {}
        for name in WATER_SPECIES:
            r = xp.asarray(batch.arrays[name], dtype=xp.float64)
            column = xp.sum(r / moist * dp / GRAVITY_M_S2, axis=0)
            col[name] = float(self.host(xp.sum(column * self.cell_weight)))
        col["condensate"] = sum(col[n] for n in CONDENSATE)
        out["GLOBAL"] = col
        return out

    def batch_profile(self, batch, label, extra: dict | None = None) -> dict:
        j, i = self.targets[label]
        prof = {}
        for name in (*WATER_SPECIES, *NUMBER_MOMENTS, "p_full", "p_half", "dp", "theta", "exner", "temperature"):
            if name in batch.arrays:
                prof[name] = [float(v) for v in self.host(batch.arrays[name][:, j, i])]
        if extra:
            for name, arr in extra.items():
                prof[name] = [float(v) for v in self.host(arr[:, j, i])]
        return prof


def install_physics_probes(trace: Trace) -> None:
    """Wrap every native-runtime scheme and the Morrison launch."""
    from woof.globe import dynamics as dyn

    # The vapor's column-local hole filler (the per-level global rescale
    # of the spectral-tracer era, which this tool was written to catch,
    # is gone with that era; vapor is the one field the filler sees).
    original_fill = dyn.MoistHybridModel._fill_column_holes

    @functools.wraps(original_fill)
    def fill_wrapped(self, q, dp):
        filled, created, max_fraction, unfillable = original_fill(self, q, dp)
        row = {"step": trace.step_index, "physics_call": trace.physics_call,
               "created_kg_m2": float(created), "max_column_fraction": float(max_fraction),
               "unfillable_kg_m2": float(unfillable),
               "min_before": float(trace.host(self.transform.backend.xp.min(q)))}
        trace.scale_rows.append(row)
        trace.log("      fill_column_holes (vapor): created %.3e kg/m2, max column fraction %.4f, unfillable %.3e, min before %.3e"
                  % (created, max_fraction, unfillable, row["min_before"]))
        return filled, created, max_fraction, unfillable

    dyn.MoistHybridModel._fill_column_holes = fill_wrapped

    for stage in PHYSICS_STAGES:
        original = getattr(nr.NativePhysicsRuntime, stage)

        def make(original, stage):
            @functools.wraps(original)
            def wrapped(self, batch, *args, **kwargs):
                before = trace.batch_columns(batch)
                out = original(self, batch, *args, **kwargs)
                after = trace.batch_columns(batch)
                row = {
                    "step": trace.step_index, "physics_call": trace.physics_call,
                    "stage": stage, "before": before, "after": after,
                }
                if stage == "_microphysics_step":
                    persistent = args[0]
                    f = persistent.arrays
                    row["global_mean_rainncv_mm"] = float(trace.grid.global_mean(trace.host(f["rainncv"])))
                    for label, (j, i) in trace.targets.items():
                        row.setdefault("surface", {})[label] = {
                            name: float(trace.host(f[name][j, i]))
                            for name in ("rainncv", "snowncv", "graupelncv", "rainnc", "snownc", "graupelnc", "land_rainbl")
                        }
                trace.physics_rows.append(row)
                deltas = []
                for label in (*trace.targets, "GLOBAL"):
                    d = {n: after[label][n] - before[label][n] for n in WATER_SPECIES}
                    if label == "GLOBAL":
                        deltas.append("GLOBAL(e-3): " + " ".join(f"d{n} {d[n]*1e3:+.4f}" for n in WATER_SPECIES))
                        continue
                    deltas.append(
                        f"{label}: dqv {d['qv']:+.4f} dqc {d['qc']:+.4f} dqr {d['qr']:+.4f} "
                        f"dqi {d['qi']:+.4f} dqs {d['qs']:+.4f} dqg {d['qg']:+.4f}"
                    )
                trace.log(f"    physics call {trace.physics_call} {stage:<18s} " + " | ".join(deltas))
                return out
            return wrapped

        setattr(nr.NativePhysicsRuntime, stage, make(original, stage))

    original_run = nr.NativePhysicsRuntime.run

    @functools.wraps(original_run)
    def run_wrapped(self, batch, cfg, persistent=None):
        trace.physics_call += 1
        j0, i0 = next(iter(trace.targets.values()))
        p_half = trace.host(batch.arrays["p_half"][:, j0, i0])
        trace.exchange_rows.append({
            "physics_call": trace.physics_call,
            "batch_p_half_first_last_pa": [float(p_half[0]), float(p_half[-1])],
            "batch_dt_s": float(batch.dt_s),
        })
        if trace.physics_call == 1:
            trace.log(
                f"    batch p_half[0]={p_half[0]:.1f} Pa p_half[-1]={p_half[-1]:.1f} Pa "
                f"(bottom-up expected: surface first), dt_s={batch.dt_s}"
            )
        return original_run(self, batch, cfg, persistent)

    nr.NativePhysicsRuntime.run = run_wrapped

    original_launch = morrison_mod.launch_morrison

    @functools.wraps(original_launch)
    def launch_wrapped(theta, qv, qc, qr, qi, qs, qg, nc, nr_, ni, ns, ng, rho, pii, pressure, dz,
                       rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr, dt, **kwargs):
        trace.morrison_call += 1
        xp = trace.xp
        arrays = {"theta": theta, "qv": qv, "qc": qc, "qr": qr, "qi": qi, "qs": qs, "qg": qg,
                  "nc": nc, "nr": nr_, "ni": ni, "ns": ns, "ng": ng, "pii": pii, "pressure": pressure, "dz": dz}
        # Kernel-level plant: on the first launch put rain straight into
        # the batch on column K (900..600 hPa), bypassing the spectral path.
        if trace.kernel_plant is not None and not trace.kernel_plant_done:
            label, amount = trace.kernel_plant
            j, i = trace.targets[label]
            p_col = trace.host(pressure[:, j, i])
            levels = [k for k, p in enumerate(p_col) if 60000.0 <= p <= 90000.0]
            for k in levels:
                qr[k, j, i] = xp.float32(amount)
                nr_[k, j, i] = xp.float32(amount / (math.pi / 6.0 * RHO_WATER * RAIN_DROP_DIAMETER_M ** 3))
            trace.kernel_plant_done = True
            trace.log(f"    KERNEL PLANT on {label} (j={j}, i={i}): qr={amount} kg/kg at batch levels {levels} "
                      f"(p {p_col[levels[0]]:.0f}..{p_col[levels[-1]]:.0f} Pa)")
        before = {}
        for label, (j, i) in trace.targets.items():
            before[label] = {name: [float(v) for v in trace.host(arr[:, j, i])] for name, arr in arrays.items()}
            before[label]["rainncv"] = float(trace.host(rainncv[j, i]))
            before[label]["rainnc"] = float(trace.host(rainnc[j, i]))
        out = original_launch(theta, qv, qc, qr, qi, qs, qg, nc, nr_, ni, ns, ng, rho, pii, pressure, dz,
                              rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr, dt, **kwargs)
        after = {}
        for label, (j, i) in trace.targets.items():
            after[label] = {name: [float(v) for v in trace.host(arr[:, j, i])] for name, arr in arrays.items()}
            after[label]["rho"] = [float(v) for v in trace.host(rho[:, j, i])]
            for name, arr in (("rainncv", rainncv), ("snowncv", snowncv), ("graupelncv", graupelncv),
                              ("rainnc", rainnc), ("snownc", snownc), ("sr", sr)):
                after[label][name] = float(trace.host(arr[j, i]))
        weights = trace.grid.global_mean(trace.host(rainncv))
        trace.morrison_global_precip_mm += float(weights)
        row = {"morrison_call": trace.morrison_call, "physics_call": trace.physics_call, "step": trace.step_index,
               "dt_s": float(dt), "global_mean_rainncv_mm": float(weights),
               "global_max_rainncv_mm": float(trace.host(xp.max(rainncv))),
               "global_mean_rainnc_mm": float(trace.grid.global_mean(trace.host(rainnc))),
               "before": before, "after": after}
        trace.morrison_rows.append(row)
        parts = [f"    MORRISON call {trace.morrison_call} dt={float(dt):.0f}s global rainncv mean {weights:.3e} max {row['global_max_rainncv_mm']:.3e} mm"]
        for label, (j, i) in trace.targets.items():
            b, a = before[label], after[label]
            rho_col = np.asarray(a["rho"]); dz_col = np.asarray(a["dz"])
            def col(prof):
                return float(np.sum(np.asarray(prof) * rho_col * dz_col))
            parts.append(
                f"{label}: qr {col(b['qr']):.3f}->{col(a['qr']):.3f} "
                f"qs {col(b['qs']):.3f}->{col(a['qs']):.3f} "
                f"qc {col(b['qc']):.3f}->{col(a['qc']):.3f} kg/m2 (rho*dz metric) "
                f"rainncv {a['rainncv']:.4f} snowncv {a['snowncv']:.4f} rainnc {a['rainnc']:.4f}"
            )
        trace.log(" | ".join(parts))
        return out

    morrison_mod.launch_morrison = launch_wrapped


def traced_step(trace: Trace, bundle: ArwenGlobalState, dt_s: float) -> ArwenGlobalState:
    """MoistHybridModel.step with a probe after every stage (same order,
    same calls; metrics not assembled)."""
    model = trace.model
    trace.step_index += 1
    half = 0.5 * float(dt_s)
    trace.probe("entry", bundle)
    first, _ = model.apply_physics(bundle, half)
    trace.probe("physics1", first)
    first, _, _, _ = model._repair_positivity(first)
    trace.probe("repair1", first)
    cfl = model.cfl(first.atmosphere, dt_s)
    if cfl > model.maximum_cfl:
        raise ValueError(f"spectral CFL {cfl:.3f} exceeds {model.maximum_cfl:.3f}")
    surface, physics_state = first.surface, first.physics_state
    pre, _ = model.semi_implicit.pre_apply(first.atmosphere, model.transform, model.vertical, dt_s)
    trace.probe("si_pre", ArwenGlobalState(pre, surface, physics_state))
    advanced = step_with_scheme(pre, float(dt_s), model.rhs, model.integrator)
    trace.probe("explicit_rk", ArwenGlobalState(advanced, surface, physics_state))
    advanced, _ = model.semi_implicit.post_apply(advanced, model.transform, model.vertical, dt_s)
    trace.probe("si_post", ArwenGlobalState(advanced, surface, physics_state))
    advanced = model._apply_diffusion(advanced, dt_s)
    trace.probe("diffusion", ArwenGlobalState(advanced, surface, physics_state))
    advanced, _ = model._fix_mass(advanced)
    trace.probe("mass_fix", ArwenGlobalState(advanced, surface, physics_state))
    advanced, transport = model._transport_grid_tracers(first.atmosphere, advanced, dt_s)
    trace.log("      grid tracer transport: courant x/y/z %.3f/%.3f/%.3f substeps %d/%d/%d floor %.3e pseudo-dp mismatch %.2e"
              % (transport["max_courant_x"], transport["max_courant_y"], transport["max_courant_z"],
                 transport["substeps_x"], transport["substeps_y"], transport["substeps_z"],
                 transport["floor_clip_kg_m2"], transport["pseudo_density_mismatch_relative"]))
    trace.probe("transport", ArwenGlobalState(advanced, surface, physics_state))
    advanced.step = bundle.step + 1
    advanced.time_s = bundle.time_s + float(dt_s)
    second, _ = model.apply_physics(ArwenGlobalState(advanced, surface, physics_state), half)
    trace.probe("physics2", second)
    repaired, _, _, _ = model._repair_positivity(second)
    trace.probe("repair2", repaired)
    repaired, _ = model._fix_total_water(repaired)
    trace.probe("water_fix", repaired)
    repaired.atmosphere.step = bundle.step + 1
    repaired.atmosphere.time_s = bundle.time_s + float(dt_s)
    model.enforce(repaired)
    return repaired


def plant_blob(trace: Trace, bundle: ArwenGlobalState, label: str, species: str, amount: float,
               p_bottom_pa: float, p_top_pa: float, sigma_cells: float, number_name: str,
               particle_mass_kg: float) -> ArwenGlobalState:
    """Add a Gaussian blob of ``species`` (and a matching number moment)
    to the spectral state, centred on the target column, between
    p_bottom and p_top at that column."""
    model = trace.model
    xp = trace.xp
    j0, i0 = trace.targets[label]
    g = model.grid_state(bundle.atmosphere, only=("p_full",))
    p_col = trace.host(g["p_full"][:, j0, i0])
    nlev, nlat, nlon = g["p_full"].shape
    del g
    levels = [k for k, p in enumerate(p_col) if p_top_pa <= p <= p_bottom_pa]
    jj = np.arange(nlat)[:, None] - j0
    ii = (np.arange(nlon)[None, :] - i0 + nlon // 2) % nlon - nlon // 2
    weight = np.exp(-0.5 * (jj ** 2 + ii ** 2) / sigma_cells ** 2).astype(np.float32)
    delta = np.zeros((nlev, nlat, nlon), dtype=np.float32)
    for k in levels:
        delta[k] = amount * weight
    # The condensate species and their number moments are grid tracers:
    # the plant lands on the grid exactly as written (no truncation, no
    # ringing), which under the spectral-tracer era was the first place
    # the water went missing.
    dtype = model.transform.backend.float_dtype
    delta_dev = xp.asarray(delta, dtype=dtype)
    assert species in GRID_TRACERS and number_name in GRID_TRACERS
    atmosphere = bundle.atmosphere.with_grid_tracers({
        species: getattr(bundle.atmosphere, species) + delta_dev,
        number_name: getattr(bundle.atmosphere, number_name) + delta_dev / dtype(particle_mass_kg),
    })
    disc = (jj ** 2 + ii ** 2) <= (3.0 * sigma_cells) ** 2
    trace.blobs[label] = (species, levels, xp.asarray(disc))
    trace.log(
        f"PLANT {species} on {label} (j={j0}, i={i0}, lat {trace.grid.latitude_deg[j0]:.2f} "
        f"lon {trace.grid.longitude_deg[i0]:.2f}): {amount} kg/kg x gaussian sigma {sigma_cells} cells at model "
        f"levels {levels} (p {p_col[levels[0]]:.0f}..{p_col[levels[-1]]:.0f} Pa); {number_name} = q / {particle_mass_kg:.3e} kg"
    )
    return ArwenGlobalState(atmosphere, bundle.surface.copy(), bundle.physics_state.copy())


def choose_targets(trace: Trace, bundle: ArwenGlobalState) -> dict[str, tuple[int, int]]:
    model = trace.model
    xp = trace.xp
    g = model.grid_state(bundle.atmosphere, only=(*WATER_SPECIES, "dp"))
    dp_g = xp.asarray(g["dp"], dtype=xp.float64) / GRAVITY_M_S2
    tpw = trace.host(xp.sum(xp.asarray(g["qv"], dtype=xp.float64) * dp_g, axis=0))
    cond = trace.host(sum(xp.sum(xp.asarray(g[n], dtype=xp.float64) * dp_g, axis=0) for n in CONDENSATE))
    nlat, nlon = tpw.shape
    lat = trace.grid.latitude_deg
    # Rain plant: the moistest column between 25S and 25N (RH high, the
    # planted rain is not evaporated by a dry column in ten minutes).
    band = (np.abs(lat) <= 25.0)[:, None]
    masked = np.where(band, tpw, -1.0)
    jr, ir = np.unravel_index(int(np.argmax(masked)), tpw.shape)
    jc, ic = np.unravel_index(int(np.argmax(cond)), cond.shape)
    targets = {
        "R": (int(jr), int(ir)),
        "S": (int(jr), int((ir + nlon // 4) % nlon)),
        "K": (int(jr), int((ir + nlon // 2) % nlon)),
        "C": (int(jc), int(ic)),
    }
    trace.log("targets: " + ", ".join(
        f"{k}=(j {j}, i {i}, lat {lat[j]:.2f}, lon {trace.grid.longitude_deg[i]:.2f}, TPW {tpw[j, i]:.2f}, cond {cond[j, i]:.4f})"
        for k, (j, i) in targets.items()))
    return targets


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--restart", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--no-plant", action="store_true")
    parser.add_argument("--kernel-plant", action="store_true")
    parser.add_argument("--rain-kg-kg", type=float, default=1.5e-3)
    parser.add_argument("--snow-kg-kg", type=float, default=1.0e-3)
    parser.add_argument("--sigma-cells", type=float, default=4.0)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    cfg = load_config(args.config)
    transform = build_transform(cfg)
    model, cold = build_model_and_cold_state(cfg, transform)
    if args.restart is not None:
        metadata, arrays = read_checkpoint(
            args.restart, expected_config_hash=cfg.config_hash,
            semi_implicit_scheme=cfg.semi_implicit_scheme,
        )
        state = state_from_checkpoint(metadata, arrays, transform.backend)
        model.enforce(state)
        del arrays
    else:
        state = cold
    del cold

    trace = Trace(model, {})
    trace.log(f"model built in {time.perf_counter() - started:.1f} s; dt_s {cfg.dt_s}; backend {cfg.backend}")
    trace.targets = choose_targets(trace, state)
    install_physics_probes(trace)

    rain_mass = math.pi / 6.0 * RHO_WATER * RAIN_DROP_DIAMETER_M ** 3
    snow_mass = math.pi / 6.0 * RHO_SNOW * SNOW_PARTICLE_DIAMETER_M ** 3
    if not args.no_plant:
        state = plant_blob(trace, state, "R", "qr", args.rain_kg_kg, 90000.0, 60000.0, args.sigma_cells, "nr", rain_mass)
        state = plant_blob(trace, state, "S", "qs", args.snow_kg_kg, 60000.0, 35000.0, args.sigma_cells, "ns", snow_mass)
        if args.kernel_plant:
            trace.kernel_plant = ("K", args.rain_kg_kg)
    trace.probe("planted", state)
    for _ in range(args.steps):
        state = traced_step(trace, state, cfg.dt_s)
    trace.log(f"Morrison global fallout summed over {trace.morrison_call} calls: {trace.morrison_global_precip_mm:.6f} mm")

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": str(args.config), "restart": None if args.restart is None else str(args.restart),
        "steps": args.steps, "dt_s": cfg.dt_s, "targets": trace.targets,
        "plant": None if args.no_plant else {
            "rain_kg_kg": args.rain_kg_kg, "snow_kg_kg": args.snow_kg_kg, "sigma_cells": args.sigma_cells,
            "kernel_plant": args.kernel_plant,
        },
        "stages": trace.rows, "physics": trace.physics_rows, "morrison": trace.morrison_rows,
        "exchange": trace.exchange_rows, "fill_water_holes": trace.scale_rows,
        "morrison_global_precip_mm": trace.morrison_global_precip_mm,
        "wall_s": time.perf_counter() - started,
    }
    (args.out / "trace.json").write_text(json.dumps(payload, indent=1))
    (args.out / "trace.txt").write_text("\n".join(trace.log_lines) + "\n")
    trace.log(f"wrote {args.out / 'trace.json'} in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
