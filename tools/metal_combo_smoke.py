#!/usr/bin/env python3
"""Bounded idealized Metal caller smoke; no parity or numerical acceptance.

Example: python tools/metal_combo_smoke.py --microphysics 0,8 --pbl 0,5
Preflight resolves the existing initialization gates without a Metal device.
An 'ok' preflight means gates resolved, never that kernels executed.
"""
from __future__ import annotations

import argparse
import tempfile
import contextlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MARKER = "WOOF_COMBO_RESULT="
FAILED_STATUSES = frozenset(("crash", "NaN"))
GF_PROFILE_ROW = (8, 5, 4, 3, 1, "periodic")
RUC_PROFILE_ROW = (0, 0, 3, 0, 0, "periodic")


class GatesResolved(Exception):
    pass


class GateModule:
    """Stop at first device allocation, after the real initializer gates."""
    __woof_metal_array_module__ = True

    def __getattr__(self, name):
        if name in ("asarray", "zeros", "empty", "full", "ones"):
            def stop(*args, **kwargs):
                raise GatesResolved()
            return stop
        raise AttributeError(name)


from woof.metal.suites import (  # noqa: E402
    TURBULENCE_PROFILES, _urban_option, configuration, execute, initialize,
    suite_rows)


def preflight(combo, *, urban=False, turbulence=None, sfclay_option=None,
              terrain_opt=None, mp_zero_out=0, mp_zero_out_all=0,
              sf_lake_physics=0, alb_sol=0, spp_lsm=0, spp_conv=0,
              h_sca_adv_order=2,
              bl_mynn_version="wrf_461", thompson_version="wrf_461",
              noahmp_glacier_smoke=False):
    cfg = configuration(combo, urban=urban, turbulence=turbulence,
                        sfclay_option=sfclay_option,
                        terrain_opt=terrain_opt,
                        mp_zero_out=mp_zero_out,
                        mp_zero_out_all=mp_zero_out_all,
                        sf_lake_physics=sf_lake_physics,
                        alb_sol=alb_sol, spp_lsm=spp_lsm,
                        spp_conv=spp_conv,
                        h_sca_adv_order=h_sca_adv_order,
                        bl_mynn_version=bl_mynn_version,
                        thompson_version=thompson_version)
    state_fields = dict(
        _metal_setup_state=True, _array_module=GateModule(),
        p=SimpleNamespace(shape=(10, 8, 8)),
        mup=SimpleNamespace(shape=(8, 8)), qv=object(), p_top=20000.)
    if cfg.bl_pbl_physics == 900:
        state_fields["e_sgs"] = object()
    state = SimpleNamespace(**state_fields)
    try:
        initialize(state, cfg, gates_only=True,
                   noahmp_glacier_smoke=noahmp_glacier_smoke)
    except GatesResolved:
        pass
    return cfg



PAIRWISE_AXES = (
    (0, 1, 6, 8, 9, 10, 16, 18, 28, 50),  # MP
    (0, 1, 2, 5, 9, 11),             # PBL
    (0, 1, 2, 5, 91),                # SFCLAY
    (0, 2, 3, 4),                    # LSM
    (0, 1, 3, 16),                   # CU
    (0, 1, 4, 11, 40),               # RAD (40 = RRTMGP LW only)
    ("periodic", "open", "specified"),
)

# Every Metal-advertised MP option is in the executable pairwise metric.
PAIRWISE_EXECUTABLE_MP = frozenset((0, 1, 6, 8, 9, 10, 16, 18, 28, 50))
PAIRWISE_EXCLUDED_MP_REASONS: dict[int, str] = {}


def pairwise_exclusion_notes():
    """Report Metal-advertised MP options missing from executable coverage."""
    from woof.core.physics import METAL_FAMILY_CAPS

    advertised = set(METAL_FAMILY_CAPS["mp"])
    represented = set(PAIRWISE_AXES[0])
    excluded = sorted(advertised - set(PAIRWISE_EXECUTABLE_MP))
    notes = []
    for mp in excluded:
        if mp not in represented:
            reason = "not present in PAIRWISE_AXES"
        else:
            reason = PAIRWISE_EXCLUDED_MP_REASONS.get(
                mp, "not in PAIRWISE_EXECUTABLE_MP; add an explicit reason")
        notes.append(f"PAIRWISE EXCLUDED MP={mp}: {reason}")
    for mp in sorted(set(PAIRWISE_EXECUTABLE_MP) - represented):
        notes.append(
            f"PAIRWISE COVERAGE GAP MP={mp}: executable allowlist option "
            "is not present in PAIRWISE_AXES")
    return notes


def pairwise_rows():
    """Greedy cover using cheap WRF and static WOOF executability filters."""
    import numpy as np
    from woof.physics_compat import pending_wrf_physics_components

    encoded = np.indices(tuple(map(len, PAIRWISE_AXES)), dtype=np.int16)
    candidates = encoded.reshape(len(PAIRWISE_AXES), -1).T
    pbl = np.asarray(PAIRWISE_AXES[1])[candidates[:, 1]]
    sfclay = np.asarray(PAIRWISE_AXES[2])[candidates[:, 2]]
    lsm = np.asarray(PAIRWISE_AXES[3])[candidates[:, 3]]
    cu = np.asarray(PAIRWISE_AXES[4])[candidates[:, 4]]
    # WRF's PBL/surface-layer table, the MYJ/Eta pairing, UW's surface-layer
    # requirement, and Grell-Freitas' KPBL dependency are all selector-only
    # rules. The static WOOF physics readiness table adds the unfinished
    # component combinations without constructing configs or Metal drivers.
    valid = ((pbl == 2) == (sfclay == 2))
    valid &= ~((pbl == 9) & (sfclay == 0))
    valid &= ~((cu == 3) & (pbl == 0))
    valid &= (sfclay != 0) | (lsm == 0)
    valid &= np.isin(
        np.asarray(PAIRWISE_AXES[0])[candidates[:, 0]],
        tuple(PAIRWISE_EXECUTABLE_MP))
    ready_cache = {}
    for index in np.flatnonzero(valid):
        row = candidates[index]
        mp, pbl_value, sfclay_value, lsm, _cu, _ra, _boundary = (
            PAIRWISE_AXES[axis][int(row[axis])]
            for axis in range(len(PAIRWISE_AXES)))
        key = (mp, pbl_value, sfclay_value, lsm)
        if key not in ready_cache:
            blockers = pending_wrf_physics_components(
                mp_physics=mp, sf_sfclay_physics=sfclay_value,
                bl_pbl_physics=pbl_value, sf_surface_physics=lsm,
                num_soil_layers=9 if lsm == 3 else 4)
            ready_cache[key] = not blockers
        if not ready_cache[key]:
            valid[index] = False
    candidates = candidates[valid]
    uncovered = {
        (left, right): np.ones(
            (len(PAIRWISE_AXES[left]), len(PAIRWISE_AXES[right])),
            dtype=np.bool_)
        for left in range(len(PAIRWISE_AXES))
        for right in range(left + 1, len(PAIRWISE_AXES))
    }
    selected = []
    while True:
        scores = np.zeros(len(candidates), dtype=np.int16)
        for (left, right), pairs in uncovered.items():
            scores += pairs[candidates[:, left], candidates[:, right]]
        best_index = int(scores.argmax())
        if scores[best_index] == 0:
            break
        best = candidates[best_index]
        selected.append(tuple(PAIRWISE_AXES[axis][int(best[axis])]
                              for axis in range(len(PAIRWISE_AXES))))
        for (left, right), pairs in uncovered.items():
            pairs[best[left], best[right]] = False
    rows = []
    for mp, pbl, sfclay, lsm, cu, ra, boundary in selected:
        rows.append((f"PAIRWISE MP={mp} PBL={pbl} SFCLAY={sfclay} "
                     f"LSM={lsm} CU={cu} RAD={ra} {boundary}",
                     (mp, pbl, lsm, cu, ra, boundary),
                     {"sfclay_option": sfclay}))
    rows.append(("MYNN GSD 4.1 | mixlength 2 | EDMF plume",
                 (8, 5, 4, 0, 0, "periodic"),
                 {"sfclay_option": 5, "bl_mynn_version": "gsd_41"}))
    rows.append(("Thompson WRF 3.9 NOAA fork",
                 (28, 0, 0, 0, 0, "periodic"),
                 {"thompson_version": "wrf_39_noaa"}))
    rows.append(("RUC alb_sol=1", (0, 0, 3, 0, 11, "periodic"),
                 {"sfclay_option": 91, "alb_sol": 1}))
    ruc_row = (0, 0, 3, 0, 0, "periodic")
    rows.append(("RUC sf_lake_physics=1", ruc_row,
                 {"sfclay_option": 91, "sf_lake_physics": 1}))
    rows.append(("RUC spp_lsm=1", ruc_row,
                 {"sfclay_option": 91, "spp_lsm": 1}))
    rows.append(("Noah-MP glacier column",
                 (8, 5, 4, 0, 0, "periodic"),
                 {"sfclay_option": 5, "noahmp_glacier_smoke": True}))
    return rows


def run_single_context(rows, *, output=None, notes=()):
    """Run a bounded set sequentially while reusing one Metal context."""
    import gc
    import signal
    from woof.metal.array import MetalArrayModule

    class RunDeadline(Exception):
        pass

    def deadline(_signum, _frame):
        raise RunDeadline("single-context smoke exceeded 285 seconds")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(285)
    module = MetalArrayModule()
    lines = list(notes)
    counts = {"ok": 0, "refused": 0, "crash": 0, "NaN": 0}
    started = time.monotonic()
    try:
        for note in notes:
            print(note, flush=True)
        for label, combo, options in rows:
            try:
                status, detail = result(combo, only=True, **options)
                if status == "ok":
                    status, detail = execute(combo, module=module, steps=1,
                                             **options)
            except RunDeadline:
                raise
            except Exception as exc:
                status, detail = "crash", f"{type(exc).__name__}: {exc}"
            counts[status] = counts.get(status, 0) + 1
            lines.append(f"{label} | {status} | {detail[:180]}")
            gc.collect()
            print(lines[-1], flush=True)
    except RunDeadline as exc:
        lines.append(f"INCOMPLETE | crash | {exc}")
        counts["crash"] += 1
        print(lines[-1], flush=True)
    finally:
        module.context.close()
        signal.alarm(0)
    elapsed = time.monotonic() - started
    summary = (f"rows={len(rows)} ok={counts['ok']} refused={counts['refused']} "
               f"crash={counts['crash']} NaN={counts['NaN']} "
               f"elapsed={elapsed:.1f}s")
    print(summary, flush=True)
    if output is not None and not any(line.startswith("INCOMPLETE") for line in lines):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(summary + "\n" + "\n".join(lines) + "\n")
    return int(any(counts[status] for status in FAILED_STATUSES))


def result(combo, only=False, *, urban=False, turbulence=None, sfclay_option=None,
           terrain_opt=None, mp_zero_out=0, mp_zero_out_all=0, steps=2,
           sf_lake_physics=0, alb_sol=0, spp_lsm=0, spp_conv=0,
           h_sca_adv_order=2,
           bl_mynn_version="wrf_461", thompson_version="wrf_461",
           noahmp_glacier_smoke=False):
    try:
        preflight(combo, urban=urban, turbulence=turbulence,
                 sfclay_option=sfclay_option, terrain_opt=terrain_opt,
                 mp_zero_out=mp_zero_out,
                 mp_zero_out_all=mp_zero_out_all,
                 sf_lake_physics=sf_lake_physics,
                 alb_sol=alb_sol, spp_lsm=spp_lsm,
                 spp_conv=spp_conv,
                 h_sca_adv_order=h_sca_adv_order,
                 bl_mynn_version=bl_mynn_version,
                 thompson_version=thompson_version,
                 noahmp_glacier_smoke=noahmp_glacier_smoke)
    except (NotImplementedError, ValueError) as exc:
        return "refused", str(exc).splitlines()[0]
    except Exception as exc:
        return "crash", f"{type(exc).__name__}: {exc}"
    if only:
        return "ok", "initialization gates resolved; no execution"
    try:
        return execute(combo, urban=urban, turbulence=turbulence,
                       sfclay_option=sfclay_option,
                       terrain_opt=terrain_opt, steps=steps,
                       mp_zero_out=mp_zero_out,
                       mp_zero_out_all=mp_zero_out_all,
                       sf_lake_physics=sf_lake_physics,
                       alb_sol=alb_sol, spp_lsm=spp_lsm,
                       spp_conv=spp_conv,
                       h_sca_adv_order=h_sca_adv_order,
                       bl_mynn_version=bl_mynn_version,
                       thompson_version=thompson_version,
                       noahmp_glacier_smoke=noahmp_glacier_smoke)
    except NotImplementedError as exc:
        return "refused", str(exc).splitlines()[0]
    except ValueError as exc:
        if "unregistered Metal kernel entry point" in str(exc):
            return "refused", "needs integration: " + str(exc)
        return "crash", f"ValueError: {exc}"
    except Exception as exc:
        return "crash", f"{type(exc).__name__}: {exc}"


def child(combo, timeout, *, urban=False, turbulence=None, sfclay_option=None,
          terrain_opt=None, mp_zero_out=0, mp_zero_out_all=0, steps=2,
          sf_lake_physics=0, alb_sol=0, spp_lsm=0, spp_conv=0,
          bl_mynn_version="wrf_461", thompson_version="wrf_461",
          noahmp_glacier_smoke=False):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    urban_option = _urban_option(urban)
    try:
        proc = _run_process_group(
            [sys.executable, str(Path(__file__).resolve()), "--case",
             json.dumps(combo),
             *( ["--urban-model", str(urban_option)]
                if urban_option in (2, 3) else
                (["--urban"] if urban_option == 1 else [])),
             *( ["--turbulence", turbulence] if turbulence else []),
             *( ["--sfclay-option", str(sfclay_option)]
                if sfclay_option is not None else []),
             *( ["--terrain-opt", str(terrain_opt)]
                if terrain_opt is not None else []),
             *( ["--mynn-version", bl_mynn_version]
                if bl_mynn_version != "wrf_461" else []),
             *( ["--thompson-version", thompson_version]
                if thompson_version != "wrf_461" else []),
             *( ["--noahmp-glacier-smoke"] if noahmp_glacier_smoke else []),
             *( ["--sf-lake-physics", str(sf_lake_physics)]
                if sf_lake_physics else []),
             *( ["--alb-sol", str(alb_sol)] if alb_sol else []),
             *( ["--spp-lsm", str(spp_lsm)] if spp_lsm else []),
             *( ["--spp-conv", str(spp_conv)] if spp_conv else []),
             "--mp-zero-out", str(mp_zero_out),
             "--mp-zero-out-all", str(mp_zero_out_all),
             "--steps", str(steps)],
            env=env, timeout=timeout, capture_output=True)
    except subprocess.TimeoutExpired:
        return "crash", f"public run timeout ({timeout:.0f}s)"
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(MARKER):
            return tuple(json.loads(line[len(MARKER):]))
    return "crash", f"exit {proc.returncode}: " + " ".join(proc.stdout.splitlines()[-2:])[:180]


def _run_process_group(command, *, env, timeout, capture_output=False):
    """Run one isolated process group and kill descendants on timeout."""
    stdout_pipe = subprocess.PIPE if capture_output else None
    proc = subprocess.Popen(
        command, env=env, stdout=stdout_pipe,
        stderr=subprocess.STDOUT if capture_output else None,
        text=True, start_new_session=True)
    try:
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise
    return subprocess.CompletedProcess(command, proc.returncode, stdout, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("microphysics", "0,1,8,10,18,28,50"), ("pbl", "0,1,5"),
                          ("land-surface", "0,2,4"), ("cumulus", "0,1,16"),
                          ("radiation", "0,1,4,11")):
        parser.add_argument("--" + name, default=default,
                            help="comma-separated WRF scheme IDs")
    parser.add_argument("--boundaries", default="periodic,open,specified")
    parser.add_argument("--sfclay-option", type=int, choices=(1, 5, 91),
                        help="force a supported surface-layer option")
    parser.add_argument("--terrain-opt", type=int, choices=(0, 1),
                        help="override terrain_opt for a focused --case")
    parser.add_argument("--h-sca-adv-order", type=int, choices=(2, 5),
                        default=2,
                        help="horizontal scalar/geopotential advection order")
    parser.add_argument("--mynn-version", choices=("wrf_461", "gsd_41"),
                        help="MYNN source form; GSD 4.1 uses its Metal v4.1 chain")
    parser.add_argument("--thompson-version", choices=("wrf_461", "wrf_39_noaa"),
                        help="Thompson source form for a focused public --case")
    parser.add_argument("--noahmp-glacier-smoke", action="store_true",
                        help="route one Noah-MP cell through NOAHMP_GLACIER")
    parser.add_argument("--mp-zero-out", type=int, choices=(0, 1, 2),
                        default=0, help="enable WRF microphysics zero-out")
    parser.add_argument("--mp-zero-out-all", type=int, choices=(0, 1),
                        default=0, help="also process scalar species arrays")
    parser.add_argument("--sf-lake-physics", type=int, choices=(0, 1),
                        default=0, help="RUC lake columns on Metal")
    parser.add_argument("--alb-sol", type=int, choices=(0, 1), default=0,
                        help="RUC sun-angle land albedo on Metal")
    parser.add_argument("--spp-lsm", type=int, choices=(0, 1), default=0,
                        help="RUC stochastic soil-property perturbation on Metal")
    parser.add_argument("--spp-conv", type=int, choices=(0, 1), default=0,
                        help="GF stochastic convective perturbation on Metal; focused --case only")
    parser.add_argument("--steps", type=int, default=2,
                        help="public run_steps count (use 1 for a focused smoke)")
    parser.add_argument("--mp-zero-out-smoke", action="store_true",
                        help="run only one WDM6 Metal case with mode 2 and scalar zero-out enabled")
    parser.add_argument("--skip-pbl-pair-rows", action="store_true",
                        help="omit the default Shin-Hong/UW x sfclay 1/91 rows")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--urban", action="store_true",
                        help="also run the bounded Thompson/MYNN/Noah-MP SLUCM row")
    parser.add_argument("--urban-model", type=int, choices=(1, 2, 3),
                        help="also run one public SLUCM (1), BEP (2), or BEP-BEM (3) row")
    parser.add_argument("--only-urban-model", type=int, choices=(2, 3),
                        help="run just the admitted public BEP (2) or BEP-BEM (3) row")
    parser.add_argument("--turbulence", choices=tuple(TURBULENCE_PROFILES),
                        help="also run one public dycore turbulence profile")
    parser.add_argument("--jobs", type=int, default=2,
                        help="number of public Metal case processes (capped by WOOF_METAL_GPU_SLOTS)")
    parser.add_argument("--case-seconds", type=float, default=180.,
                        help="per-case timeout; no total sweep budget")
    parser.add_argument("--shard",
                        help="run zero-based shard i/N of the full row matrix")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pairwise", action="store_true",
                      help="run a greedy pairwise covering set sequentially in one Metal context")
    mode.add_argument("--suites", action="store_true",
                      help="run named regional suites sequentially in one Metal context")
    parser.add_argument("--_single-context-mode", choices=("pairwise", "suites"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case_seconds <= 0 or args.steps <= 0:
        parser.error("--case-seconds and --steps must be positive")
    if args.thompson_version and not args.case:
        parser.error("--thompson-version currently requires --case")
    if args.noahmp_glacier_smoke and not args.case:
        parser.error("--noahmp-glacier-smoke requires --case")
    if args.noahmp_glacier_smoke:
        try:
            glacier_combo = json.loads(args.case)
        except (TypeError, json.JSONDecodeError):
            parser.error("--noahmp-glacier-smoke requires a JSON --case")
        try:
            valid_glacier_combo = (
                isinstance(glacier_combo, list) and len(glacier_combo) == 6
                and int(glacier_combo[2]) == 4)
        except (TypeError, ValueError):
            valid_glacier_combo = False
        if not valid_glacier_combo:
            parser.error("--noahmp-glacier-smoke requires a case with LSM=4")
    if args.spp_conv:
        try:
            gf_spp_combo = json.loads(args.case)
            valid_gf_spp = (isinstance(gf_spp_combo, list)
                            and len(gf_spp_combo) == 6
                            and int(gf_spp_combo[3]) == 3)
        except (TypeError, json.JSONDecodeError, ValueError):
            valid_gf_spp = False
        if not valid_gf_spp:
            parser.error("--spp-conv requires a focused --case with CU=3")
    from woof.metal.bundle_integrity import verify_metal_bundle
    verify_metal_bundle(ROOT)
    if args.mp_zero_out_smoke:
        combo = (16, 1, 2, 0, 0, "periodic")
        with contextlib.redirect_stdout(io.StringIO()):
            status, detail = result(
                combo, only=True, mp_zero_out=2, mp_zero_out_all=1,
                steps=args.steps)
        if status == "ok" and not args.preflight_only:
            status, detail = child(
                combo, args.case_seconds, mp_zero_out=2,
                mp_zero_out_all=1, steps=args.steps)
        print(f"MP ZERO-OUT | {status} | {detail[:180]}", flush=True)
        return int(status in FAILED_STATUSES)
    if args.case:
        with contextlib.redirect_stdout(io.StringIO()):
            urban = args.urban_model if args.urban_model is not None else args.urban
            outcome = result(tuple(json.loads(args.case)),
                         only=args.preflight_only, urban=urban,
                         turbulence=args.turbulence,
                         sfclay_option=args.sfclay_option,
                         terrain_opt=args.terrain_opt,
                         mp_zero_out=args.mp_zero_out,
                         mp_zero_out_all=args.mp_zero_out_all,
                         sf_lake_physics=args.sf_lake_physics,
                         alb_sol=args.alb_sol, spp_lsm=args.spp_lsm,
                         spp_conv=args.spp_conv,
                         h_sca_adv_order=args.h_sca_adv_order,
                         steps=args.steps,
                         noahmp_glacier_smoke=args.noahmp_glacier_smoke,
                         bl_mynn_version=args.mynn_version or "wrf_461",
                         thompson_version=args.thompson_version or "wrf_461")
        print(MARKER + json.dumps(outcome))
        return int(outcome[0] in FAILED_STATUSES)
    if args.jobs <= 0:
        parser.error("jobs must be positive")
    try:
        gpu_slots = int(os.environ.get("WOOF_METAL_GPU_SLOTS", "2"))
    except ValueError:
        parser.error("WOOF_METAL_GPU_SLOTS must be a positive integer")
    if gpu_slots < 1:
        parser.error("WOOF_METAL_GPU_SLOTS must be a positive integer")
    args.jobs = min(args.jobs, gpu_slots)
    if args.pairwise or args.suites or args._single_context_mode:
        selected_mode = (args._single_context_mode
                         or ("pairwise" if args.pairwise else "suites"))
        selected = pairwise_rows() if selected_mode == "pairwise" else suite_rows()
        notes = (pairwise_exclusion_notes()
                 if selected_mode == "pairwise" else ())
        if selected_mode == "pairwise" and not selected:
            parser.error("pairwise cover has no WRF-valid executable rows")
        print(f"single-context rows: {len(selected)}", flush=True)
        if args._single_context_mode:
            return run_single_context(
                selected, output=Path(tempfile.gettempdir()) /
                ("woof_combo_pairwise.txt" if selected_mode == "pairwise"
                 else "woof_combo_suites.txt"), notes=notes)
        try:
            proc = _run_process_group(
                [sys.executable, str(Path(__file__).resolve()),
                 "--_single-context-mode", selected_mode],
                env=dict(os.environ, PYTHONPATH=str(ROOT)), timeout=285)
        except subprocess.TimeoutExpired:
            print(f"INCOMPLETE | crash | {selected_mode} exceeded 285 seconds",
                  flush=True)
            return 124
        return proc.returncode
    if args.only_urban_model is not None:
        urban_option = int(args.only_urban_model)
        combo = (8, 2, 4, 0, 0, "periodic")
        with contextlib.redirect_stdout(io.StringIO()):
            status, detail = result(
                combo, only=args.preflight_only, urban=urban_option,
                sfclay_option=1, mp_zero_out=args.mp_zero_out,
                mp_zero_out_all=args.mp_zero_out_all, steps=args.steps)
        if status == "ok" and not args.preflight_only:
            status, detail = child(
                combo, args.case_seconds, urban=urban_option,
                sfclay_option=1, mp_zero_out=args.mp_zero_out,
                mp_zero_out_all=args.mp_zero_out_all, steps=args.steps)
        print(f"URBAN {urban_option} | {status} | {detail[:180]}", flush=True)
        return int(status in FAILED_STATUSES)
    try:
        axes = [[int(x) for x in getattr(args, key).split(",")]
                for key in ("microphysics", "pbl", "land_surface", "cumulus", "radiation")]
        boundaries = args.boundaries.split(",")
        if not set(boundaries) <= {"periodic", "open", "specified"}:
            raise ValueError("boundaries must be periodic,open,specified")
    except ValueError as exc:
        parser.error(str(exc))
    rows = []
    print("MP | PBL | LSM | CU | RAD | boundary | status | detail", flush=True)
    # GF and RUC are individually gated real forecast profiles. Keep their
    # canonical rows in the public matrix while the broad Cartesian axes
    # continue to grow one admitted option at a time; keep RUC last so its
    # known thdif failure remains the final matrix result.
    requested = set(itertools.product(*axes, boundaries))
    combos = [row for row in (GF_PROFILE_ROW,) if row in requested]
    combos.extend(combo for combo in itertools.product(*axes, boundaries)
                  if combo not in (GF_PROFILE_ROW, RUC_PROFILE_ROW))
    if RUC_PROFILE_ROW in requested:
        combos.append(RUC_PROFILE_ROW)
    rows.extend(("combo", combo, {}) for combo in combos)
    if not args.skip_pbl_pair_rows:
        for pbl in (0, 11, 9):
            for sfclay_option in (1, 91):
                combo = (0, pbl, 0, 0, 0, "periodic")
                rows.append((f"PBL {pbl} | SFCLAY {sfclay_option}", combo,
                             {"sfclay_option": sfclay_option}))
    mynn_row = (8, 5, 4, 16, 1, "periodic")
    if mynn_row in requested:
        rows.append(("MYNN PBL | SFCLAY 5", mynn_row,
                     {"sfclay_option": 5}))
    p3_row = (50, 5, 4, 16, 11, "periodic")
    if p3_row in requested:
        rows.append(("P3 | MYNN PBL | SFCLAY 5 | RRTM+Dudhia",
                     p3_row, {"sfclay_option": 5}))
    if args.urban:
        rows.append(("URBAN", (8, 5, 4, 0, 0, "periodic"), {"urban": True}))
    if args.urban_model is not None:
        # Use the admitted Thompson/MYNN/Noah-MP profile also used by the
        # bounded SLUCM row; option 2/3 urban routes require its paired
        # surface-layer and no-CuPy physics selection.
        rows.append((f"URBAN {args.urban_model}",
                     (8, 5, 4, 0, 0, "periodic"),
                     {"urban": args.urban_model, "sfclay_option": 1}))
    if args.turbulence:
        rows.append((f"TURBULENCE {args.turbulence}",
                     (0, 0, 0, 0, 0, "periodic"),
                     {"turbulence": args.turbulence}))

    if args.shard is not None:
        try:
            shard_index, shard_count = (int(part) for part in args.shard.split("/", 1))
            if shard_count < 1 or not 0 <= shard_index < shard_count:
                raise ValueError
        except ValueError:
            parser.error("--shard must be a zero-based index in i/N form, e.g. 0/8")
        rows = [row for index, row in enumerate(rows)
                if index % shard_count == shard_index]
        if not rows:
            parser.error("--shard selects no rows")

    def run_row(row):
        label, combo, options = row
        options = {**options, "mp_zero_out": args.mp_zero_out,
                   "mp_zero_out_all": args.mp_zero_out_all,
                   "noahmp_glacier_smoke": args.noahmp_glacier_smoke,
                   "steps": args.steps}
        status, detail = result(combo, only=True, **options)
        if status == "ok" and not args.preflight_only:
            status, detail = child(combo, args.case_seconds, **options,
                                   terrain_opt=args.terrain_opt)
        if label == "combo":
            line = " | ".join(map(str, (*combo, status, detail[:180])))
        else:
            line = f"{label} | {status} | {detail[:180]}"
        return line, status

    # The worker pool limits simultaneous device contexts while removing the
    # former sweep-wide deadline. Results stay in matrix order for stable diffs.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        failed = False
        for line, status in pool.map(run_row, rows):
            print(line, flush=True)
            failed |= status in FAILED_STATUSES
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
