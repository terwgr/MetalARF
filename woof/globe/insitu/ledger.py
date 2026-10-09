"""The in-situ ledger runtime: the observer the step loop calls.

Per step it computes the conservation row on the device (``budgets``),
drains the physics component capture (``capture``), samples the spectral
kinetic energy at its cadence (``spectra``), and buffers everything ON THE
DEVICE.  Every ``flush_every`` steps the buffer crosses to the host in ONE
concatenated read, the rows are appended to ``insitu.ndjson``, the
tripwires are evaluated row by row in order (so the onset STEP is exact
even though the evaluation runs at the flush), and a trip writes a
checkpoint snapshot of the state at the flush plus a trip record with the
last ``history_rows`` rows and the name of the term that moved.

On the cupy backend the batched flush is the only device-to-host transfer
the instrument makes; nothing here calls ``float()`` on a device array
between flushes.
"""
from __future__ import annotations

import json
import math
import os
from collections import deque
from pathlib import Path
import time

import numpy as np

from ..checkpoint import bundle_arrays, write_checkpoint
from .budgets import (
    CONSERVED_TERMS,
    LEDGER_NAMES,
    LEDGER_TERMS,
    LedgerGeometry,
    ledger_row,
)
from .capture import CAPTURE_NAMES, ComponentCapture, attach_capture
from .energy import ENERGY_COMPONENTS, OperatorEnergyLedger
from .spectra import SpectralKineticEnergy
from .tripwires import TripwireSet

INSITU_SCHEMA = "gpuwm.arwen-global-insitu/v1"
LEDGER_NAME = "insitu.ndjson"
SNAPSHOT_PREFIX = "insitu_trip_"
REFUSED_PREFIX = "insitu_refused_"
TRIP_RECORD_SUFFIX = ".json"

_HOST_METRICS = (
    "spectral_cfl",
    "mass_fixer_log_offset",
    "global_water_fixer_kg_m2",
    "positivity_repair_water_kg_m2",
    "maximum_repaired_negative_mixing_ratio",
    "maximum_repaired_negative_number_per_kg",
    "semi_implicit_max_divergence_increment_s1",
)


def insitu_owned_files(outdir: Path) -> list[Path]:
    """Files the ledger writes into a run directory (swept on --overwrite)."""
    return [
        outdir / LEDGER_NAME,
        *sorted(outdir.glob(f"{SNAPSHOT_PREFIX}*")),
        *sorted(outdir.glob(f"{REFUSED_PREFIX}*")),
    ]


def _json_number(value) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _sanitize(value, nonfinite: list[str], path: str = ""):
    """JSON-safe copy: non-finite floats become null and are listed."""
    if isinstance(value, dict):
        return {
            key: _sanitize(child, nonfinite, f"{path}.{key}" if path else str(key))
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _sanitize(child, nonfinite, f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    number = _json_number(value)
    if number is None:
        nonfinite.append(path)
    return number


class InsituLedger:
    def __init__(
        self,
        cfg,
        model,
        outdir: str | Path,
        *,
        tripwire_overrides: dict[str, float | None] | None = None,
    ):
        self.options = cfg.insitu
        self.model = model
        self.transform = model.transform
        self.backend = model.transform.backend
        self.xp = self.backend.xp
        self.outdir = Path(outdir)
        self.path = self.outdir / LEDGER_NAME
        self.config_hash = cfg.config_hash
        self.semi_implicit_scheme = cfg.semi_implicit_scheme
        self.integrator = cfg.integrator
        self.dt_s = float(cfg.dt_s)
        self.nlev = int(model.nlev)
        self.geometry = LedgerGeometry(model.transform, model.rotation_rate_s)
        self.spectra = SpectralKineticEnergy(model.transform)
        self.capture = ComponentCapture(model.transform.grid.quadrature_weights)
        self.energy = OperatorEnergyLedger(model)
        self.tripwires = TripwireSet(
            precision=cfg.precision,
            maximum_cfl=cfg.maximum_cfl,
            enabled=self.options.tripwires,
            overrides=tripwire_overrides,
            # The CFL pre-warning is armed only where the refusal it warns
            # about exists.  The semi-Lagrangian core measures the
            # advective Courant number and does not refuse on it; its
            # refusal is the Lipschitz gate, which has its own.
            advective_cfl_refusal=not getattr(model, "semi_lagrangian", False),
        )
        self._pending: list[dict] = []
        self.history: deque[dict] = deque(maxlen=int(self.options.history_rows))
        self._last_bundle = None
        self._flushed_step: int | None = None
        # Summary state.
        self.step_rows = 0
        self.spectra_rows = 0
        self.flushes = 0
        self.first_row: dict | None = None
        self.last_row: dict | None = None
        self.max_abs_mass_fixer = 0.0
        self.max_abs_water_fixer = 0.0
        self.max_levy = 0.0
        self.max_component_change: dict[str, float] = {}
        # Per-operator column-total energy net, summed over sampled steps.
        self.energy_rows = 0
        self.energy_net_j_m2: dict[str, float] = {}
        # Per-operator per-band column kinetic net by hemisphere
        # ([northern, southern] per band label), summed over sampled steps,
        # and the band energy at the opening mark of the sampled steps
        # (summed, so the mean is the sum over energy_rows).
        self.energy_band_net_j_m2: dict[str, dict[str, list[float]]] = {}
        self.energy_band_start_j_m2: dict[str, list[float]] = {}
        self.trips: list[dict] = []
        self.snapshots: list[str] = []
        self._snapshotted: set[str] = set()
        self.refused_snapshot: str | None = None
        self.observer_wall_s = 0.0
        self.attached_to: list[str] = []
        self._write_header()

    # -- wiring -----------------------------------------------------------

    def attach(self) -> "InsituLedger":
        self.model.observer = self
        if self.model.physics is not None:
            self.attached_to = attach_capture(self.model.physics, self.capture)
        self.capture.active = (1 % int(self.options.tendencies_every) == 0)
        self.capture.begin_step()
        return self

    def detach(self) -> None:
        if getattr(self.model, "observer", None) is self:
            self.model.observer = None
        if self.model.physics is not None:
            attach_capture(self.model.physics, None)

    # -- the observer hook ------------------------------------------------

    def begin_energy_step(self, atmosphere) -> bool:
        """Called by ``MoistHybridModel.step`` before its first operator:
        opens the energy marks for this step when it is sampled and says
        whether it is."""
        next_step = int(atmosphere.step) + 1
        self.energy.active = (next_step % int(self.options.energy_every) == 0)
        self.energy.begin_step(atmosphere)
        return self.energy.active

    def mark_energy(self, name: str, atmosphere) -> None:
        self.energy.mark(name, atmosphere)

    def after_step(self, model, bundle, metrics: dict) -> None:
        """Called by ``MoistHybridModel.step`` before ``enforce``."""
        started = time.perf_counter()
        step = int(bundle.step)
        physics = self.capture.drain()
        levy = float(metrics.get("positivity_repair_water_kg_m2", 0.0))
        for half in ("first_half_physics", "second_half_physics"):
            info = metrics.get(half)
            if isinstance(info, dict):
                levy += float(info.get("exchange_clamp_water_kg_m2", 0.0))
        host_metrics = {
            name: float(metrics[name]) for name in _HOST_METRICS if name in metrics
        }
        host_metrics["levy_kg_m2"] = levy
        entry = {
            "kind": "step",
            "step": step,
            "time_s": float(bundle.time_s),
            "device": ledger_row(model, bundle, self.geometry),
            "metrics": host_metrics,
            "physics": physics,
            "spectra": None,
            "energy": self.energy.drain() or None,
        }
        self.energy.active = False
        if step % int(self.options.spectra_every) == 0:
            entry["spectra"] = self.spectra.sample(
                bundle.atmosphere.vorticity, bundle.atmosphere.divergence
            )
        self._pending.append(entry)
        self._last_bundle = bundle
        self.capture.active = ((step + 1) % int(self.options.tendencies_every) == 0)
        self.capture.begin_step()
        # flush() times itself; stop this timer first so the receipt's
        # observer seconds do not carry the flush twice (adversarial review,
        # 2026-09-01: 9.8 vs 5.6 ms/step at flush_every 1 vs 10 on T21).
        self.observer_wall_s += time.perf_counter() - started
        if step % int(self.options.flush_every) == 0:
            self.flush()

    # -- the batched flush ------------------------------------------------

    def flush(self) -> list[dict]:
        if not self._pending:
            return []
        started = time.perf_counter()
        pending = self._pending
        self._pending = []
        pieces = []
        for entry in pending:
            pieces.append(entry["device"])
            for _, _, vector in entry["physics"]:
                pieces.append(vector)
            if entry["spectra"] is not None:
                pieces.append(entry["spectra"])
            if entry["energy"] is not None:
                for _, vector in entry["energy"]:
                    pieces.append(vector)
        # Grid-sized work and reductions stay on Metal. Only the compact
        # observer vectors are read here, at the configured flush boundary;
        # ordinary NumPy/CuPy runs keep their batched concatenation path.
        from ...metal.array import MetalArray, MetalArrayModule
        if any(isinstance(piece, MetalArray) for piece in pieces):
            if all(isinstance(piece, MetalArray) for piece in pieces):
                first = pieces[0]
                context = first.buffer.context
                dtype = first.dtype
                if dtype != np.dtype(np.float64):
                    raise TypeError("Global in-situ Metal vectors must be FP64")
                if any(piece.buffer.context is not context or piece.dtype != dtype
                       for piece in pieces):
                    raise ValueError(
                        "Global in-situ Metal vectors must share FP64 residency"
                    )

                # Preserve the observer's device-resident buffering through
                # flush: pack all compact vectors with Metal blits, then make
                # one device-to-host read for JSON and tripwire evaluation.
                total = sum(piece.size for piece in pieces)
                packed = MetalArrayModule(context).empty((total,), dtype=dtype)
                cursor = 0
                pairs = []
                for piece in pieces:
                    if piece.size:
                        target = packed.view(
                            (piece.size,), dtype=dtype,
                            byte_offset=cursor * dtype.itemsize,
                        )
                        pairs.append((piece, target))
                        cursor += piece.size
                for start in range(0, len(pairs), 64):
                    context.submit_bookkeeping(pairs[start:start + 64])
                host = packed.to_numpy()
            else:
                host_pieces = [
                    piece.to_numpy().reshape(-1)
                    if isinstance(piece, MetalArray)
                    else np.asarray(self.backend.to_numpy(piece)).reshape(-1)
                    for piece in pieces
                ]
                host = np.asarray(np.concatenate(host_pieces), dtype=np.float64)
        else:
            host = np.asarray(
                self.backend.to_numpy(self.xp.concatenate(pieces)), dtype=np.float64
            )
        cursor = 0
        rows: list[dict] = []
        trips: list[dict] = []
        for entry in pending:
            width = len(LEDGER_NAMES)
            terms = {
                name: float(value)
                for name, value in zip(LEDGER_NAMES, host[cursor:cursor + width])
            }
            cursor += width
            # A list, not a dict: the ORDER the components ran in is part
            # of the record and json's sorted keys would lose it.
            physics: list[dict[str, object]] = []
            for call, component, _ in entry["physics"]:
                values = host[cursor:cursor + len(CAPTURE_NAMES)]
                cursor += len(CAPTURE_NAMES)
                physics.append({
                    "call": int(call),
                    "component": component,
                    **{name: float(value) for name, value in zip(CAPTURE_NAMES, values)},
                })
            row = {
                "kind": "step",
                "step": entry["step"],
                "time_s": entry["time_s"],
                "terms": terms,
                "metrics": entry["metrics"],
                "physics": physics,
            }
            # Consumed in the order the pieces were packed: the step row,
            # its physics records, its spectra sample, then its energy marks.
            if entry["spectra"] is not None:
                width = 2 * (self.spectra.truncation + 1) + 4 * self.nlev
                sample = {
                    "kind": "spectra",
                    "step": entry["step"],
                    "time_s": entry["time_s"],
                    "top_decile_start_degree": int(self.spectra.band_start),
                    **self.spectra.unpack(host[cursor:cursor + width], self.nlev),
                }
                cursor += width
            else:
                sample = None
            if entry["energy"] is not None:
                names = [name for name, _ in entry["energy"]]
                width = len(names) * self.energy.width
                row["energy"] = self.energy.unpack(names, host[cursor:cursor + width])
                cursor += width
            rows.append(row)
            trips.extend(self.tripwires.evaluate_row(row))
            if sample is not None:
                rows.append(sample)
                trips.extend(self.tripwires.evaluate_spectra(sample))
        if cursor != host.shape[0]:
            raise AssertionError(
                f"insitu flush consumed {cursor} of {host.shape[0]} values"
            )
        # One trip record per tripwire per flush: the first row it fired on
        # is the onset inside this batch.
        first_by_name: dict[str, dict] = {}
        for trip in trips:
            first_by_name.setdefault(trip["tripwire"], trip)
        trips = list(first_by_name.values())
        for row in rows:
            self._account(row)
        self._append(rows + trips)
        self.trips.extend(trips)
        self.flushes += 1
        self._flushed_step = pending[-1]["step"]
        for trip in trips:
            self._snapshot(trip)
        self.observer_wall_s += time.perf_counter() - started
        return rows

    def _account(self, row: dict) -> None:
        if row["kind"] == "spectra":
            self.spectra_rows += 1
            return
        self.step_rows += 1
        self.history.append(row)
        if self.first_row is None:
            self.first_row = row
        self.last_row = row
        metrics = row["metrics"]
        self.max_abs_mass_fixer = max(
            self.max_abs_mass_fixer, abs(metrics.get("mass_fixer_log_offset", 0.0))
        )
        self.max_abs_water_fixer = max(
            self.max_abs_water_fixer,
            abs(metrics.get("global_water_fixer_kg_m2", 0.0)),
        )
        self.max_levy = max(self.max_levy, abs(metrics.get("levy_kg_m2", 0.0)))
        energy = row.get("energy")
        if energy is not None:
            self.energy_rows += 1
            for name, value in energy["column_total_net"].items():
                if math.isfinite(value):
                    self.energy_net_j_m2[name] = self.energy_net_j_m2.get(name, 0.0) + value
            labels = energy["bands"]["labels"]
            for label, pair in zip(labels, energy["band_start"]["column_kinetic_j_m2"]["bands"]):
                total = self.energy_band_start_j_m2.setdefault(label, [0.0, 0.0])
                for i in range(2):
                    if math.isfinite(pair[i]):
                        total[i] += pair[i]
            for name, entry in energy["band_net"].items():
                per_band = self.energy_band_net_j_m2.setdefault(name, {})
                for label, pair in zip(labels, entry["column_kinetic_j_m2"]["bands"]):
                    total = per_band.setdefault(label, [0.0, 0.0])
                    for i in range(2):
                        if math.isfinite(pair[i]):
                            total[i] += pair[i]
        for record in row["physics"]:
            for name, value in record.items():
                if name.endswith("max_abs_change") and math.isfinite(value):
                    key = f"{record['component']}.{name}"
                    self.max_component_change[key] = max(
                        self.max_component_change.get(key, 0.0), value
                    )

    # -- files ------------------------------------------------------------

    def _write_header(self) -> None:
        header = {
            "kind": "header",
            "schema": INSITU_SCHEMA,
            "config_hash": self.config_hash,
            "truncation": int(self.transform.truncation),
            "nlev": self.nlev,
            "dt_s": self.dt_s,
            "options": self.options.identity,
            "terms": [
                {"name": name, "definition": definition}
                for name, definition in LEDGER_TERMS
            ],
            "physics_capture_stats": list(CAPTURE_NAMES),
            "energy_components": list(ENERGY_COMPONENTS),
            "energy_every": int(self.options.energy_every),
            "energy_bands": self.energy.describe_bands(),
            "tripwires": self.tripwires.describe(),
        }
        self.outdir.mkdir(parents=True, exist_ok=True)
        self._append([header])

    def _append(self, rows: list[dict]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                nonfinite: list[str] = []
                payload = _sanitize(row, nonfinite)
                if nonfinite:
                    payload["nonfinite"] = nonfinite
                stream.write(json.dumps(payload, sort_keys=True, allow_nan=False))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _write_bundle(self, path: Path, bundle) -> Path:
        try:
            return write_checkpoint(
                path, bundle, config_hash=self.config_hash,
                to_numpy=self.backend.to_numpy,
                semi_implicit_scheme=self.semi_implicit_scheme,
                integrator=self.integrator,
            )
        except ValueError as exc:
            if "non-finite" not in str(exc):
                raise
            # A refused non-finite state still deserves to be on disk; it
            # cannot carry the checkpoint contract, so it is a raw archive
            # named as such.
            raw = path.with_name(path.name.replace(".npz", ".raw.npz"))
            arrays = bundle_arrays(bundle, self.backend.to_numpy)
            with raw.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    __note__=np.asarray(
                        "non-finite state; not a checkpoint: " + str(exc)
                    ),
                    **arrays,
                )
            return raw

    def _snapshot(self, trip: dict) -> None:
        name = trip["tripwire"]
        if not self.options.snapshot_on_trip or self._last_bundle is None:
            return
        if name in self._snapshotted or len(self.snapshots) >= int(self.options.max_snapshots):
            return
        self._snapshotted.add(name)
        step = int(self._last_bundle.step)
        stem = f"{SNAPSHOT_PREFIX}{name}_step{step:08d}"
        checkpoint = self._write_bundle(self.outdir / f"{stem}.npz", self._last_bundle)
        record = {
            "schema": INSITU_SCHEMA,
            "trip": trip,
            "snapshot_step": step,
            "snapshot_lag_steps": step - int(trip["step"]),
            "checkpoint": str(checkpoint),
            "history": list(self.history),
        }
        nonfinite: list[str] = []
        payload = _sanitize(record, nonfinite)
        if nonfinite:
            payload["nonfinite"] = nonfinite
        record_path = self.outdir / f"{stem}{TRIP_RECORD_SUFFIX}"
        record_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self.snapshots.append(str(checkpoint))

    # -- lifecycle --------------------------------------------------------

    def on_failure(self, last_good_state) -> dict:
        """Flush what the failing step recorded and archive the state the
        refusal saw, if the step reached the observer before raising."""
        self.flush()
        bundle = self._last_bundle
        if bundle is not None and int(bundle.step) > int(last_good_state.step):
            path = self.outdir / f"{REFUSED_PREFIX}step{int(bundle.step):08d}.npz"
            self.refused_snapshot = str(self._write_bundle(path, bundle))
        return self.summary()

    def close(self) -> dict:
        self.flush()
        return self.summary()

    def summary(self) -> dict:
        drift: dict[str, dict[str, float | None]] = {}
        if self.first_row is not None and self.last_row is not None:
            hours = (self.last_row["time_s"] - self.first_row["time_s"]) / 3600.0
            for name in CONSERVED_TERMS:
                first = self.first_row["terms"][name]
                last = self.last_row["terms"][name]
                per_hour = (last - first) / hours if hours > 0.0 else None
                drift[name] = {
                    "first": _json_number(first),
                    "last": _json_number(last),
                    "absolute_per_hour": _json_number(per_hour) if per_hour is not None else None,
                    "relative_per_hour": (
                        _json_number(per_hour / abs(first))
                        if per_hour is not None and first not in (0.0, None) and math.isfinite(first)
                        else None
                    ),
                }
        by_name: dict[str, int] = {}
        for trip in self.trips:
            by_name[trip["tripwire"]] = by_name.get(trip["tripwire"], 0) + 1
        return {
            "schema": INSITU_SCHEMA,
            "ledger_path": str(self.path),
            "options": self.options.identity,
            "attached_physics_capture": list(self.attached_to),
            "step_rows": int(self.step_rows),
            "spectra_rows": int(self.spectra_rows),
            "flushes": int(self.flushes),
            "max_abs_mass_fixer_log_offset": _json_number(self.max_abs_mass_fixer),
            "max_abs_global_water_fixer_kg_m2": _json_number(self.max_abs_water_fixer),
            "max_levy_kg_m2": _json_number(self.max_levy),
            "max_component_max_abs_change": {
                key: _json_number(value)
                for key, value in sorted(self.max_component_change.items())
            },
            "drift_per_hour": drift,
            "energy": {
                "sampled_every": int(self.options.energy_every),
                "sampled_steps": int(self.energy_rows),
                "column_total_net_j_m2": {
                    name: _json_number(value)
                    for name, value in sorted(self.energy_net_j_m2.items())
                },
                "mean_column_total_net_j_m2_per_step": {
                    name: _json_number(value / self.energy_rows)
                    for name, value in sorted(self.energy_net_j_m2.items())
                } if self.energy_rows else {},
                # Per operator, per band: the column kinetic net summed
                # over the sampled steps, northern / southern / global.
                "bands": self.energy.describe_bands(),
                "band_column_kinetic_net_j_m2": {
                    name: {
                        label: {
                            "northern": _json_number(pair[0]),
                            "southern": _json_number(pair[1]),
                            "global": _json_number(pair[0] + pair[1]),
                        }
                        for label, pair in sorted(per_band.items())
                    }
                    for name, per_band in sorted(self.energy_band_net_j_m2.items())
                },
                "mean_band_column_kinetic_j_m2": {
                    label: {
                        "northern": _json_number(pair[0] / self.energy_rows),
                        "southern": _json_number(pair[1] / self.energy_rows),
                        "global": _json_number((pair[0] + pair[1]) / self.energy_rows),
                    }
                    for label, pair in sorted(self.energy_band_start_j_m2.items())
                } if self.energy_rows else {},
                "wall_seconds": float(self.energy.wall_s),
            },
            "first_trip": self.trips[0] if self.trips else None,
            "trip_count": int(len(self.trips)),
            "trips_by_tripwire": by_name,
            "snapshots": list(self.snapshots),
            "refused_snapshot": self.refused_snapshot,
            "tripwires": self.tripwires.describe(),
            "observer_wall_seconds": float(self.observer_wall_s),
            "capture_wall_seconds": float(self.capture.wall_s),
        }


__all__ = [
    "INSITU_SCHEMA",
    "LEDGER_NAME",
    "InsituLedger",
    "insitu_owned_files",
]
