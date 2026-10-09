"""[insitu] configuration: cadences for the in-situ ledger runtime.

The ledger is DIAGNOSTIC ONLY.  Nothing in this table enters
``config_hash`` and nothing here may change a prognostic bit (proved by
``tests/test_arwen_global_insitu.py::test_ledger_on_and_off_are_bit_identical``).
It ships enabled by default: every failure of the week of 2026-08-31 (the
day-2 jet event, the reservoir-floor death at hour 37.7, the polar sag) was
diagnosed post-mortem from three-hourly checkpoints, and the whole point of
the instrument is to have watched every step and to have snapshotted the
state at onset.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class InsituOptions:
    enabled: bool = True
    # Every step's scalar ledger row is computed on the device and buffered
    # there; every ``flush_every`` steps ONE batched device-to-host read
    # lands the buffer in insitu.ndjson.  The tripwires are evaluated at
    # the flush, so a trip snapshot can lag onset by up to flush_every - 1
    # steps; flush_every = 1 makes the snapshot step-exact at the price of
    # one host read per step (a device sync on the cupy backend).
    flush_every: int = 10
    # Spectral kinetic energy by total degree (rotational and divergent,
    # per level) from the vorticity/divergence coefficients the dycore
    # already holds - no transform is paid, so this is nearly free.
    spectra_every: int = 10
    # Per-component physics tendency capture (global-mean and max-abs
    # change of theta, qv, u, v per component per call).  Each mark copies
    # the four fields on the device (the reference suite's convective
    # adjustment and the native kernels update in place, so a
    # reference-holding capture would read zero change) - measured at 0.3%
    # of a T63/40-level step on one CPU core (2026-09-01), so every step
    # is the default.
    tendencies_every: int = 1
    # Per-operator total-energy ledger (insitu.energy): every
    # ``energy_every`` steps the atmosphere is marked around each operator
    # of the step and the per-level net change of kinetic, internal and
    # potential energy is booked.  Marks at every step cost 5.4% of a
    # T255 native IMEX step and 5.9% of a split step on the card
    # (insitu.energy records the measurement), so the default samples
    # every 50 steps: 0.11% of the run.
    energy_every: int = 50
    # Rows of ledger history written beside a trip snapshot.
    history_rows: int = 50
    # A tripwire that trips writes a checkpoint snapshot of the state at
    # the flush it tripped in (the ONSET snapshot); each named tripwire
    # snapshots at most once per run, and this caps the total so a run
    # that trips everything cannot fill the disk with checkpoints.
    snapshot_on_trip: bool = True
    max_snapshots: int = 8
    tripwires: bool = True

    def __post_init__(self) -> None:
        for name in (
            "flush_every", "spectra_every", "tendencies_every",
            "energy_every", "history_rows", "max_snapshots",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"insitu.{name} must be a positive integer")

    @property
    def identity(self) -> dict[str, object]:
        return asdict(self)


def insitu_options_from_table(table: dict) -> InsituOptions:
    allowed = set(InsituOptions.__dataclass_fields__)
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"unknown keys in [insitu]: {', '.join(unknown)}")
    kwargs = {}
    for name in ("enabled", "snapshot_on_trip", "tripwires"):
        if name in table:
            if not isinstance(table[name], bool):
                raise ValueError(f"insitu.{name} must be true or false")
            kwargs[name] = table[name]
    for name in (
        "flush_every", "spectra_every", "tendencies_every", "energy_every",
        "history_rows", "max_snapshots",
    ):
        if name in table:
            kwargs[name] = table[name]
    return InsituOptions(**kwargs)


__all__ = ["InsituOptions", "insitu_options_from_table"]
