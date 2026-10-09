"""The external analysis as a weak low-pass background constraint.

Design amendment E (2026-09-06): an external analysis (GDAS, IFS) already
contains many of the observations the cycle assimilates, so offering it
as millions of independent pseudo-observations double-counts information.
Here it is an explicit weak constraint on the largest scales only: after
the analysis increment, the control's spectral fields move a stated
fraction of the way toward the external analysis on the degrees below a
stated band and nowhere else,

    x <- x + w L (x_ext - x)

with ``L`` a per-degree low-pass (weight one at and below
``full_degree``, zero at and above ``zero_degree``, raised-cosine between;
the defaults 30 and 40 hold the constraint to wavelengths longer than
about 1,000 km) and ``w`` the weight per cycle.  Degree 0 of ln ps (the
global mean of surface pressure) is excluded: the model's mass fixer owns
it and would re-absorb any shift on the first step (the v1 door's mass
convention).

The record names the source (path, kind, the analysis cycle it holds),
its valid time, its availability time (the file's modification time, or
the time the caller states), the affected degree band, the weight and the
increment it produced, separately from the observation increment.  An
anchor whose valid time lies more than ``max_age_s`` from the analysis
instant is not applied and the record says so; the constraint holds
selected scales only and the door never claims it makes the cycle unable
to drift.

Sources: an analysis checkpoint of this configuration (``kind
checkpoint``), or a GRIB analysis decoded through the tree's own analysis
ingest at this truncation (``kind grib``: the Rust mapped engine
underneath, the same route the cold start takes).
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import SPECTRAL_FIELDS
from .da.analysis import taper_weights
from .da_control import degree_power
from .obs_table import parse_valid_time
from .state import ArwenGlobalState

ANCHOR_KINDS = ("checkpoint", "grib")


@dataclass(frozen=True)
class AnchorOptions:
    """The weak constraint's parameters.

    weight
        Fraction of the low-passed departure applied per analysis (0.1
        by default: a tenth of the way toward the external analysis on the
        constrained scales each cycle).
    full_degree, zero_degree
        The low-pass: weight one at and below ``full_degree``, zero at and
        above ``zero_degree`` (defaults 30 and 40; degree 40 is a
        wavelength of about 1,000 km).
    max_age_s
        The anchor is applied only when its valid time lies within this
        of the analysis instant (default three hours: a six-hourly
        analysis anchors the three hours either side of it).
    fields
        The spectral fields constrained (all five by default).
    """

    weight: float = 0.1
    full_degree: int = 30
    zero_degree: int = 40
    max_age_s: float = 3.0 * 3600.0
    fields: tuple[str, ...] = SPECTRAL_FIELDS

    def __post_init__(self) -> None:
        w = float(self.weight)
        if not math.isfinite(w) or not 0.0 <= w <= 1.0:
            raise ValueError("anchor weight must lie in [0, 1]")
        if int(self.full_degree) < 0 or int(self.zero_degree) <= int(self.full_degree):
            raise ValueError("anchor degrees need 0 <= full_degree < zero_degree")
        if not math.isfinite(float(self.max_age_s)) or float(self.max_age_s) <= 0.0:
            raise ValueError("anchor max_age_s must be finite and positive")
        unknown = [f for f in self.fields if f not in SPECTRAL_FIELDS]
        if unknown or not self.fields:
            raise ValueError(f"anchor fields must be a nonempty subset of {SPECTRAL_FIELDS}, got {self.fields}")
        object.__setattr__(self, "fields", tuple(self.fields))

    def identity(self) -> dict[str, object]:
        return {
            "weight": float(self.weight), "full_degree": int(self.full_degree),
            "zero_degree": int(self.zero_degree), "max_age_s": float(self.max_age_s),
            "fields": list(self.fields),
        }


@dataclass
class ExternalAnchor:
    """An external analysis in spectral form at the control's truncation."""

    kind: str
    source: str
    valid_utc: dt.datetime
    available_utc: dt.datetime | None
    fields: dict[str, np.ndarray]
    identity: dict[str, object]

    def record(self) -> dict[str, object]:
        return {
            "kind": self.kind, "source": self.source,
            "valid_utc": self.valid_utc.isoformat(timespec="seconds"),
            "available_utc": None if self.available_utc is None else self.available_utc.isoformat(timespec="seconds"),
            **self.identity,
        }


def _file_time(path: Path) -> dt.datetime:
    return dt.datetime.fromtimestamp(os.path.getmtime(path), tz=dt.timezone.utc)


def load_anchor(cfg, transform, spec: dict[str, str], *, scratch_destination=None) -> ExternalAnchor:
    """An anchor from a ``--anchor`` spelling's options: ``path`` (a
    checkpoint of this configuration or a GRIB analysis), ``kind``
    (``checkpoint`` or ``grib``; default from the suffix), ``valid_utc``
    (required for a GRIB, read from the checkpoint's chain or stated for
    a checkpoint), ``mapping`` (the GRIB's analysis mapping, default the
    config's), ``available_utc`` (default the file's modification time)."""
    path = Path(spec.get("path", ""))
    if not path.is_file():
        raise FileNotFoundError(f"anchor {path} does not exist")
    kind = spec.get("kind") or ("checkpoint" if path.suffix == ".npz" else "grib")
    if kind not in ANCHOR_KINDS:
        raise ValueError(f"anchor kind must be one of {ANCHOR_KINDS}")
    valid = parse_valid_time(spec["valid_utc"]) if spec.get("valid_utc") else None
    available = parse_valid_time(spec["available_utc"]) if spec.get("available_utc") else _file_time(path)
    to_numpy = transform.backend.to_numpy
    if kind == "checkpoint":
        from .checkpoint import read_checkpoint, state_from_checkpoint

        metadata, arrays = read_checkpoint(
            path, expected_config_hash=cfg.config_hash,
            semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator,
        )
        state = state_from_checkpoint(metadata, arrays, transform.backend)
        if valid is None:
            raise ValueError(
                "an anchor checkpoint needs valid_utc (the instant its state stands "
                "for), because a checkpoint carries model time, not a calendar instant"
            )
        fields = {name: np.asarray(to_numpy(getattr(state.atmosphere, name)), dtype=np.complex128)
                  for name in SPECTRAL_FIELDS}
        identity = {"checkpoint_self_sha256": metadata["self_sha256"], "step": int(metadata["step"]),
                    "time_s": float(metadata["time_s"])}
    else:
        if valid is None:
            raise ValueError("an anchor GRIB needs valid_utc (its analysis cycle instant)")
        from .analysis_initial import analysis_initial_state

        anchor_cfg = dataclasses.replace(
            cfg, initial_mode="analysis", analysis_grib=str(path),
            analysis_mapping=spec.get("mapping") or cfg.analysis_mapping or "gdas-global",
        )
        state, _terrain, provenance = analysis_initial_state(
            anchor_cfg, transform, scratch_destination=scratch_destination,
        )
        fields = {name: np.asarray(to_numpy(getattr(state.atmosphere, name)), dtype=np.complex128)
                  for name in SPECTRAL_FIELDS}
        identity = {"mapping": anchor_cfg.analysis_mapping,
                    "grib_sha256": (provenance or {}).get("sha256") if isinstance(provenance, dict) else None}
    return ExternalAnchor(kind=kind, source=str(path), valid_utc=valid, available_utc=available,
                          fields=fields, identity=identity)


def parse_anchor_spec(text: str) -> dict[str, str]:
    """``PATH`` or ``PATH:key=value;key=value`` (the door's stream
    spelling); ``path`` may also be given as an option."""
    head, _, options_text = text.partition(":")
    # A Windows drive letter is one character before the colon.
    if len(head) == 1 and options_text and (options_text.startswith("\\") or options_text.startswith("/")):
        rest, _, options_text = options_text.partition(":")
        head = f"{head}:{rest}"
    options: dict[str, str] = {}
    for part in filter(None, options_text.split(";")):
        if "=" not in part:
            raise ValueError(f"anchor option {part!r} is not key=value")
        key, value = part.split("=", 1)
        options[key.strip()] = value.strip()
    if head and "path" not in options:
        options["path"] = head
    return options


def anchor_options_from_spec(options: dict[str, str]) -> AnchorOptions:
    kwargs: dict[str, object] = {}
    if "weight" in options:
        kwargs["weight"] = float(options["weight"])
    if "full_degree" in options:
        kwargs["full_degree"] = int(options["full_degree"])
    if "zero_degree" in options:
        kwargs["zero_degree"] = int(options["zero_degree"])
    if "max_age_s" in options:
        kwargs["max_age_s"] = float(options["max_age_s"])
    if "fields" in options:
        kwargs["fields"] = tuple(filter(None, options["fields"].split(",")))
    return AnchorOptions(**kwargs)


def low_pass_weights(truncation: int, options: AnchorOptions) -> np.ndarray:
    return taper_weights(int(truncation), int(options.full_degree), int(options.zero_degree))


def apply_anchor(
    state: ArwenGlobalState, model, transform, anchor: ExternalAnchor, options: AnchorOptions,
    *, analysis_time: dt.datetime,
) -> tuple[ArwenGlobalState, dict[str, object]]:
    """``x <- x + w L (x_ext - x)`` on the constrained fields and degrees;
    degree 0 of ln ps excluded.  Returns ``(state, record)``; when the
    anchor is out of its age window the state is returned unchanged and
    the record says why."""
    backend = transform.backend
    xp = backend.xp
    to_numpy = backend.to_numpy
    t = int(transform.truncation)
    age_s = abs((analysis_time - anchor.valid_utc).total_seconds())
    base = {
        "source": anchor.record(),
        "options": options.identity(),
        "analysis_time_utc": analysis_time.isoformat(timespec="seconds"),
        "age_s": float(age_s),
        "affected_band": {
            "full_degree": int(options.full_degree), "zero_degree": int(options.zero_degree),
            "note": "weight one at and below full_degree, zero at and above zero_degree, cos^2 between; "
                    "degree 0 of ln ps excluded (the mass fixer owns the global mean)",
        },
        "claim": (
            "constrains the listed degrees of the listed fields by the stated fraction "
            "per analysis; it does not make the cycle unable to drift on other scales"
        ),
    }
    if age_s > float(options.max_age_s):
        return state, {**base, "applied": False,
                       "reason": f"the anchor is {age_s:g} s from the analysis instant, beyond max_age_s {options.max_age_s:g}"}
    weights = low_pass_weights(t, options)
    w_dev = xp.asarray(weights)
    fields = list(state.atmosphere.fields())
    increments: dict[str, object] = {}
    departure_power: dict[str, object] = {}
    for index, name in enumerate(SPECTRAL_FIELDS):
        if name not in options.fields:
            continue
        ext = anchor.fields.get(name)
        if ext is None:
            continue
        current = fields[index]
        ext_dev = xp.asarray(ext, dtype=current.dtype)
        if ext_dev.shape != current.shape:
            raise ValueError(
                f"anchor field {name} has shape {ext_dev.shape}, the state {current.shape}: "
                "the anchor must be at the control's truncation and level count"
            )
        departure = ext_dev - current
        low = departure * w_dev[:, None].astype(current.real.dtype)
        if name == "log_surface_pressure":
            low[..., 0, 0] = 0.0
        increment = float(options.weight) * low
        fields[index] = current + increment
        increments[name] = increment
        power = degree_power(np.asarray(to_numpy(departure), dtype=np.complex128))
        constrained = float(np.sum(power * weights))
        departure_power[name] = {
            "total": float(power.sum()),
            "in_constrained_band": constrained,
            "fraction_in_band": (constrained / float(power.sum())) if power.sum() > 0 else 0.0,
        }
    moved = any(bool(xp.any(inc != 0)) for inc in increments.values())
    if not moved:
        # Nothing to add: the state is returned as it is (bitwise), so an
        # anchor equal to the background leaves no trace but its record.
        new_state = state
    else:
        new_state = ArwenGlobalState(state.atmosphere.with_fields(fields), state.surface, state.physics_state.copy())
        if "qv" in increments and bool(xp.any(increments["qv"] != 0)):
            new_state, _n, _t, _f = model._repair_positivity(new_state)
        model.enforce(new_state)
        model.release_syntheses()
    increment_rms = {
        name: float(math.sqrt(degree_power(np.asarray(to_numpy(inc), dtype=np.complex128)).sum()))
        for name, inc in increments.items()
    }
    return new_state, {
        **base, "applied": True,
        "weights_per_degree": [float(w) for w in weights],
        "departure_power": departure_power,
        "increment_grid_rms": increment_rms,
    }


__all__ = [
    "ANCHOR_KINDS",
    "AnchorOptions",
    "ExternalAnchor",
    "anchor_options_from_spec",
    "apply_anchor",
    "load_anchor",
    "low_pass_weights",
    "parse_anchor_spec",
]
