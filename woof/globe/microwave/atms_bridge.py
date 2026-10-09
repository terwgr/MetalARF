"""Python side of the ``rw_atms`` bridge.

``rw_atms`` (``tools/rustwx/crates/rw-atms``) is the only thing that opens
an ATMS SDR HDF5 file anywhere in this tree.  This module resolves the
binary the same way :mod:`woof.netcdf_bridge` resolves ``rw_netcdf``
(an environment override, the crate's target directories, the bundled
and staged bridge directories), drives its three passes, and maps the
flat little-endian arrays it writes with numpy.

Refusals name the breakage: a missing binary is reported with the build
one-liner, a schema drift with the schema it found, a short array with
the byte counts.  Nothing here falls back to a Python decoder, because a
Python decoder for an observation stream is a release blocker by the
Python-boundary law, not a shortcut.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from woof.bridges import (
    RUSTWX_CRATE_RELATIVE,
    cargo_build_one_liner,
    default_bridge_dir,
    executable_name,
    packaged_bridge_dir,
)

ATMS_NAME = "rw_atms"
ATMS_ENV = "WOOF_RW_ATMS"
DECODE_SCHEMA = "gpuwm-rw-atms-decode-v1"
THIN_SCHEMA = "gpuwm-rw-atms-thin-v1"
INVENTORY_SCHEMA = "gpuwm-rw-atms-inventory-v1"
ABI_MARKER = "gpuwm-rw-atms-decode-v1"


class AtmsBridgeMissing(RuntimeError):
    """No ``rw_atms`` binary could be found."""


class AtmsDecodeError(RuntimeError):
    """``rw_atms`` refused or returned something other than its contract."""


def _crate_dir() -> Path:
    return Path(__file__).resolve().parents[3] / RUSTWX_CRATE_RELATIVE


def atms_candidates() -> tuple[Path, ...]:
    filename = executable_name(ATMS_NAME)
    candidates: list[Path] = []
    override = os.environ.get(ATMS_ENV)
    if override:
        candidates.append(Path(override))
    root = Path(__file__).resolve().parents[3]
    candidates.extend((
        _crate_dir() / "target" / "release" / filename,
        _crate_dir() / "target" / "debug" / filename,
        root / "libexec" / "bridges" / filename,
        packaged_bridge_dir() / filename,
        default_bridge_dir() / filename,
    ))
    return tuple(candidates)


def find_atms_bin() -> Path | None:
    override = os.environ.get(ATMS_ENV)
    if override and not Path(override).is_file():
        raise AtmsBridgeMissing(
            f"{ATMS_ENV} names a missing file: {override}.  Point it at a built "
            f"{ATMS_NAME} binary, or unset {ATMS_ENV} to search the usual places."
        )
    for candidate in atms_candidates():
        if candidate.is_file():
            return candidate
    return None


def atms_remedy() -> str:
    return (
        f"build it with `{cargo_build_one_liner(RUSTWX_CRATE_RELATIVE)} -p rw-atms` "
        f"or set {ATMS_ENV} to a built {ATMS_NAME}"
    )


def resolve_atms_bin() -> Path:
    found = find_atms_bin()
    if found is None:
        searched = "\n  ".join(str(path) for path in atms_candidates())
        raise AtmsBridgeMissing(
            f"{ATMS_NAME} not found; searched:\n  {searched}\n{atms_remedy()}"
        )
    return found


def _run(arguments: list[str], *, what: str, executable: Path | None = None) -> str:
    binary = executable or resolve_atms_bin()
    completed = subprocess.run(
        [str(binary), *arguments], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise AtmsDecodeError(
            f"{ATMS_NAME} {what} failed (exit {completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout


def abi_matches(executable: Path | None = None) -> bool:
    return ABI_MARKER in _run(["--abi"], what="--abi", executable=executable)


def inventory(path: str | Path, *, executable: Path | None = None) -> dict:
    document = json.loads(_run(["inventory", str(path)], what="inventory", executable=executable))
    if document.get("schema") != INVENTORY_SCHEMA:
        raise AtmsDecodeError(
            f"inventory schema {document.get('schema')!r} is not {INVENTORY_SCHEMA}"
        )
    return document


def _read_flat(directory: Path, record: dict) -> np.ndarray:
    path = directory / record["filename"]
    shape = tuple(int(n) for n in record["shape"])
    expected = int(np.prod(shape)) * np.dtype(record["dtype"]).itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise AtmsDecodeError(
            f"{path} holds {actual} bytes, the receipt says {expected} for shape {shape}"
        )
    return np.fromfile(path, dtype=np.dtype(record["dtype"])).reshape(shape)


@dataclass(frozen=True)
class Decoded:
    """One decode pass: the arrays and the receipt."""

    directory: Path
    metadata: dict
    brightness_temperature_k: np.ndarray  # (nscan, 96, 22)
    latitude_deg: np.ndarray  # (nscan, 96)
    longitude_deg: np.ndarray
    satellite_zenith_deg: np.ndarray
    satellite_azimuth_deg: np.ndarray
    solar_zenith_deg: np.ndarray
    beam_time_unix_s: np.ndarray  # (nscan, 96) float64
    granule_index: np.ndarray  # (nscan,)


def read_decoded(directory: str | Path) -> Decoded:
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != DECODE_SCHEMA:
        raise AtmsDecodeError(
            f"{directory / 'metadata.json'}: schema {metadata.get('schema')!r} is not {DECODE_SCHEMA}"
        )
    arrays = metadata["arrays"]
    return Decoded(
        directory=directory,
        metadata=metadata,
        brightness_temperature_k=_read_flat(directory, arrays["brightness_temperature_k"]),
        latitude_deg=_read_flat(directory, arrays["latitude_deg"]),
        longitude_deg=_read_flat(directory, arrays["longitude_deg"]),
        satellite_zenith_deg=_read_flat(directory, arrays["satellite_zenith_deg"]),
        satellite_azimuth_deg=_read_flat(directory, arrays["satellite_azimuth_deg"]),
        solar_zenith_deg=_read_flat(directory, arrays["solar_zenith_deg"]),
        beam_time_unix_s=_read_flat(directory, arrays["beam_time_unix_s"]),
        granule_index=_read_flat(directory, arrays["granule_index"]),
    )


def decode(pairs: list[tuple[str | Path, str | Path]], outdir: str | Path, *,
           executable: Path | None = None) -> Decoded:
    """Decode SDR/GEO granule pairs into ``outdir`` and read them back."""
    if not pairs:
        raise ValueError("decode needs at least one SDR/GEO pair")
    arguments = ["decode", str(outdir)]
    for sdr, geo in pairs:
        arguments.extend((str(sdr), str(geo)))
    _run(arguments, what="decode", executable=executable)
    return read_decoded(outdir)


@dataclass(frozen=True)
class Thinned:
    """One thinning pass: cell means with counts and spreads."""

    directory: Path
    metadata: dict
    cell_bin: np.ndarray
    cell_j: np.ndarray
    cell_i: np.ndarray
    count: np.ndarray
    tb_mean_k: np.ndarray  # (ncell, 22)
    tb_std_k: np.ndarray
    tb_count: np.ndarray
    lat_mean_deg: np.ndarray
    lon_mean_deg: np.ndarray
    zenith_mean_deg: np.ndarray
    azimuth_mean_deg: np.ndarray
    solar_zenith_mean_deg: np.ndarray
    scan_angle_abs_mean_deg: np.ndarray
    time_mean_unix_s: np.ndarray

    @property
    def ncell(self) -> int:
        return int(self.cell_bin.size)


def read_thinned(directory: str | Path) -> Thinned:
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != THIN_SCHEMA:
        raise AtmsDecodeError(
            f"{directory / 'metadata.json'}: schema {metadata.get('schema')!r} is not {THIN_SCHEMA}"
        )
    arrays = metadata["arrays"]
    values = {name: _read_flat(directory, record) for name, record in arrays.items()}
    return Thinned(directory=directory, metadata=metadata, **values)


def thin(decoded_dir: str | Path, outdir: str | Path, *, latitudes_deg, nlon: int,
         origin_unix_s: float, bin_s: float, max_zenith_deg: float = 90.0,
         executable: Path | None = None) -> Thinned:
    """Colocate a decoded pass onto latitude rings x ``nlon`` longitudes x
    time bins.  ``latitudes_deg`` is written to ``outdir/latitudes.txt``
    for the binary and for the record."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rings = np.asarray(latitudes_deg, dtype=np.float64)
    if rings.ndim != 1 or rings.size < 2:
        raise ValueError("latitudes_deg must be a 1-D array of at least two rings")
    ring_file = outdir / "latitudes.txt"
    ring_file.write_text("\n".join(f"{value:.10f}" for value in rings) + "\n", encoding="utf-8")
    _run(
        [
            "thin", str(outdir), str(decoded_dir),
            "--latitudes", str(ring_file),
            "--nlon", str(int(nlon)),
            "--origin-unix-s", repr(float(origin_unix_s)),
            "--bin-s", repr(float(bin_s)),
            "--max-zenith", repr(float(max_zenith_deg)),
        ],
        what="thin",
        executable=executable,
    )
    return read_thinned(outdir)
