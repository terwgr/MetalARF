"""The door that estimates the hybrid filter's static covariance:
``woof global da static-covariance``.

The lagged-forecast pairs the caller names (the checkpoint of the longer
forecast and of the shorter one valid at the same instant, the 24 h and
12 h forecasts from consecutive analyses in the sample of record) are
differenced in the model's own spectral variables, the table is estimated
(:func:`woof.globe.da.static_covariance.estimate_static_covariance`)
on the config's transform and level ladder, written with its receipt
(every pair's files and hashes, the instants, the balanced shares, the
config hash) and charted: the per-degree variance spectra of the control
variables, the vertical correlations per band and the balanced share of
each variable's variance per band.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .checkpoint import read_checkpoint
from .config import ArwenGlobalConfig
from .da.static_covariance import (
    CONTROL_VARIABLES,
    STATIC_COVARIANCE_SCHEMA,
    StaticCovariance,
    estimate_static_covariance,
    lagged_pair_differences,
)

TABLE_NAME = "static-covariance.npz"
RECEIPT_NAME = "static-covariance-receipt.json"
RECEIPT_SCHEMA = "gpuwm.arwen-global-static-covariance-receipt/v1"

__all__ = ["RECEIPT_NAME", "TABLE_NAME", "estimate_table", "reference_p_full_hpa", "static_covariance_charts"]


def reference_p_full_hpa(cfg: ArwenGlobalConfig, ps_pa: float = 101325.0) -> np.ndarray:
    a = np.asarray(cfg.vertical.a_half_pa, dtype=np.float64)
    b = np.asarray(cfg.vertical.b_half, dtype=np.float64)
    p_half = a + b * float(ps_pa)
    return np.sqrt(p_half[:-1] * p_half[1:]) / 100.0


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def static_covariance_charts(table: StaticCovariance, out_dir: Path) -> list[str]:
    """The table's charts (analysis charts, matplotlib): variance spectra,
    vertical correlations per band, balanced shares.  Returns the files."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    n = np.arange(table.truncation + 1)
    p = table.reference_p_full_hpa
    labels = {
        "vorticity": "vorticity (s^-2)", "divergence_unbalanced": "unbalanced divergence (s^-2)",
        "theta_unbalanced": "unbalanced theta (K^2)", "qv": "vapor (kg^2/kg^2)",
        "log_surface_pressure_unbalanced": "unbalanced ln ps",
    }
    # 1. variance spectra, a few levels each.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, name in zip(axes.ravel(), CONTROL_VARIABLES):
        v = table.variance[name]
        if v.ndim == 1:
            ax.loglog(n[1:], v[1:], color="black")
        else:
            picks = np.unique(np.clip(np.round(np.linspace(0, table.nlev - 1, 5)).astype(int), 0, table.nlev - 1))
            for k in picks:
                label = f"{p[k]:.0f} hPa" if np.isfinite(p[k]) else f"level {k}"
                ax.loglog(n[1:], np.maximum(v[1:, k], 1e-300), label=label)
            ax.legend(fontsize=7)
        ax.set_title(labels[name], fontsize=9)
        ax.set_xlabel("total wavenumber n")
        ax.set_ylabel("E|c_nm|^2 per degree")
    axes.ravel()[-1].axis("off")
    share = table.receipt.get("balance", {}).get("balanced_share_of_variance", {})
    ax = axes.ravel()[-1]
    ax.axis("on")
    width = 0.25
    x = np.arange(len(table.bands))
    for i, key in enumerate(("theta", "divergence", "log_surface_pressure")):
        vals = [np.nan if s is None else s for s in share.get(key, [None] * len(x))]
        ax.bar(x + (i - 1) * width, vals, width, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}-{b}" for a, b in table.bands], rotation=45, fontsize=7)
    ax.set_ylabel("balanced share of variance")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title("balanced share per wavenumber band", fontsize=9)
    fig.suptitle(f"static covariance T{table.truncation}, {table.receipt.get('samples')} lagged pairs")
    fig.tight_layout()
    path = out_dir / "static-covariance-spectra.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    files.append(str(path))
    # 2. vertical correlations per band for theta_u and vorticity.
    for name in ("theta_unbalanced", "vorticity", "qv"):
        nb = len(table.bands)
        cols = min(nb, 4)
        rows = int(np.ceil(nb / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), squeeze=False)
        for b in range(rows * cols):
            ax = axes.ravel()[b]
            if b >= nb:
                ax.axis("off")
                continue
            im = ax.imshow(table.vertical_correlation[name][b], vmin=-1, vmax=1, cmap="RdBu_r", origin="lower")
            ax.set_title(f"degrees {table.bands[b][0]}-{table.bands[b][1]}", fontsize=8)
            ax.set_xlabel("level")
            ax.set_ylabel("level")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
        fig.suptitle(f"vertical correlation of {labels[name]} per band")
        path = out_dir / f"static-covariance-vertical-correlation-{name}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        files.append(str(path))
    # 3. the theta-on-Phi regression for the first bands.
    fig, axes = plt.subplots(1, min(3, len(table.bands)), figsize=(4.5 * min(3, len(table.bands)), 4), squeeze=False)
    for b, ax in enumerate(axes.ravel()):
        m = table.theta_on_phi[b]
        lim = float(np.abs(m).max()) or 1.0
        im = ax.imshow(m, vmin=-lim, vmax=lim, cmap="RdBu_r", origin="lower")
        ax.set_title(f"theta on Phi_b, degrees {table.bands[b][0]}-{table.bands[b][1]}", fontsize=8)
        ax.set_xlabel("Phi_b level")
        ax.set_ylabel("theta level")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = out_dir / "static-covariance-balance-regression.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    files.append(str(path))
    return files


def estimate_table(
    cfg: ArwenGlobalConfig, pairs, out: str | Path, *, version: str | None = None, ridge: float = 1.0e-3,
    charts: bool = True, overwrite: bool = False, progress=None, config_path: str | Path | None = None,
    backend: str | None = None, precision: str = "float64",
) -> dict:
    """The door: the table from the pairs under ``cfg``, written to
    ``out`` with its receipt and charts.  Returns the receipt.  The
    transform is the config's (truncation, grid, dealiasing) on ``backend``
    (the config's when None) at ``precision`` (float64 by default: the
    estimation is a statistics job on differences a few thousandths of the
    fields, and the balance operator's two transforms and Laplacians are
    not to lose digits to the run's single precision)."""
    start = time.perf_counter()
    out = Path(out)
    table_path = out / TABLE_NAME
    if table_path.exists() and not overwrite:
        raise FileExistsError(f"{table_path} exists; pass --overwrite to replace it")
    pairs = [(Path(a), Path(b)) for a, b in pairs]
    if len(pairs) < 2:
        raise ValueError("a static covariance needs at least two lagged pairs")
    for later, earlier in pairs:
        for path in (later, earlier):
            if not path.is_file():
                raise FileNotFoundError(f"checkpoint {path} does not exist")
    from woof.globe.spectral.transform import SphericalHarmonicTransform

    transform = SphericalHarmonicTransform.create(
        cfg.truncation, nlat=cfg.nlat, nlon=cfg.nlon, dealias_factor=cfg.dealias_factor,
        backend=cfg.backend if backend is None else backend, precision=str(precision),
        tensor_core_contractions=False,
    )
    from woof.globe.spectral.vector import VorticityDivergenceOperator

    vector = VorticityDivergenceOperator(transform)
    say = progress or (lambda line: None)
    differences = []
    instants = []
    for record, diff in lagged_pair_differences(pairs, read=lambda p: read_checkpoint(p)):
        shape = diff["theta"].shape
        if shape[-1] != transform.truncation + 1:
            raise ValueError(
                f"pair {record['later']['path']} is T{shape[-1] - 1} and the config T{transform.truncation}"
            )
        if shape[0] != cfg.vertical.nlev:
            raise ValueError(f"pair {record['later']['path']} has {shape[0]} levels and the config {cfg.vertical.nlev}")
        for key in ("later", "earlier"):
            record[key]["sha256"] = _digest(Path(record[key]["path"]))
        differences.append((record, diff))
        instants.append(record["later"]["time_s"])
        say(f"pair {len(differences)} of {len(pairs)}: {Path(record['later']['path']).name} minus "
            f"{Path(record['earlier']['path']).name}")
    table = estimate_static_covariance(
        differences, transform, vector, ridge=ridge, reference_p_full_hpa=reference_p_full_hpa(cfg), progress=say)
    label = version or f"lagged-{len(pairs)}-pairs"
    table.receipt.update({
        "version": label,
        "config_hash": cfg.config_hash,
        "config": None if config_path is None else str(config_path),
        "transform": dict(transform.identity) if isinstance(getattr(transform, "identity", None), dict) else None,
        "vertical": cfg.vertical.describe() if hasattr(cfg.vertical, "describe") else None,
        "made_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "door": "woof global da static-covariance",
    })
    out.mkdir(parents=True, exist_ok=True)
    table.save(table_path)
    chart_files = static_covariance_charts(table, out / "charts") if charts else []
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "door": "woof global da static-covariance",
        "table": str(table_path),
        "sha256": table.sha256(),
        "table_schema": STATIC_COVARIANCE_SCHEMA,
        "truncation": int(table.truncation),
        "nlev": int(table.nlev),
        "samples": int(table.receipt["samples"]),
        "version": label,
        "bands": table.receipt["bands"],
        "pairs": table.receipt["pairs"],
        "balanced_share_of_variance": table.receipt["balance"]["balanced_share_of_variance"],
        "variance_summary": {
            name: {
                "total": float(np.sum((2.0 * np.arange(table.truncation + 1) + 1.0)[:, None] * table.variance[name]
                                      if table.variance[name].ndim == 2 else
                                      (2.0 * np.arange(table.truncation + 1) + 1.0) * table.variance[name]) / (4.0 * np.pi)),
            } for name in CONTROL_VARIABLES
        },
        "charts": chart_files,
        "config_hash": cfg.config_hash,
        "wall_seconds": time.perf_counter() - start,
        "status": "pass",
    }
    receipt_path = out / RECEIPT_NAME
    temporary = receipt_path.with_name(f".{receipt_path.name}.partial-{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt_path)
    receipt["receipt_path"] = str(receipt_path)
    return receipt
