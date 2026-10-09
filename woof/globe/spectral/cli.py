"""CLI for the standalone Level-3 global spectral research core."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from .checkpoint import read_checkpoint
from .compression import (
    SCALAR_SCHEMA,
    WIND_SCHEMA,
    compress_scalar,
    compress_wind,
    decode_scalar,
    decode_wind,
    read_compressed_scalar,
    read_compressed_wind,
    write_compressed_scalar,
    write_compressed_wind,
)
from .config import load_config
from .export import export_checkpoint_latlon, read_latlon_export
from .pins import pins_receipt
from .receipt import check_receipt
from .runner import build_model_and_state, build_transform, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m woof.globe.spectral",
        description=(
            "Research-only global spherical-harmonic shallow-water and dry "
            "primitive-equation prototype."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("pins", help="print the committed Level-3 arithmetic identity")

    check = sub.add_parser(
        "transform-check", help="run analysis/synthesis and Parseval controls"
    )
    check.add_argument("--truncation", type=int, default=15)
    check.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    check.add_argument(
        "--precision", choices=("float32", "float64"), default="float64"
    )
    check.add_argument("--dealias-factor", type=float, default=1.5)
    check.add_argument("--seed", type=int, default=7)

    runp = sub.add_parser("run", help="integrate a registered TOML experiment")
    runp.add_argument("config", type=Path)
    runp.add_argument("--outdir", type=Path, default=Path("out/global-spectral"))
    runp.add_argument("--restart", type=Path, default=None)
    runp.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only Level-3-owned files in --outdir",
    )

    bench = sub.add_parser("benchmark", help="measure transform and one RHS evaluation")
    bench.add_argument("config", type=Path)
    bench.add_argument("--iterations", type=int, default=5)
    bench.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="unmeasured steady-state warmup calls after the separately timed first call",
    )

    inspect = sub.add_parser("inspect", help="validate and print checkpoint metadata")
    inspect.add_argument("checkpoint", type=Path)

    receipt = sub.add_parser("check-receipt", help="validate a self-hashed run receipt")
    receipt.add_argument("receipt", type=Path)

    scalar = sub.add_parser(
        "compress-scalar",
        help="compress one Gaussian-grid scalar from an NPZ into SH coefficients",
    )
    scalar.add_argument("config", type=Path)
    scalar.add_argument("input", type=Path)
    scalar.add_argument("output", type=Path)
    scalar.add_argument("--field", required=True)
    scalar.add_argument("--space", choices=("linear", "log"), default="linear")
    scalar.add_argument("--floor", type=float, default=1.0e-20)
    scalar.add_argument("--keep-degree", type=int, default=None)
    scalar.add_argument("--no-preserve-mean", action="store_true")
    scalar.add_argument("--overwrite", action="store_true")

    dscalar = sub.add_parser(
        "decompress-scalar",
        help="decode a compressed scalar into a portable NPZ",
    )
    dscalar.add_argument("config", type=Path)
    dscalar.add_argument("input", type=Path)
    dscalar.add_argument("output", type=Path)
    dscalar.add_argument("--field", default="field")
    dscalar.add_argument("--keep-degree", type=int, default=None)
    dscalar.add_argument("--overwrite", action="store_true")

    wind = sub.add_parser(
        "compress-wind",
        help="compress Gaussian-grid east/north wind as vorticity/divergence",
    )
    wind.add_argument("config", type=Path)
    wind.add_argument("input", type=Path)
    wind.add_argument("output", type=Path)
    wind.add_argument("--u-field", default="u")
    wind.add_argument("--v-field", default="v")
    wind.add_argument("--keep-degree", type=int, default=None)
    wind.add_argument("--overwrite", action="store_true")

    dwind = sub.add_parser(
        "decompress-wind",
        help="decode compressed vorticity/divergence wind into an NPZ",
    )
    dwind.add_argument("config", type=Path)
    dwind.add_argument("input", type=Path)
    dwind.add_argument("output", type=Path)
    dwind.add_argument("--u-field", default="u")
    dwind.add_argument("--v-field", default="v")
    dwind.add_argument("--keep-degree", type=int, default=None)
    dwind.add_argument("--overwrite", action="store_true")

    compressed = sub.add_parser(
        "inspect-compressed", help="validate and print compressed-field metadata"
    )
    compressed.add_argument("input", type=Path)

    export = sub.add_parser(
        "export-latlon",
        help="sample a checkpoint onto a hash-bound regular global lat/lon grid",
    )
    export.add_argument("config", type=Path)
    export.add_argument("checkpoint", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--nlat", type=int, required=True)
    export.add_argument("--nlon", type=int, required=True)
    export.add_argument("--overwrite", action="store_true")

    iexport = sub.add_parser(
        "inspect-export", help="validate and print a regular lat/lon export"
    )
    iexport.add_argument("input", type=Path)
    return parser


def _benchmark(config: Path, iterations: int, warmup: int) -> dict:
    cfg = load_config(config)
    transform = build_transform(cfg)
    model, state, _cold, _restart = build_model_and_state(cfg, transform)

    def transform_call():
        grid = transform.inverse(state.fields()[0])
        transform.forward(grid)

    transform.backend.synchronize()
    start = time.perf_counter()
    transform_call()
    transform.backend.synchronize()
    transform_first_s = time.perf_counter() - start
    for _ in range(warmup):
        transform_call()
    transform.backend.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        transform_call()
    transform.backend.synchronize()
    transform_s = (time.perf_counter() - start) / iterations

    transform.backend.synchronize()
    start = time.perf_counter()
    model.rhs(state)
    transform.backend.synchronize()
    rhs_first_s = time.perf_counter() - start
    for _ in range(warmup):
        model.rhs(state)
    transform.backend.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        model.rhs(state)
    transform.backend.synchronize()
    rhs_s = (time.perf_counter() - start) / iterations
    return {
        "model": cfg.model,
        "truncation": cfg.truncation,
        "nlat": transform.grid.nlat,
        "nlon": transform.grid.nlon,
        "backend": cfg.backend,
        "precision": cfg.precision,
        "transform_first_seconds": transform_first_s,
        "transform_roundtrip_steady_seconds": transform_s,
        "rhs_first_seconds": rhs_first_s,
        "rhs_steady_seconds": rhs_s,
        "iterations": iterations,
        "warmup": warmup,
    }


def _require_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output {path} exists; pass --overwrite to replace it")


def _npz_field(path: Path, name: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if name not in archive:
            raise ValueError(
                f"NPZ {path} has no field {name!r}; available: {sorted(archive.files)}"
            )
        return np.array(archive[name], copy=True)


def _write_npz(path: Path, overwrite: bool, **fields) -> None:
    _require_output(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **fields)


def _compressed_header(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if "__header__" not in archive:
            raise ValueError(f"compressed archive {path} has no header")
        return json.loads(str(archive["__header__"].item()))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "pins":
            print(json.dumps(pins_receipt(), indent=2, sort_keys=True))
            return 0
        if args.command == "transform-check":
            from .transform import SphericalHarmonicTransform

            transform = SphericalHarmonicTransform.create(
                args.truncation,
                backend=args.backend,
                precision=args.precision,
                dealias_factor=args.dealias_factor,
            )
            print(
                json.dumps(
                    transform.transform_check(seed=args.seed),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "run":
            cfg = load_config(args.config)

            def progress(diag):
                print(
                    f"step={diag['step']} time={diag['time_s']:.0f}s "
                    f"max_wind={diag['max_wind_m_s']:.3f}m/s"
                )

            result = run(
                cfg,
                args.outdir,
                restart=args.restart,
                progress=progress,
                overwrite=args.overwrite,
            )
            print(
                json.dumps(
                    {
                        "name": result["name"],
                        "model": result["model"],
                        "status": result["status"],
                        "receipt": result["receipt_path"],
                        "wall_seconds": result["wall_seconds"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0 if result["status"] == "pass" else 1
        if args.command == "benchmark":
            if args.iterations < 1:
                raise ValueError("--iterations must be >= 1")
            if args.warmup < 0:
                raise ValueError("--warmup must be >= 0")
            print(
                json.dumps(
                    _benchmark(args.config, args.iterations, args.warmup),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "inspect":
            metadata, _ = read_checkpoint(args.checkpoint)
            print(json.dumps(metadata, indent=2, sort_keys=True))
            return 0
        if args.command == "check-receipt":
            payload = check_receipt(args.receipt)
            print(
                json.dumps(
                    {
                        "status": payload["status"],
                        "self_sha256": payload["self_sha256"],
                    },
                    indent=2,
                )
            )
            return 0 if payload["status"] == "pass" else 1
        if args.command == "compress-scalar":
            _require_output(args.output, args.overwrite)
            cfg = load_config(args.config)
            transform = build_transform(cfg)
            field = _npz_field(args.input, args.field)
            compressed = compress_scalar(
                transform,
                field,
                space=args.space,
                floor=args.floor,
                preserve_mean=not args.no_preserve_mean,
                keep_degree=args.keep_degree,
                source_identity={"path": str(args.input), "field": args.field},
            )
            write_compressed_scalar(args.output, compressed)
            print(json.dumps(compressed.header, indent=2, sort_keys=True))
            return 0
        if args.command == "decompress-scalar":
            compressed = read_compressed_scalar(args.input)
            cfg = load_config(args.config)
            transform = build_transform(cfg)
            field = decode_scalar(
                transform, compressed, keep_degree=args.keep_degree
            )
            _write_npz(args.output, args.overwrite, **{args.field: field})
            print(json.dumps({"output": str(args.output), "shape": list(field.shape)}, indent=2))
            return 0
        if args.command == "compress-wind":
            _require_output(args.output, args.overwrite)
            cfg = load_config(args.config)
            transform = build_transform(cfg)
            u = _npz_field(args.input, args.u_field)
            v = _npz_field(args.input, args.v_field)
            compressed = compress_wind(
                transform,
                u,
                v,
                keep_degree=args.keep_degree,
                source_identity={
                    "path": str(args.input),
                    "u_field": args.u_field,
                    "v_field": args.v_field,
                },
            )
            write_compressed_wind(args.output, compressed)
            print(json.dumps(compressed.header, indent=2, sort_keys=True))
            return 0
        if args.command == "decompress-wind":
            compressed = read_compressed_wind(args.input)
            cfg = load_config(args.config)
            transform = build_transform(cfg)
            u, v = decode_wind(
                transform, compressed, keep_degree=args.keep_degree
            )
            _write_npz(
                args.output,
                args.overwrite,
                **{args.u_field: u, args.v_field: v},
            )
            print(json.dumps({"output": str(args.output), "shape": list(u.shape)}, indent=2))
            return 0
        if args.command == "inspect-compressed":
            header = _compressed_header(args.input)
            schema = header.get("schema")
            if schema == SCALAR_SCHEMA:
                header = read_compressed_scalar(args.input).header
            elif schema == WIND_SCHEMA:
                header = read_compressed_wind(args.input).header
            else:
                raise ValueError(f"unknown compressed schema {schema!r}")
            print(json.dumps(header, indent=2, sort_keys=True))
            return 0
        if args.command == "export-latlon":
            _require_output(args.output, args.overwrite)
            cfg = load_config(args.config)
            transform = build_transform(cfg)
            path = export_checkpoint_latlon(
                cfg,
                transform,
                args.checkpoint,
                args.output,
                nlat=args.nlat,
                nlon=args.nlon,
            )
            metadata, _ = read_latlon_export(path)
            print(json.dumps(metadata, indent=2, sort_keys=True))
            return 0
        if args.command == "inspect-export":
            metadata, _ = read_latlon_export(args.input)
            print(json.dumps(metadata, indent=2, sort_keys=True))
            return 0
    except (
        ValueError,
        FloatingPointError,
        FileNotFoundError,
        FileExistsError,
        ModuleNotFoundError,
        OSError,
    ) as exc:
        print(f"global-spectral: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
