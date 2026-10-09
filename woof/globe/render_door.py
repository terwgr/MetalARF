"""``woof global render``: checkpoints to pictures in one command.

Before this door a picture took two commands and a hop between two
distributions: export render-ready tapes with ``export``, find them, then
drive the engine's ``render``.  ``wrfout_export`` says so in its own
docstring.  A capability that needs a reader to know the intermediate format
is a capability with no door, so this is the door.

THE RENDER PATH IS THE ENGINE'S RUST RENDERER AND NOTHING ELSE.  Every
weather-field picture comes out of ``rw_wrfbatch`` through
``woof.render.render_wrfouts_rust``.  There is no matplotlib fallback here
and there will not be one: the engine's own ``--engine matplotlib`` escape
exists for a different problem and drawing a weather field with it is not
something this package offers.  When the renderer is not staged, this door
refuses and names the binary and the bundle it comes from.

The tapes are an intermediate, so by default they are written under the
output directory and removed when the pictures are drawn.  ``--keep-tapes``
keeps them, and ``--tapes-dir`` puts them somewhere a reader chooses; either
way the export receipt travels with them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .configs_dir import config_argument
from .status import StatusWriter

__all__ = ["add_render_arguments", "render"]

#: The checkpoint files a run directory holds, in time order.
CHECKPOINT_GLOB = "arwen_global_step*.npz"

#: What a bare `--products` means.  `all` renders the renderer's full catalog,
#: which for a global tape is minutes rather than seconds, so the default is
#: four surface fields a reader looks at first.
#:
#: THESE ARE THE RENDERER'S OWN SLUGS, and they were read off its catalog for a
#: global tape rather than invented (`woof render TAPE --list-products`).  The
#: first spelling of this default named `t2,wind10,precip,olr`, none of which is
#: a slug the catalog carries, so a bare `woof global render` drew nothing at
#: all and said so only in the skipped list.  A default that renders nothing is
#: a door that does not exist.  MEASURED 2026-09-07 against a T255 tape: these
#: four report `renderable`, while `simulated_ir_satellite` (the outgoing
#: longwave a fifth entry would want) reports `missing-fields` because the tape
#: does not store it, so it is deliberately not here.
DEFAULT_PRODUCTS = "2m_temperature,mslp_10m_winds,10m_wind_speed_and_direction,total_qpf"


def add_render_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=config_argument, default=None,
        help="the experiment TOML the checkpoints were run under, or the name "
             "of a shipped experiment.  Optional: a run directory written by "
             "`run` or `go` carries a copy of its own config, and that copy is "
             "used when this flag is absent")
    parser.add_argument(
        "inputs", nargs="+", type=Path, metavar="CHECKPOINT_OR_RUNDIR",
        help="checkpoints in time order, run directories (every "
             f"{CHECKPOINT_GLOB} inside, sorted), or wrfout tapes that are "
             "already exported")
    parser.add_argument(
        "--outdir", type=Path, required=True,
        help="where the pictures go, laid out <outdir>/<domain>/<product>/<valid-day>/")
    parser.add_argument(
        "--start-date", required=True,
        help="analysis valid time as YYYY-MM-DD_HH:MM:SS; checkpoint times offset from it")
    parser.add_argument(
        "--products", default=DEFAULT_PRODUCTS, metavar="LIST",
        help=f"comma-separated products, or 'all' (default: {DEFAULT_PRODUCTS})")
    parser.add_argument(
        "--size", default="1600x1000", metavar="WxH",
        help="picture size in pixels (default 1600x1000)")
    parser.add_argument(
        "--nlat", type=int, default=360,
        help="latitude points of the export grid (default 360)")
    parser.add_argument(
        "--nlon", type=int, default=720,
        help="longitude points of the export grid (default 720)")
    parser.add_argument(
        "--bbox", type=float, nargs=4, default=None,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="crop to a lat/lon window (degrees, lon in -180..180)")
    parser.add_argument(
        "--tapes-dir", type=Path, default=None,
        help="where the intermediate tapes are written (default: a directory "
             "under --outdir that is removed when the pictures are drawn)")
    parser.add_argument(
        "--keep-tapes", action="store_true",
        help="keep the exported tapes and their receipt")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace tapes and pictures that already exist")


def _sweep_render_scratch(outdir: Path, status) -> None:
    """Remove the renderer's scratch root when it is empty.

    The engine gives each render its own store under `<outdir>.render-scratch`
    and removes the store when the file's render finishes; on some filesystems
    the empty directory survives, and what a reader then sees beside their
    pictures is a directory named "scratch" that nothing cleaned up.

    Only ever removed when it is COMPLETELY EMPTY -- no entries at all, not
    merely no files.  Two reasons, and the second is measured: a concurrent
    render may own a store in there and taking it would kill that run's
    output; and on Windows a store path inside a deep output directory
    exceeds MAX_PATH, at which point `Path.is_file()` answers False for a
    file that exists.  A sweep that trusted that answer would call a
    populated store empty, so the test is the one question a long path
    cannot corrupt: does the directory have any entry.
    """

    scratch = Path(str(outdir) + ".render-scratch")
    if not scratch.is_dir():
        return
    try:
        entries = any(scratch.iterdir())
    except OSError:
        return
    if entries:
        status.note(
            f"left {scratch}: the renderer's own scratch stores are still "
            "there.  They are the engine's to remove and it did not; nothing "
            "here deletes a directory it did not create the contents of")
        return
    try:
        shutil.rmtree(scratch)
    except OSError:
        pass


def _config_for(args: argparse.Namespace) -> Path:
    """The config to export under: the flag, or the run directory's own copy.

    A reader who has just run a forecast should not have to re-state which
    experiment produced it.  `run` and `go` leave a byte copy of the config in
    the run directory precisely so this question has an answer; when neither
    the flag nor the copy is there, the refusal says which run directory was
    looked in rather than printing "config is required".
    """

    from .cli import read_run_sidecar

    if args.config is not None:
        return Path(args.config)
    looked = []
    for entry in args.inputs:
        path = Path(entry)
        directory = path if path.is_dir() else path.parent
        looked.append(str(directory))
        found = read_run_sidecar(directory)
        if found is not None:
            return found
    raise ValueError(
        "no --config was given and none of the inputs sit in a run directory "
        "that carries its own config copy.\n"
        "Looked in: " + ", ".join(looked) + "\n"
        "Pass --config with the experiment TOML (or a shipped experiment "
        "name), or render a run directory written by `run` or `go`, which "
        "leaves its config beside the checkpoints.")


def _expand(inputs) -> tuple[list[Path], list[Path]]:
    """Split the positional inputs into (checkpoints, already-exported tapes)."""

    checkpoints: list[Path] = []
    tapes: list[Path] = []
    for entry in inputs:
        path = Path(entry)
        if path.is_dir():
            found = sorted(path.glob(CHECKPOINT_GLOB))
            if not found:
                raise FileNotFoundError(
                    f"{path} is a directory with no {CHECKPOINT_GLOB} in it; "
                    "point at a run directory, a checkpoint, or an exported tape")
            checkpoints.extend(found)
        elif path.suffix == ".npz":
            checkpoints.append(path)
        elif path.suffix in (".nc", ""):
            tapes.append(path)
        else:
            raise ValueError(
                f"{path} is neither a checkpoint (.npz), a run directory, nor a "
                "wrfout tape")
    return checkpoints, tapes


def _require_renderer() -> Path:
    """The renderer, or a refusal naming the binary and its bundle."""

    from .doors import find_door

    found = find_door("rw_wrfbatch")
    if found is None:
        raise RuntimeError(
            "rw_wrfbatch is not staged, so there is nothing that can draw a "
            "weather field here.  It ships in the engine's bridge bundle: run "
            "`woof fetch-bridges`.  This door has no matplotlib fallback.")
    return found


def _parse_size(spec: str) -> tuple[int, int]:
    try:
        width, height = spec.lower().split("x", 1)
        return int(width), int(height)
    except ValueError:
        raise ValueError(
            f"--size {spec!r} is not WxH, for example 1600x1000") from None


def render(args: argparse.Namespace) -> int:
    from .config import load_config
    from .wrfout_export import EXPORT_RECEIPT_NAME, export_wrfout

    outdir = Path(args.outdir)
    status = StatusWriter(outdir, "render", stages=("export", "draw"))
    if status.log_path is not None:
        print(f"log: {status.log_path}")
    try:
        _require_renderer()
        checkpoints, tapes = _expand(args.inputs)
        size = _parse_size(args.size)

        if checkpoints:
            tapes_dir = Path(args.tapes_dir) if args.tapes_dir else outdir / "tapes"
            status.stage("export", step_count=len(checkpoints))
            cfg = load_config(_config_for(args))
            written = export_wrfout(
                cfg, checkpoints, tapes_dir,
                nlat=args.nlat, nlon=args.nlon,
                start_date=args.start_date,
                overwrite=args.overwrite,
                bbox=tuple(args.bbox) if args.bbox else None,
            )
            status.note(f"{len(written)} tapes under {tapes_dir}")
            tapes = [*tapes, *written]
        else:
            tapes_dir = None

        if not tapes:
            raise ValueError("nothing to render: no checkpoints and no tapes")

        status.stage("draw", step_count=len(tapes))
        from woof.render import render_wrfouts_rust

        # Every keyword on this call exists on the engine's renderer, and a
        # test holds it there: `series` was passed here from the day the door
        # was written and NO engine has ever declared it, so every draw died
        # at the call with a TypeError after the tapes had been exported
        # (measured 2026-09-07 against 2.7.0 and against the tree the model is
        # developed on; both signatures agree and neither has it).
        pictures, failures, skipped = render_wrfouts_rust(
            tapes, products=args.products, timeidx=None,
            outdir=outdir, size=size)
        status.note(f"{len(pictures)} pictures, {len(failures)} failures")

        if tapes_dir is not None and not args.keep_tapes:
            receipt = tapes_dir / EXPORT_RECEIPT_NAME
            keep = receipt.read_text(encoding="utf-8") if receipt.is_file() else None
            shutil.rmtree(tapes_dir, ignore_errors=True)
            if keep is not None:
                outdir.mkdir(parents=True, exist_ok=True)
                (outdir / EXPORT_RECEIPT_NAME).write_text(keep, encoding="utf-8")
            status.note("intermediate tapes removed; --keep-tapes keeps them")

        _sweep_render_scratch(outdir, status)

        print(json.dumps({
            "pictures": [str(path) for path in pictures],
            "failures": list(failures),
            "skipped": [list(row) for row in skipped],
            "outdir": str(outdir),
        }, indent=2, sort_keys=True))
        if failures:
            status.failed(f"{len(failures)} product(s) did not draw")
            return 1
        status.done(f"{len(pictures)} pictures")
        return 0
    except Exception as exc:
        status.failed(f"{type(exc).__name__}: {exc}")
        raise
