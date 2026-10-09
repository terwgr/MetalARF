"""``woof global go``: one command from a config to a day of pictures.

The engine's 2.7 line introduced a one-command launch that prepares,
forecasts and renders while naming the stage it is in, with the detail in the
run's log.  This is that shape for the global model, and it is the command a
terminal workspace drives: it writes ``status.json`` on every stage
transition, so a workspace can draw a progress bar without parsing prose.

Three stages, and each one is a command a reader can also run alone.  Nothing
here reimplements them: ``go`` builds the same argument namespaces the
standalone doors take and calls the same handlers, so a flag that works there
works here and there is no second copy of anything to drift.

  statics   build the surface fields for this config's grid, when the config
            asks for a real planet and the cache is not already there.
            Skipped, out loud, for a synthetic planet or a warm cache.
  forecast  integrate the config.
  render    export render-ready tapes and draw them through the Rust
            renderer.  ``--no-render`` stops after the forecast.

WHY A SKIP IS ANNOUNCED.  A stage that quietly does nothing is how a reader
comes to believe statics were built when the cache was stale, or that
pictures were drawn when the renderer was missing.  Every skip prints the
reason and lands in the log.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import types

from .configs_dir import config_argument
from .status import StatusWriter

__all__ = ["add_go_arguments", "go"]

#: The stages, in order, so `status.json` can size its own progress from the
#: first write rather than growing a stage count as it goes.
STAGES = ("statics", "forecast", "render")


def add_go_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config", type=config_argument,
        help="the experiment TOML, or the name of a shipped experiment")
    parser.add_argument(
        "--outdir", type=Path, required=True,
        help="the run directory; checkpoints, receipt, status.json and the log "
             "go here, and pictures under <outdir>/pictures")
    parser.add_argument(
        "--start-date", default=None,
        help="analysis valid time as YYYY-MM-DD_HH:MM:SS for the render stage; "
             "without it the forecast still runs and the render stage is "
             "skipped out loud, because a tape with no valid time is a tape "
             "nobody can place in time")
    parser.add_argument(
        "--products", default=None, metavar="LIST",
        help="comma-separated products for the render stage")
    parser.add_argument(
        "--geog-root", type=Path, default=None,
        help="the WPS_GEOG archive for the statics stage; without it the "
             "config's own [statics] geog_root is used")
    parser.add_argument(
        "--no-statics", action="store_true",
        help="skip the statics stage even for a real planet")
    parser.add_argument(
        "--no-render", action="store_true",
        help="stop after the forecast")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace an existing run directory's artifacts")


def _statics_needed(cfg, status: StatusWriter) -> bool:
    """Whether the statics stage has anything to do, said out loud."""

    if cfg.statics.source != "real":
        status.note("statics skipped: this config runs a synthetic planet")
        return False
    from woof.globe.spectral.grid import GaussianGrid

    from .statics import cache_paths

    grid = GaussianGrid.create(
        cfg.truncation, nlat=cfg.nlat, nlon=cfg.nlon,
        dealias_factor=cfg.dealias_factor)
    cache, _ = cache_paths(cfg.statics, grid)
    if cache.is_file():
        status.note(f"statics skipped: the cache is already built at {cache}")
        return False
    return True


def go(args: argparse.Namespace) -> int:
    from . import cli
    from .config import load_config
    from .render_door import render

    outdir = Path(args.outdir)
    status = StatusWriter(outdir, "go", stages=STAGES)
    if status.log_path is not None:
        print(f"log: {status.log_path}")
    try:
        cfg = load_config(args.config)

        status.stage("statics")
        if args.no_statics:
            status.note("statics skipped: --no-statics")
        elif _statics_needed(cfg, status):
            print("go: building statics")
            leg = types.SimpleNamespace(
                config=args.config, out=None, overwrite=args.overwrite,
                sector_degrees=_default_sector_degrees(),
                geog_root=args.geog_root)
            code = cli._statics(leg)
            if code != 0:
                status.failed("the statics stage did not finish")
                return code

        status.stage("forecast")
        print("go: forecasting")
        leg = _run_namespace(args)
        code = cli._run(leg)
        if code != 0:
            status.failed("the forecast stage did not finish")
            return code

        status.stage("render")
        if args.no_render:
            status.note("render skipped: --no-render")
            status.done("forecast complete, render skipped by request")
            return 0
        if not args.start_date:
            status.note(
                "render skipped: --start-date was not given, and a tape with "
                "no valid time is a tape nobody can place in time")
            print("go: render skipped, --start-date was not given")
            status.done("forecast complete, render skipped for want of a valid time")
            return 0
        print("go: rendering")
        checkpoints = sorted(outdir.glob("arwen_global_step*.npz"))
        if not checkpoints:
            status.note("render skipped: the forecast wrote no checkpoints")
            status.done("forecast complete, nothing to render")
            return 0
        leg = types.SimpleNamespace(
            config=args.config, inputs=checkpoints,
            outdir=outdir / "pictures", start_date=args.start_date,
            products=args.products or _default_products(),
            size="1600x1000", nlat=360, nlon=720, bbox=None,
            tapes_dir=None, keep_tapes=False, overwrite=args.overwrite)
        code = render(leg)
        if code != 0:
            status.failed("the render stage did not finish")
            return code
        status.done("statics, forecast and pictures complete")
        return 0
    except Exception as exc:
        status.failed(f"{type(exc).__name__}: {exc}")
        raise


def _default_products() -> str:
    from .render_door import DEFAULT_PRODUCTS

    return DEFAULT_PRODUCTS


def _default_sector_degrees() -> float:
    from .statics import SECTOR_DEGREES

    return SECTOR_DEGREES


def _run_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """The `run` door's namespace, built from `go`'s smaller surface.

    Every flag `run` declares and `go` does not is filled with `run`'s OWN
    default, read off `run`'s parser rather than restated here: a second copy
    of a default list is how `go` comes to size a card differently from the
    command it claims to be running.
    """

    from . import cli

    parser = argparse.ArgumentParser()
    cli._add_run_arguments(parser)
    defaults = vars(parser.parse_args([str(args.config), "--outdir", str(args.outdir)]))
    defaults["config"] = args.config
    defaults["outdir"] = Path(args.outdir)
    defaults["overwrite"] = args.overwrite
    return argparse.Namespace(**defaults)
