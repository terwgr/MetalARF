"""Command line interface for WOOF global Level 5."""
from __future__ import annotations

import argparse
import dataclasses
import functools
import glob
import json
from pathlib import Path
import sys

import numpy as np

from woof.globe.physics.registry import global_physics_manifest

from ._version import CONSOLE_SCRIPT
from .checkpoint import read_checkpoint
from .config import load_config
from .configs_dir import config_argument, config_root, list_configs
from .export import export_parent, read_parent_export
from .migration import (
    migrate_level4_checkpoint,
    read_migration_receipt,
)
from .native_qualification import (
    qualify_native_adapter,
    read_native_contract_candidate,
    read_native_device_evidence,
)
from .physics.builtin_adapters import ensure_builtin_global_physics_adapters
from .pins import pins_receipt
from .receipt import check_receipt
from .regional.artifact import (
    read_parent_series,
    read_regional_frame,
    read_regional_target,
    write_parent_series,
    write_regional_target,
)
from .regional.translate import translate_parent_to_regional_frame
from .runner import run, vertical_grid_sentence


# `render_door` and `go_door` are reached through these two one-line shims
# rather than imported at module scope, deliberately: both pull in the
# engine's renderer and the static builder, and a `--help` should never pay
# for a stack it is not about to use.
def add_go_arguments(parser):
    from .go_door import add_go_arguments as _add

    return _add(parser)


def add_render_arguments(parser):
    from .render_door import add_render_arguments as _add

    return _add(parser)


#: The three legs this model is reached by from BOTH doors -- `python -m
#: arwen_global` and `woof global` -- declared once each.  Two
#: hand-kept copies of an argument list is how the main door comes to
#: accept a flag the module door dropped, or to spell a default
#: differently; the help text is single-sourced for the same reason.
_RUN_HELP = (
    "run a registered WOOF global TOML experiment; the card is priced "
    "and the band count and host tier chosen before anything is "
    "allocated, so no memory flag has to be set")
_ASSIMILATE_HELP = (
    "assimilate minutes-fresh point observations into a checkpoint "
    "(research-grade v1: surface pressure, temperature, winds, dewpoint)")
_CYCLE_HELP = (
    "forecast and assimilate in one process: integrate to each analysis "
    "instant, analyse the resident state against the observation sources, "
    "write the hour's analysis checkpoint, continue from it, then run the "
    "forecast on to the end")
_MICROWAVE_HELP = (
    "the microwave leg: fetch, decode, thin and score ATMS brightness temperatures "
    "against analysis columns and write the operator entry (the subcommands of "
    "woof global microwave)")
_ABI_SCORE_HELP = (
    "score SimSat ABI brightness temperatures of a checkpoint against the GOES-R "
    "Level 1b radiances of one scan, on the ABI fixed grid by lattice index")
_ABI_REFERENCE_HELP = (
    "CRTM as the reference beside the ABI operator: write the analysed columns under the "
    "both-clear blocks for tools/crtm_reference, or score the reference's brightness "
    "temperatures against the observation and SimSat by term")
_ABI_FAST_MODEL_HELP = (
    "train the ABI clear-sky fast forward model's coefficient table on the reference's layer optical "
    "depths, or run the Rust operator (rw_goes forward) on a columns stream")
_EXPORT_WRFOUT_HELP = (
    "write render-ready global lat-lon wrfout tapes from checkpoints")
_STATICS_HELP = (
    "build the static surface fields (land use, soil, vegetation, LAI, "
    "albedo, snow albedo, deep soil temperature) for a config's Gaussian "
    "grid from the WPS_GEOG archive and cache them once per truncation")
_DA_HELP = (
    "the data-assimilation door: init an ensemble from an analysis, cycle "
    "hourly through the observation streams, analyze one instant, fresh "
    "(fetch the newest analysis and every stream, init, cycle to the newest "
    "observation hour, hand back the analysis checkpoint), forecast from it")
_DA_INIT_HELP = (
    "build the ensemble from an analysis (a checkpoint, or the config's cold "
    "start written as step 0) and write its manifest (da-ensemble.json)")
_DA_CYCLE_HELP = (
    "the hourly cycle: fetch each --stream for the window, decode, analyse "
    "through the filter, write the analysis checkpoint and the ensemble "
    "manifest; every report carries the DA scorecard (O-B and O-A per "
    "stream, variable and region) and the receipt the wall budget per cycle")
_DA_ANALYZE_HELP = (
    "one analysis at a stated instant from stated observation tables into a "
    "checkpoint, its scorecard printed")
_DA_FRESH_HELP = (
    "a fresh global analysis now: fetch the newest GDAS analysis (or read "
    "--analysis-grib), derive the run config from the base TOML, init if no "
    "ensemble exists in --outdir, cycle hourly through every --stream and "
    "--obs to the newest observation hour, hand back the analysis checkpoint "
    "and the forecast command")
_GO_HELP = (
    "one command: build the statics this config needs, integrate it, and "
    "draw the pictures, naming the stage it is in while the detail goes to "
    "the run's log")
_RENDER_HELP = (
    "checkpoints or a run directory to product pictures in one command, "
    "through the Rust renderer; the render-ready tapes are an intermediate "
    "and are removed unless --keep-tapes asks for them")
_CONFIGS_HELP = "list the experiment TOMLs that ship inside this package"
_DA_FORECAST_HELP = (
    "the forecast from an analysis checkpoint (the runner's restart), the "
    "receipt naming the analysis it started from")
_DA_LOCALISATION_HELP = (
    "derive the per-class vertical localisation cutoffs from an ensemble "
    "store: the members' vertical correlations per report class and region "
    "and the Gaspari-Cohn support fitted to each, on the host, written as a "
    "receipt beside the values FilterOptions carries")


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config", type=config_argument,
        help="WOOF global experiment TOML (truncation, vertical coordinate, "
             "physics suite, gates, and [time] integrator: 'sl_si', the "
             "two-time-level semi-Lagrangian core, is the default at every "
             "truncation and steps 300 s when dt_s is omitted, with its own "
             "drain and gather (order 16 at 720 s, the six-point gather, "
             "off-centring 0.55) when those are omitted (at equal cost it "
             "grades five scorecard rows better, four worse and nine level "
             "against the Eulerian core, better at the surface on "
             "temperature and wind and worse on dewpoint and aloft, the rows "
             "named on the door page, at about a quarter of the wall per "
             "forecast day); 'imex_ssp3', the Eulerian IMEX pair, is "
             "selectable by name, its step bounded by the wind and set by a "
             "rule when dt_s is omitted (the largest whole step keeping the "
             "strongest analysis day on disk under 0.70 of the CFL gate: "
             "90 s at T255, 60 s at T383, 40 s at T533), its ten-step "
             "identity pinned)",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("out/arwen-global"),
        help="where the checkpoints and the self-hashed receipt are written "
             "(default out/arwen-global)",
    )
    parser.add_argument(
        "--restart", type=Path, default=None,
        help="continue from this checkpoint instead of the cold state; the "
             "config hash it carries has to be this config's",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace this run's own files in --outdir; without it an "
             "existing output is refused rather than half-rewritten",
    )
    parser.add_argument(
        "--until-s", type=float, default=None,
        help="stop the integration at this model time in seconds from the "
             "run's start instead of the config's duration_s (a cycling "
             "segment: the next segment restarts from the checkpoint this "
             "one ends on); the config identity is unchanged and the "
             "receipt records the segment end",
    )
    parser.add_argument(
        "--profile-steps", type=int, default=0,
        help="time every operator of this many steps (after --profile-warmup "
             "unprofiled ones) on the host and on the device stream, write "
             "profile.json beside the receipt and print the table; the run "
             "itself is unchanged (default 0, off)",
    )
    parser.add_argument(
        "--profile-warmup", type=int, default=2,
        help="steps left unprofiled before the profiled window (default 2: "
             "table builds, kernel compiles and the allocator's first "
             "shapes are not a step's steady cost)",
    )
    parser.add_argument(
        "--metal", action="store_true",
        help="enable the optional Apple Metal execution path",
    )
    _add_memory_lever_arguments(parser)


def _add_memory_lever_arguments(parser: argparse.ArgumentParser) -> None:
    """The engine memory levers, overriding the config's ``[memory]``.

    NOTHING HERE HAS TO BE SET.  With no flag and no ``[memory]`` table
    the sizer reads the card before a byte is allocated and chooses the
    band count and the pinned host tier together, and the receipt's
    ``sizer`` block says what it chose and why.  MEASURED 2026-09-06 at
    the ten-step probe of record: a bare T383 L40 run (34.7 km) fits a 16
    GB RTX 5070 Ti, which it does not do resident at any radiation chunk,
    and a bare T533 L40 run (25.0 km) fits a 32 GB RTX 5090, which OOM'd
    at step 2 before this work.  T533 on a 16 GB card, and T799 anywhere,
    are refused by name rather than started.
    """
    parser.add_argument(
        "--latitude-bands", type=int, default=None, metavar="N",
        help="how many latitude bands grid space is streamed through "
             "(config [memory].latitude_bands, default 0 = the sizer "
             "chooses): 1 is the resident run, and above one every "
             "grid-space operator of the step runs a band at a time while "
             "spectral space stays whole.  The Legendre contraction keeps "
             "K = N = nlat at every band count, so the band count changes "
             "no arithmetic and enters no identity: a banded run shares a "
             "config hash, a checkpoint lineage and a receipt with the "
             "resident run and reproduces its checkpoints byte for byte.  "
             "It buys capacity, not speed: a truncation the card cannot "
             "hold resident runs, and one that fits gains nothing.  "
             "NOTHING HAS TO BE SET: with no memory flag and no [memory] "
             "table the sizer reads the card first and chooses this and "
             "the host tier together, and the receipt's 'sizer' block "
             "says what it chose and why",
    )
    parser.add_argument(
        "--host-spill", choices=("auto", "on", "off"), default=None,
        help="whether the persistent grid state -- the ten grid tracers, "
             "the surface reservoirs and the native physics namespace -- "
             "lives in pinned host memory instead of on the card (config "
             "[memory].host_spill, default 'auto').  It reaches the card "
             "only as the copy its consumer builds anyway, so what stops "
             "existing is the original standing beside that copy for the "
             "life of the run.  'auto' parks the minimum the predicted "
             "peak needs, coldest slice first; 'on' parks all three; "
             "'off' keeps every slice on the card.  Memory only, same "
             "bits: a spilled run reproduces the resident run's "
             "checkpoints byte for byte and shares its config hash.  "
             "WHAT IT BUYS, MEASURED 2026-09-06 at the ten-step probe of "
             "record with nothing set: T383 L40 (34.7 km) starts, fits "
             "and reports on a 16 GB RTX 5070 Ti, which it does not do "
             "resident at any radiation chunk, and T533 L40 (25.0 km) on "
             "a 32 GB RTX 5090, which ran out of memory at step 1 before "
             "this work.  T533 on a 16 GB card, and T799 on either, are "
             "refused at the door by name",
    )
    parser.add_argument(
        "--cards", type=int, default=None, metavar="P",
        help="how many GPUs share the band schedule (config "
             "[memory].cards, default 1).  Grid space is partitioned by "
             "latitude band and spectral space is replicated; in the "
             "default 'gather' exchange the waist rows cross the wire and "
             "the Legendre contraction still runs at K = N = nlat on every "
             "card, so two cards return the single-card answer bit for "
             "bit.  Every rank needs --card-rank and the same "
             "--card-addresses list",
    )
    parser.add_argument(
        "--card-exchange", choices=("gather", "partial"), default=None,
        help="'gather' (default) ships the Fourier waist's latitude rows "
             "and keeps the contraction whole, which is bit-identical to "
             "one card and enters no identity; 'partial' ships partial "
             "Legendre sums and adds them in rank order, which is a third "
             "of the bytes and a CHANGE OF ARITHMETIC that carries its own "
             "pin and joins the config hash",
    )
    parser.add_argument(
        "--card-axis", choices=("band", "order"), default=None,
        help="which decomposition a run above one card uses (config "
             "[memory].card_axis, default 'band').  'band' partitions grid "
             "space by latitude and gathers waist rows -- the SPEED axis, "
             "and the one a model run uses.  'order' partitions the Legendre "
             "orders and gathers coefficient columns, holding one card's "
             "fraction of the table -- the CAPACITY axis past the "
             "whole-table wall, proven bit-identical at the transform and "
             "refused for a model run until a truncation that needs it has a "
             "card that fits it",
    )
    parser.add_argument(
        "--card-agreement", choices=("refuse", "record"), default=None,
        help="what a gather run above one card does when its cards return "
             "different bits for a contraction the step presents (config "
             "[memory].card_agreement, default 'refuse').  'refuse' stops the "
             "run by name at that contraction, before the waist it feeds is "
             "assembled from both cards' rows.  'record' carries on, lists "
             "every disagreeing shape in the receipt and FAILS the run's "
             "two_card_contractions_agree gate row: a timing device for a pair "
             "of unlike cards whose bits do not agree, and never a run of "
             "record",
    )
    parser.add_argument(
        "--card-rank", type=int, default=None, metavar="R",
        help="this process's rank, 0-based, inside --cards",
    )
    parser.add_argument(
        "--card-addresses", default=None, metavar="H:P,H:P",
        help="the rendezvous host:port of every rank IN RANK ORDER, comma "
             "separated.  Use the address of the fast interconnect between "
             "the cards, not the management network",
    )
    parser.add_argument(
        "--card-transport", choices=("auto", "tcp", "nccl"), default=None,
        help="'auto' (default) takes NCCL where it imports and TCP "
             "otherwise; the two measured within 1 percent of each other on "
             "this link",
    )
    parser.add_argument(
        "--card-weights", default=None, metavar="W,W",
        help="one relative card speed per rank, in rank order, overriding "
             "the per-band cost profile measured at run start.  The "
             "assignment is outside the arithmetic (gate BIT-6), so this "
             "changes who computes a row and not what the row is",
    )
    parser.add_argument(
        "--card-halo-rows", type=int, default=None, metavar="N",
        help="latitude rows exchanged across a card boundary for the "
             "meridional sweep's deep halo (default 16).  It must cover 2n "
             "for the step's sub-step count n; a step that needs more is "
             "refused by name rather than swept from stale neighbour rows",
    )
    parser.add_argument(
        "--spectral-chunk", type=int, default=None,
        help="widest field stack one transform call carries (config "
             "[memory].spectral_chunk, default 6): a smaller chunk bounds "
             "the complex Fourier temporaries a wide stack materializes "
             "and costs one more pass of the per-order loop; it is the "
             "Legendre GEMM's M dimension and moves bits at a narrow "
             "vertical ladder, so it carries its own config identity",
    )
    parser.add_argument(
        "--synthesis-memo", choices=("on", "off"), default=None,
        help="serve repeated syntheses of one state within a step from a "
             "memo (default on); off recomputes every synthesis, which is "
             "the memory-tightest form and the same bits: a ten-step T255 "
             "native A/B is byte-identical across all 126 checkpoint "
             "arrays, so the two share one config hash and one lineage",
    )
    parser.add_argument(
        "--legendre-band", type=int, default=None,
        help="orders per packed Legendre band (config [memory]."
             "legendre_band, default 32): the strided-batched GEMM's batch "
             "count on cupy, and ARITHMETIC there - a band other than 32 "
             "changes the analysis of a single plane by about 4e-06 at "
             "T255 float32 and takes a ten-step run to a different state, "
             "so it carries its own config and transform identity and a "
             "checkpoint written under it will not restart under another",
    )
    parser.add_argument(
        "--streaming", choices=("on", "off"), default=None,
        help="hold no Legendre table and regenerate the basis one band of "
             "orders at a time (default off): the entry point for a "
             "truncation whose tables do not fit, at 37.7x a resident "
             "synthesis and 15,144x a resident analysis (MEASURED T383)",
    )
    parser.add_argument(
        "--device-allocator", choices=("default", "slab", "async"), default=None,
        help="which allocator the run spends its device bytes through "
             "(config [memory].device_allocator, default 'default'): "
             "'default' is the CuPy pool the process already carries, which "
             "held the least over what the run had live and cost the least "
             "wall on both shapes measured on an RTX 5070 Ti 2026-09-06; "
             "'slab' takes one contiguous arena before the first model byte "
             "and extends it, cutting it with an exact-fit coalescing free "
             "list, so what the card holds is the arena and no pool's binned "
             "free blocks; 'async' installs the driver's cudaMallocAsync "
             "pool.  Memory only, same bits: a ten-step checkpoint gate is "
             "byte-identical under each (T85, 3 checkpoints, 45 arrays)",
    )


def _memory_lever_overrides(args: argparse.Namespace) -> dict[str, object]:
    """The [memory] overrides this invocation names, refused rather than
    dropped when the value cannot be run."""
    overrides: dict[str, object] = {}
    bands = getattr(args, "latitude_bands", None)
    if bands is not None:
        if int(bands) < 0:
            raise ValueError(
                "--latitude-bands must be >= 0: zero asks the sizer to "
                "choose the count, one is the resident run, and a negative "
                "count is not a streaming granularity"
            )
        overrides["latitude_bands"] = int(bands)
    spill = getattr(args, "host_spill", None)
    if spill is not None:
        from .config import HOST_SPILL_MODES

        if str(spill) not in HOST_SPILL_MODES:
            raise ValueError(
                f"--host-spill must be one of {list(HOST_SPILL_MODES)}: an "
                "unknown name would leave the run guessing whether its "
                "persistent grid state is on the card"
            )
        overrides["host_spill"] = str(spill)
    cards = getattr(args, "cards", None)
    if cards is not None:
        if int(cards) < 1:
            raise ValueError(
                "--cards must be >= 1: a run with no card has nothing to run "
                "on, and one card is the shipped default"
            )
        overrides["cards"] = int(cards)
    exchange = getattr(args, "card_exchange", None)
    if exchange is not None:
        overrides["card_exchange"] = str(exchange)
    axis = getattr(args, "card_axis", None)
    if axis is not None:
        overrides["card_axis"] = str(axis)
    agreement = getattr(args, "card_agreement", None)
    if agreement is not None:
        overrides["card_agreement"] = str(agreement)
    rank = getattr(args, "card_rank", None)
    if rank is not None:
        if int(rank) < 0:
            raise ValueError("--card-rank must be >= 0")
        overrides["card_rank"] = int(rank)
    addresses = getattr(args, "card_addresses", None)
    if addresses is not None:
        parsed = tuple(a.strip() for a in str(addresses).split(",") if a.strip())
        if not parsed:
            raise ValueError(
                "--card-addresses needs one host:port per rank, in rank order"
            )
        overrides["card_addresses"] = parsed
    transport = getattr(args, "card_transport", None)
    if transport is not None:
        overrides["card_transport"] = str(transport)
    declared = getattr(args, "card_weights", None)
    if declared is not None:
        values = tuple(
            float(v) for v in str(declared).split(",") if v.strip())
        if not values or any(v <= 0.0 for v in values):
            raise ValueError(
                "--card-weights takes one positive number per rank, in rank "
                "order; a card with a weight of zero would be given no band "
                "and would pay every exchange for nothing"
            )
        overrides["card_weights"] = values
    halo = getattr(args, "card_halo_rows", None)
    if halo is not None:
        if int(halo) < 1:
            raise ValueError(
                "--card-halo-rows must be >= 1: the meridional sweep reads at "
                "least two rows past its block, so a card boundary with no "
                "halo would sweep from stale neighbour values"
            )
        overrides["card_halo_rows"] = int(halo)
    if overrides.get("cards", 1) > 1:
        if "card_addresses" not in overrides:
            raise ValueError(
                "--cards greater than one needs --card-addresses: every rank "
                "must be told where every rank listens, in rank order, and "
                "there is no discovery service to guess it from"
            )
        if len(overrides["card_addresses"]) != int(overrides["cards"]):
            raise ValueError(
                f"--cards {overrides['cards']} with "
                f"{len(overrides['card_addresses'])} addresses: one host:port "
                "per rank, in rank order"
            )
        if int(overrides.get("card_rank", 0)) >= int(overrides["cards"]):
            raise ValueError(
                f"--card-rank {overrides.get('card_rank', 0)} is outside the "
                f"{overrides['cards']} ranks this run has"
            )
    chunk = getattr(args, "spectral_chunk", None)
    if chunk is not None:
        if int(chunk) < 1:
            raise ValueError(
                "--spectral-chunk must be >= 1: a chunk of zero or less "
                "presents the transform an empty field stack and the chunk "
                "loop returns nothing rather than the state"
            )
        overrides["spectral_chunk"] = int(chunk)
    band = getattr(args, "legendre_band", None)
    if band is not None:
        if int(band) < 1:
            raise ValueError("--legendre-band must be >= 1")
        overrides["legendre_band"] = int(band)
    memo = getattr(args, "synthesis_memo", None)
    if memo is not None:
        overrides["synthesis_memo"] = memo == "on"
    streaming = getattr(args, "streaming", None)
    if streaming is not None:
        overrides["streaming"] = streaming == "on"
    allocator = getattr(args, "device_allocator", None)
    if allocator is not None:
        from .device_memory import DEVICE_ALLOCATORS

        if str(allocator) not in DEVICE_ALLOCATORS:
            raise ValueError(
                f"--device-allocator must be one of {DEVICE_ALLOCATORS}: an "
                "unknown name would leave the run on whatever allocator the "
                "process carried while the receipt named another"
            )
        overrides["device_allocator"] = str(allocator)
    return overrides


def _add_assimilation_option_arguments(
    parser: argparse.ArgumentParser, *, with_obs: bool = True,
) -> None:
    """The observation sources and the analysis options, shared by the
    `assimilate`, `cycle` and `da` legs so the doors cannot drift apart.
    ``with_obs=False`` leaves the sources to :func:`_add_stream_arguments`."""
    if with_obs:
        parser.add_argument(
            "--obs", action="append", required=True, metavar="PATH_OR_URL",
            help="observation CSV stream (URL or local path, gzip ok); repeatable",
        )
    parser.add_argument(
        "--length-scale-km", type=float, default=None,
        help="horizontal influence radius of one report (default 300)",
    )
    parser.add_argument(
        "--max-age-minutes", type=float, default=None,
        help="discard reports older than this before the analysis time "
             "(default 90)",
    )
    parser.add_argument(
        "--elevation-limit-m", type=float, default=None,
        help="discard a surface report whose station elevation differs from "
             "the model's by more than this (default 500)",
    )
    parser.add_argument(
        "--wind-balance", choices=("rotational", "unconstrained"), default=None,
        help="how the wind increment enters the state: rotational (default) "
             "applies its streamfunction part only; unconstrained also "
             "applies the divergence the scalar spreading produced, which "
             "is gravity-wave energy, kept for measuring against the default",
    )
    parser.add_argument(
        "--moisture-update", choices=("on", "off"), default=None,
        help="analyse dewpoint reports into specific humidity (default off): "
             "the dewpoint innovation spread with its own vertical "
             "localization, the vapor re-derived at every level, capped at "
             "saturation and floored at zero; selectable because on the "
             "graded 24 h cycle it won the 2 m dewpoint (18 h bias -2.76 to "
             "-0.97 K) and lost sea-level pressure (2.70 to 4.19 hPa rmse) "
             "beyond the admission rule; off leaves water untouched as the "
             "v1 door did",
    )
    parser.add_argument(
        "--humidity-decay-height-m", type=float, default=None,
        help="e-folding height above the surface of a surface dewpoint "
             "report's moisture increment (default 1500)",
    )


def _add_assimilate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config", type=config_argument,
        help="the experiment TOML the checkpoint was run under, or a shipped "
             "experiment's bare name")
    _add_memory_lever_arguments(parser)
    parser.add_argument(
        "checkpoint", type=Path, help="background state to analyse")
    _add_assimilation_option_arguments(parser)
    parser.add_argument(
        "--out", type=Path, required=True,
        help="checkpoint to write the analysis to; the o-minus-b/o-minus-a "
             "report is written beside it",
    )
    parser.add_argument(
        "--analysis-time", default=None,
        help="ISO-8601 analysis instant; default is the newest decoded report",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace an existing analysis checkpoint and report",
    )


def _add_cycle_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config", type=config_argument,
        help="WOOF global experiment TOML; its duration_s is the end of the "
             "whole run, cycle and forecast",
    )
    _add_memory_lever_arguments(parser)
    _add_assimilation_option_arguments(parser)
    parser.add_argument(
        "--outdir", type=Path, required=True,
        help="where the hourly checkpoints, each cycle's analysis checkpoint "
             "(arwen_global_analysis_step*.npz) and report "
             "(assimilation-report-step*.json), and the receipt are written",
    )
    parser.add_argument(
        "--cycles", type=int, required=True,
        help="how many analyses to form, one every --interval-s of model "
             "time from the run's start; refused when they do not fit "
             "before the end of the run",
    )
    parser.add_argument(
        "--interval-s", type=float, default=3600.0,
        help="model time between analyses in seconds (default 3600); must be "
             "a whole number of steps",
    )
    parser.add_argument(
        "--start-utc", default=None,
        help="ISO-8601 instant model time zero stands for, so each analysis "
             "time is this plus its model time; default is the config's "
             "physics start_time_utc, refused when neither exists",
    )
    parser.add_argument(
        "--until-s", type=float, default=None,
        help="stop the integration at this model time in seconds from the "
             "run's start instead of the config's duration_s (equal to the "
             "last analysis time for a cycle without a forecast leg)",
    )
    parser.add_argument(
        "--restart", type=Path, default=None,
        help="continue from this checkpoint instead of the cold state (an "
             "interrupted cycle resumes from its last analysis or hourly "
             "checkpoint); the config hash it carries has to be this config's",
    )
    parser.add_argument(
        "--keep-backgrounds", action="store_true",
        help="also write each analysis hour's background checkpoint; by "
             "default only the analysis is written there, the background's "
             "identity riding in the report and the chain",
    )
    parser.add_argument(
        "--partial-analyses", choices=("on", "off"), default="on",
        help="when some variables fail the gate of record (default on): "
             "withdraw their reports and analyse the hour again with the "
             "rest, which have to pass on their own, carrying only the "
             "failed variables as the background; off carries the whole "
             "background whenever any variable fails, as the per-segment "
             "chain did",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace this cycle's own files in --outdir; without it an "
             "existing output is refused rather than half-rewritten",
    )


def _add_filter_argument(parser: argparse.ArgumentParser, *, with_ensemble: bool = False) -> None:
    parser.add_argument(
        "--filter", choices=("successive-correction", "letkf"), default=None,
        help="the analysis filter (default letkf on fresh since 2026-09-06, "
             "successive-correction on init and cycle: the deterministic v1 "
             "door); letkf is the dual-resolution ensemble "
             "filter (N members at their own truncation resident in one "
             "process; the control analysed from its own innovations through "
             "the ensemble covariance, the members recentred on it)",
    )
    if with_ensemble:
        parser.add_argument(
            "--ensemble-truncation", type=int, default=None,
            help="the ensemble members' spectral truncation for --filter letkf "
                 "(default 127; at or below the deterministic truncation)",
        )
    parser.add_argument(
        "--control-increment", choices=("control", "ensemble-mean"), default="control",
        help="letkf: how the control is updated (default control: its own "
             "innovations through the ensemble covariance; ensemble-mean is the "
             "comparison experiment, the ensemble-mean increment embedded)",
    )
    parser.add_argument(
        "--recentre-fraction", type=float, default=1.0,
        help="letkf: how far the ensemble mean moves to the control analysis "
             "restricted to the ensemble truncation (default 1.0, full; a "
             "fraction is partial recentring)",
    )
    parser.add_argument(
        "--recentre-mode", choices=("increment", "state"), default="increment",
        help="letkf: how the members follow the control analysis (default "
             "increment: the ensemble-mean increment is replaced by the "
             "control's, anchor included, at the ensemble truncation, so the "
             "members keep their own terrain-consistent background; state: the "
             "control analysis truncated to the ensemble truncation replaces "
             "the ensemble mean, which carries a finer orography's surface "
             "pressure onto the coarser grid)",
    )
    parser.add_argument(
        "--taper-full-degree", type=int, default=None,
        help="letkf: the total degree up to which the control increment keeps "
             "full weight (default 0.6 of the ensemble truncation)",
    )
    parser.add_argument(
        "--taper-zero-degree", type=int, default=None,
        help="letkf: the total degree at and above which the control increment "
             "is zero (default the ensemble truncation)",
    )
    parser.add_argument(
        "--additive-inflation", type=float, default=None, metavar="FRACTION",
        help="letkf: additive inflation as a fraction of the initial "
             "perturbation amplitude re-drawn after every analysis (default "
             "off: RTPS is the one inflation mechanism)",
    )
    from .da.options import DEFAULT_HYBRID_BETA
    from .da_door import DEFAULT_FRESH_HYBRID_BETA

    parser.add_argument(
        "--filter-option", action="append", default=None, metavar="NAME=VALUE",
        help="letkf: one FilterOptions field set by name (repeatable), for "
             "example amv_height_assignment_sigma_pa=10000 or "
             "refractivity_vertical_cutoff_lnp=1.5; an unknown name is refused "
             "with the field list; the effective options ride in every "
             "analysis report and the ensemble manifest",
    )
    parser.add_argument(
        "--hybrid-beta", type=float, default=None, metavar="BETA",
        help="letkf: the ensemble weight of the hybrid covariance beta B_ens + "
             "(1 - beta) B_static the control's gain is built from, in (0, 1] "
             f"(default {DEFAULT_FRESH_HYBRID_BETA:g} on fresh, the completed "
             f"system's, and {DEFAULT_HYBRID_BETA:g} on init and cycle; 1 is the "
             "ensemble alone); below one the static covariance table is "
             "sampled into the localised solve",
    )
    parser.add_argument(
        "--static-covariance", default="packaged", metavar="TABLE",
        help="letkf: the static covariance table for --hybrid-beta below one: "
             "'packaged' (default, the lagged-forecast estimate shipped with "
             "the package), a path written by `woof global da "
             "static-covariance`, or 'none' (beta 1 only)",
    )
    parser.add_argument(
        "--static-samples", type=int, default=None, metavar="K",
        help="letkf: draws from the static covariance per analysis for the "
             "augmented solve (default the package's 64)",
    )
    parser.add_argument(
        "--letkf-solve-path", choices=("auto", "device", "host"), default=None,
        help="letkf: where the localised solve runs (default auto: the "
             "members' own namespace, the card when the model is on one; host "
             "is the numpy reference the device path is compared against, the "
             "same code in the other array module); the receipt names the path "
             "taken and its wall",
    )
    parser.add_argument(
        "--operator-precision", choices=("state", "float64"), default=None,
        help="letkf: how the point operators contract a device-resident state "
             "(default state: a float32 state's coefficients through float32 "
             "GEMMs over 256-term blocks summed in float64, the state's own "
             "precision; float64 forces the float64 contraction, the host "
             "path's arithmetic to rounding)",
    )


def _filter_settings_from_args(args):
    return {
        "solve_path": getattr(args, "letkf_solve_path", None),
        "operator_precision": getattr(args, "operator_precision", None),
    }


def _add_cycle_setting_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--observation-bin-s", type=float, default=None,
        help="compare every report with the state at the bin instant nearest "
             "its valid time, bins this many seconds wide (a whole multiple of "
             "the model step, dividing the interval); default: at the analysis "
             "instant on cycle, and on fresh under letkf 600 s or the smallest "
             "multiple of the step above it that divides the interval",
    )
    parser.add_argument(
        "--increment-application", choices=("direct", "iau"), default="iau",
        help="iau (default): the window re-integrated from its start with the "
             "increment added in equal parts at every step (the members take "
             "theirs over the next window); direct: the increment inserted at "
             "the analysis instant",
    )
    parser.add_argument(
        "--anchor", default=None, metavar="PATH[:key=value;...]",
        help="the external analysis as a weak low-pass constraint on the "
             "control: a checkpoint of this config or a GRIB analysis, with "
             "valid_utc=..., weight=0.1, full_degree=30, zero_degree=40, "
             "max_age_s=10800, fields=theta,log_surface_pressure,...",
    )


def _filter_overrides_from_args(args) -> dict:
    from .da.options import parse_filter_overrides

    return parse_filter_overrides(getattr(args, "filter_option", None))


def _control_options_from_args(args):
    from .da_control import ControlOptions

    from .da.options import DEFAULT_HYBRID_BETA
    from .da_door import DEFAULT_FRESH_HYBRID_BETA

    table = getattr(args, "static_covariance", "packaged")
    if table is not None and str(table).strip().lower() == "none":
        table = None
    beta = getattr(args, "hybrid_beta", None)
    if beta is None:
        beta = DEFAULT_FRESH_HYBRID_BETA if getattr(args, "da_command", None) == "fresh" else DEFAULT_HYBRID_BETA
    return ControlOptions(
        increment_source=getattr(args, "control_increment", "control"),
        recentre_fraction=float(getattr(args, "recentre_fraction", 1.0)),
        recentre_mode=getattr(args, "recentre_mode", "increment"),
        taper_full_degree=getattr(args, "taper_full_degree", None),
        taper_zero_degree=getattr(args, "taper_zero_degree", None),
        increment_application=getattr(args, "increment_application", "iau"),
        hybrid_beta=float(beta),
        static_covariance=table,
        static_samples=getattr(args, "static_samples", None),
    )


def _add_stream_arguments(parser: argparse.ArgumentParser, *, obs_required: bool) -> None:
    parser.add_argument(
        "--obs", action="append", required=obs_required, metavar="PATH_OR_URL",
        default=None,
        help="observation table (URL or local path, gzip ok) in the obs-table "
             "vocabulary, decoded once for every cycle; repeatable",
    )
    parser.add_argument(
        "--stream", action="append", default=None, metavar="NAME[:key=value;...]",
        help="observation stream fetched for every analysis window and "
             "recorded (URL or path, bytes, SHA-256, latency behind real "
             "time): iem-asos[:networks=IA_ASOS,IL_ASOS;bbox=W,S,E,N], "
             "local-tables:paths=A.csv,B.csv, atms[:satellites=noaa-20,noaa-21;"
             "cache=DIR] and goes-abi[:satellites=G19;bands=13,8;cache=DIR] "
             "(the radiance streams, letkf only); repeatable; the table is "
             "woof.globe.da_streams.STREAM_TABLE; fresh with no "
             "--stream and no --obs runs the shipped roster "
             "(da_door.DEFAULT_FRESH_STREAMS), and naming any stream replaces it",
    )


def _add_da_init_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config", type=config_argument,
        help="WOOF global experiment TOML, or a shipped experiment's bare name")
    parser.add_argument(
        "--outdir", type=Path, required=True,
        help="where the deterministic checkpoint, the ensemble manifest and "
             "the DA receipt are written",
    )
    parser.add_argument(
        "--from-checkpoint", type=Path, default=None,
        help="the analysis checkpoint the ensemble is built from; default is "
             "the config's cold start written as step 0 into --outdir",
    )
    parser.add_argument(
        "--members", type=int, default=1,
        help="ensemble members (default 1: the deterministic filter carries "
             "one; more need --filter letkf)",
    )
    parser.add_argument(
        "--analysis-time", default=None,
        help="ISO-8601 instant the analysis stands for, recorded in the manifest",
    )
    _add_filter_argument(parser, with_ensemble=True)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace an existing manifest and cold-start checkpoint in --outdir",
    )


def _add_da_cycle_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config", type=config_argument,
        help="WOOF global experiment TOML; its duration_s is the end of the "
             "whole run, cycle and forecast",
    )
    _add_stream_arguments(parser, obs_required=False)
    parser.add_argument(
        "--metal", action="store_true",
        help="run the cycle forecast legs and model-coupled analysis operators through the Global Metal runner",
    )
    _add_assimilation_option_arguments(parser, with_obs=False)
    parser.add_argument(
        "--outdir", type=Path, required=True,
        help="where the hourly checkpoints, each cycle's analysis checkpoint "
             "and report, the fetch manifests (fetch/), the ensemble manifest, "
             "the run receipt and the DA receipt are written",
    )
    parser.add_argument(
        "--cycles", type=int, required=True,
        help="how many analyses to form, one every --interval-s of model "
             "time from the run's start",
    )
    parser.add_argument(
        "--interval-s", type=float, default=3600.0,
        help="model time between analyses in seconds (default 3600)",
    )
    parser.add_argument(
        "--start-utc", default=None,
        help="ISO-8601 instant model time zero stands for; default the "
             "config's physics start_time_utc, refused when neither exists",
    )
    parser.add_argument(
        "--until-s", type=float, default=None,
        help="stop the integration at this model time (default the config's "
             "duration_s; equal to the last analysis time for a cycle without "
             "a forecast leg)",
    )
    parser.add_argument(
        "--restart", type=Path, default=None,
        help="continue from this checkpoint instead of the cold state",
    )
    parser.add_argument(
        "--ensemble", type=Path, default=None,
        help="the ensemble manifest (da-ensemble.json) to cycle from; its "
             "deterministic checkpoint is the restart unless --restart names one",
    )
    _add_filter_argument(parser, with_ensemble=True)
    _add_cycle_setting_arguments(parser)
    parser.add_argument(
        "--keep-backgrounds", action="store_true",
        help="also write each analysis hour's background checkpoint",
    )
    parser.add_argument(
        "--partial-analyses", choices=("on", "off"), default="on",
        help="when some variables fail the gate of record (default on): "
             "withdraw them and analyse the hour again with the rest",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace this cycle's own files in --outdir",
    )


def _add_da_analyze_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config", type=config_argument,
        help="the experiment TOML the checkpoint was run under, or a shipped "
             "experiment's bare name")
    parser.add_argument(
        "checkpoint", type=Path, help="background state to analyse")
    _add_assimilation_option_arguments(parser)
    parser.add_argument(
        "--out", type=Path, required=True,
        help="directory for the analysis checkpoint, the report (with its "
             "scorecard) and the DA receipt",
    )
    parser.add_argument(
        "--analysis-time", default=None,
        help="ISO-8601 analysis instant; default is the newest decoded report",
    )
    _add_filter_argument(parser)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace an existing analysis checkpoint and report",
    )


def _add_da_fresh_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config", type=config_argument,
        help="the BASE experiment TOML; fresh derives the run config from it "
             "(fresh-config.toml in --outdir: the fetched analysis as the "
             "initial state, duration = cycle span + --forecast-hours)",
    )
    _add_stream_arguments(parser, obs_required=False)
    _add_assimilation_option_arguments(parser, with_obs=False)
    parser.add_argument(
        "--outdir", type=Path, required=True,
        help="where the analysis fetch (analysis/), the stream fetches "
             "(fetch/), the cycle and the receipts are written",
    )
    parser.add_argument(
        "--analysis-grib", type=Path, default=None,
        help="an analysis GRIB already on disk (with --analysis-cycle) "
             "instead of fetching the newest GDAS cycle",
    )
    parser.add_argument(
        "--analysis-cycle", default=None,
        help="ISO-8601 instant of the analysis cycle to fetch or of "
             "--analysis-grib; default the newest published GDAS cycle",
    )
    parser.add_argument(
        "--start-utc", default=None,
        help="for an analytic base config (a smoke or OSSE case, nothing "
             "fetched): the instant model time zero stands for",
    )
    parser.add_argument(
        "--until-utc", default=None,
        help="the newest observation hour to cycle to (ISO-8601); default the "
             "current hour minus --observation-latency-s, floored to the hour",
    )
    parser.add_argument(
        "--observation-latency-s", type=float, default=3600.0,
        help="how far behind real time the observation streams are complete "
             "(default 3600)",
    )
    parser.add_argument(
        "--forecast-hours", type=float, default=24.0,
        help="the forecast length the derived config allows after the last "
             "analysis (default 24)",
    )
    parser.add_argument(
        "--interval-s", type=float, default=3600.0,
        help="model time between analyses in seconds (default 3600)",
    )
    parser.add_argument(
        "--members", type=int, default=None,
        help="ensemble members when fresh has to init (default 32 under "
             "letkf, 1 under successive-correction)",
    )
    _add_filter_argument(parser, with_ensemble=True)
    _add_cycle_setting_arguments(parser)
    parser.add_argument(
        "--cutoff-utc", default=None,
        help="the information cutoff (ISO-8601): the analysis handed back is "
             "the latest constructible from information available by then "
             "(default now); the newest observation hour is the cutoff minus "
             "--observation-latency-s, floored",
    )
    parser.add_argument(
        "--fetch-engine", choices=("auto", "rust", "python"), default="auto",
        help="the fetch engine for the GDAS analysis (default auto: rw_fetch "
             "when built)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace the derived config, the ensemble and the cycle in --outdir",
    )
    parser.add_argument(
        "--keep-backgrounds", action="store_true",
        help="also write each analysis hour's background checkpoint (the free "
             "forecast to the instant, beside the analysis handed back), so the "
             "increment can be read from the two files; what is analysed does "
             "not change",
    )


def _add_da_forecast_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config", type=config_argument,
        help="the experiment TOML the analysis was formed under (fresh writes "
             "fresh-config.toml)",
    )
    parser.add_argument(
        "--analysis", type=Path, required=True,
        help="the analysis checkpoint to start from (the DA receipt's "
             "analysis_checkpoint)",
    )
    parser.add_argument(
        "--outdir", type=Path, required=True,
        help="where the forecast checkpoints and receipts are written",
    )
    parser.add_argument(
        "--until-s", type=float, default=None,
        help="stop the forecast at this model time from the run's start "
             "(default the config's duration_s)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace this run's own files in --outdir",
    )


_DA_STATIC_HELP = (
    "estimate the static background-error covariance of the hybrid filter "
    "from lagged-forecast pairs (the 24 h forecast minus the 12 h forecast "
    "valid at one instant, one pair per analysis time): the per-degree "
    "variances of the model's spectral variables, the vertical correlations "
    "per wavenumber band and the linear-balance regressions, written as a "
    "versioned table with its receipt and charts"
)


def _add_da_static_covariance_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config", type=config_argument,
        help="the experiment TOML the forecasts ran under, or a shipped "
             "experiment's bare name (the transform and the level ladder the "
             "table is estimated on)",
    )
    parser.add_argument(
        "--pair", action="append", required=True, metavar="LATER,EARLIER",
        help="one lagged pair: the checkpoint of the longer forecast and the "
             "checkpoint of the shorter one valid at the same instant "
             "(f024,f012); repeatable, at least two pairs",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="where the table (static-covariance.npz), its receipt and its "
             "charts are written",
    )
    parser.add_argument(
        "--version", default=None,
        help="the table's version label carried in the receipt (default the "
             "sample's date range)",
    )
    parser.add_argument(
        "--ridge", type=float, default=1.0e-3,
        help="the relative ridge of the balance regressions (default 1e-3)",
    )
    parser.add_argument(
        "--no-charts", action="store_true",
        help="skip the variance-spectrum and correlation charts",
    )
    parser.add_argument(
        "--backend", choices=("numpy", "cupy"), default=None,
        help="the array backend the transform runs on (default the config's; "
             "numpy estimates on a host without a card)",
    )
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float64",
        help="the transform precision of the estimation (default float64, "
             "whatever the run's precision)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace an existing table in --out",
    )


def _add_da_localisation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config", type=config_argument,
        help="the experiment TOML the ensemble was built under, or a shipped "
             "experiment's bare name (its vertical coordinate)",
    )
    parser.add_argument(
        "--ensemble", type=Path, required=True,
        help="the ensemble store (the directory holding the member checkpoints "
             "and their manifest, or the manifest itself)",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="where the derivation receipt is written (JSON)",
    )
    parser.add_argument(
        "--step", type=int, default=None,
        help="read the members at this step instead of the manifest's",
    )


def _add_da_arguments(parser: argparse.ArgumentParser) -> None:
    """The six legs of `woof global da`, declared once for both doors."""
    legs = parser.add_subparsers(dest="da_command", required=True)
    _add_da_init_arguments(legs.add_parser("init", help=_DA_INIT_HELP))
    _add_da_cycle_arguments(legs.add_parser("cycle", help=_DA_CYCLE_HELP))
    _add_da_analyze_arguments(legs.add_parser("analyze", help=_DA_ANALYZE_HELP))
    _add_da_fresh_arguments(legs.add_parser("fresh", help=_DA_FRESH_HELP))
    _add_da_forecast_arguments(legs.add_parser("forecast", help=_DA_FORECAST_HELP))
    _add_da_static_covariance_arguments(legs.add_parser("static-covariance", help=_DA_STATIC_HELP))
    _add_da_localisation_arguments(legs.add_parser("localisation", help=_DA_LOCALISATION_HELP))
def _add_microwave_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "microwave_args", nargs=argparse.REMAINDER,
        help="the microwave door's own subcommand and arguments: fetch, decode, thin, columns, "
             "calibrate, score (each with --help)",
    )


def _microwave(args: argparse.Namespace) -> int:
    from .microwave.__main__ import main as microwave_main

    return microwave_main(list(args.microwave_args))


def _add_export_wrfout_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config", type=config_argument,
        help="the experiment TOML the checkpoints were run under, or a shipped "
             "experiment's bare name")
    parser.add_argument(
        "checkpoints", nargs="+", type=Path,
        help="checkpoints to export, one tape each, in time order",
    )
    parser.add_argument(
        "--outdir", type=Path, required=True, help="where the tapes are written")
    parser.add_argument(
        "--nlat", type=int, default=360,
        help="latitude points of the regular output grid (default 360)",
    )
    parser.add_argument(
        "--nlon", type=int, default=720,
        help="longitude points of the regular output grid (default 720)",
    )
    parser.add_argument(
        "--start-date", required=True,
        help="analysis valid time as YYYY-MM-DD_HH:MM:SS; checkpoint times offset from it",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace tapes that already exist")
    parser.add_argument(
        "--bbox", type=float, nargs=4, default=None,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="crop the tape to a lat/lon window (degrees, lon in -180..180)",
    )


def _add_abi_score_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config", type=config_argument,
        help="the experiment TOML the checkpoint was run under, or a shipped "
             "experiment's bare name")
    parser.add_argument(
        "checkpoint", type=Path, help="the checkpoint whose state is rendered (an analysis or a forecast)")
    parser.add_argument(
        "--start-date", required=True,
        help="analysis valid time as YYYY-MM-DD_HH:MM:SS; the checkpoint time offsets from it",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="where tapes, planes, packs, tables, charts and the receipt go")
    parser.add_argument(
        "--goes-rad", action="append", required=True, metavar="BAND=FILE",
        help="an ABI-L1b-Rad granule of the scan for one band, e.g. 13=OR_ABI-L1b-RadF-M6C13_...nc; repeatable",
    )
    parser.add_argument(
        "--goes-acm", type=Path, default=None,
        help="the ABI-L2-ACM clear-sky mask of the same scan (without it no pixel is obs-clear or obs-cloudy)",
    )
    parser.add_argument(
        "--goes-cmip", action="append", default=[], metavar="BAND=FILE",
        help="an ABI-L2-CMIP granule of the same band and scan, cross-checking the L1b inversion; repeatable",
    )
    parser.add_argument(
        "--tile", action="append", default=None, metavar="LABEL,LATMIN,LATMAX,LONMIN,LONMAX",
        help="a render tile (every corner must be on the visible disk); default: the four GOES-East "
             "tiles of abi_operator.DEFAULT_TILES",
    )
    parser.add_argument(
        "--bands", default="13,8", help="ABI bands to score, comma-separated (default 13,8)")
    parser.add_argument(
        "--rw-goes", type=Path, default=None, help="the rw_goes front door (else WOOF_RW_GOES, the tree, PATH)")
    parser.add_argument(
        "--simsat-cli", type=Path, default=None,
        help="simsat-render-ir, for the condensate mask that classes simulated columns clear or cloudy",
    )
    parser.add_argument("--threads", type=int, default=None, help="SimSat render threads (default: rayon's)")
    parser.add_argument(
        "--block", type=int, default=24, help="block side in lattice pixels for the block table (default 24)")
    parser.add_argument(
        "--zenith-max", type=float, default=None, help="drop pairs beyond this satellite zenith angle (deg)")
    parser.add_argument(
        "--calibrate", action="store_true",
        help="also plant skin and upper-vapor changes in the checkpoint and read them back in both bands",
    )
    parser.add_argument(
        "--goes-received", action="append", default=[], metavar="BAND=ISO",
        help="when this system first held the band's radiance granule (the fetch manifest's fetched_at), "
             "recorded in the pack's provenance row; repeatable")
    parser.add_argument("--no-charts", action="store_true", help="skip the matplotlib analysis charts")
    parser.add_argument("--no-reuse-tapes", action="store_true", help="export every tape again")
    parser.add_argument("--nlat", type=int, default=720, help="latitude points of the tape grid (default 720)")
    parser.add_argument("--nlon", type=int, default=1440, help="longitude points of the tape grid (default 1440)")


def _add_abi_reference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("mode", choices=("columns", "score"),
                        help="columns: write the gpuwm-da.abi-columns.v1 stream; score: read CRTM outputs and score")
    parser.add_argument(
        "--blocks", action="append", required=True, metavar="BAND=CSV",
        help="the block table rw_goes colocate wrote for a band (band13-blocks.csv); repeatable")
    parser.add_argument("--out", type=Path, required=True,
                        help="columns: the stream to write (its .json and .npz sidecars beside it); score: the directory")
    parser.add_argument("--zenith-max", type=float, default=70.0, help="columns: keep blocks to this zenith (deg)")
    parser.add_argument("--config", type=config_argument, default=None,
                        help="columns: the experiment TOML, or a shipped "
                             "experiment's bare name (the vertical coordinate)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="columns: the checkpoint whose surface planes are read")
    parser.add_argument("--tapes", default=None, help="columns: glob of the export tapes the blocks were rendered from")
    parser.add_argument("--valid", default=None, help="columns: the analysis valid time YYYY-MM-DD_HH:MM:SS (season)")
    parser.add_argument(
        "--emissivity", action="append", default=None, metavar="BAND=VALUE",
        help="columns: the per-band surface emissivity carried for the reference's user-emissivity run "
             "(default the operator's own 0.99 for every band asked)")
    parser.add_argument("--columns", type=Path, default=None, help="score: the columns .npz the columns mode wrote")
    parser.add_argument("--run", action="append", default=None, metavar="NAME=OUT.bin",
                        help="score: a CRTM output stream by name; repeatable")
    parser.add_argument("--primary", default=None, help="score: the run with CRTM's own surface models (the reference)")
    parser.add_argument("--simsat-emissivity-run", default=None,
                        help="score: the run at the operator's emissivity (the absorption term reads against it)")
    parser.add_argument("--zenith-gate", type=float, default=60.0, help="score: the gate's zenith bound (deg)")
    parser.add_argument("--gate-k", type=float, default=None, help="score: the gate (default the operator's 1.5 K)")
    parser.add_argument("--no-charts", action="store_true", help="score: skip the analysis charts")
    parser.add_argument("--table", type=Path, default=None,
                        help="score: the fast-model table of the primary run; with it the door writes the operator entries the "
                             "scorecard admits (operator-entries.json) and the four assessments")
    parser.add_argument("--reference-run", default=None,
                        help="score: with --table, the run that is the numerical reference (CRTM with its own surface models); "
                             "the entries' Jacobian agreement and brightness-temperature difference are measured against it")


def _add_abi_fast_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("mode", choices=("train", "forward"),
                        help="train: fit the table from a reference run; forward: run rw_goes forward on a columns stream")
    parser.add_argument("--columns", type=Path, required=True,
                        help="train: the columns .npz the abi-reference columns door wrote; forward: the columns .bin stream")
    parser.add_argument("--out", type=Path, required=True, help="train: the table JSON; forward: the output stream")
    parser.add_argument("--reference", type=Path, default=None,
                        help="train: the reference run (gpuwm-da.abi-crtm.v1) whose layer optical depths are fitted")
    parser.add_argument("--pack", action="append", default=None, metavar="BAND=GOESPACK",
                        help="train: the gpuwm-obs.goes-bt.v1 pack whose Planck row the band uses; repeatable")
    parser.add_argument("--planck", action="append", default=None, metavar="BAND=fk1,fk2,bc1,bc2",
                        help="train: Planck constants given directly (instead of --pack); repeatable")
    parser.add_argument("--form", action="append", default=None, metavar="BAND=linear|two_term",
                        help="train: the layer model form per band (default 13=linear, others two_term)")
    parser.add_argument("--provenance", type=Path, default=None,
                        help="train: a JSON file of reference provenance (CRTM version, coefficient hashes) copied into the table")
    parser.add_argument("--satellite", default="G19", help="train: the instrument the Planck rows belong to (default G19)")
    parser.add_argument("--table", type=Path, default=None, help="forward: the coefficient table")
    parser.add_argument("--rw-goes", type=Path, default=None, help="forward: the rw_goes front door")
    parser.add_argument("--emis-mode", type=int, default=0, help="forward: 0 the table's emissivity, 1 the columns' own")
    parser.add_argument("--bands", default=None, help="forward: bands to evaluate, comma-separated (default: the table's)")
    parser.add_argument("--threads", type=int, default=None, help="forward: worker threads")


def _add_statics_arguments(parser: argparse.ArgumentParser) -> None:
    from .statics import SECTOR_DEGREES

    parser.add_argument(
        "config", type=config_argument,
        help="the experiment TOML whose grid (truncation, nlat, nlon, "
             "dealias) and [statics] table select the build",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="cache file to write; the default is the path the run itself "
             "reads, <cache_dir>/arwen-global-statics-T<t>-<nlat>x<nlon>-"
             "<geog_data_res>.npz (a different --out is not found by the "
             "run unless [statics] cache_dir names its directory)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace an existing cache and its sidecar",
    )
    parser.add_argument(
        "--sector-degrees", type=float, default=SECTOR_DEGREES,
        help="longitude width of one build sector (default "
             f"{SECTOR_DEGREES:g}; bounds the source window read at once)",
    )
    parser.add_argument(
        "--geog-root", type=Path, default=None,
        help="the WPS_GEOG archive to build from, overriding [statics] "
             "geog_root for this build only.  Without it the config's own "
             "value is used, and without that the engine's archive search "
             "answers.  A flag is here because the archive is a property of "
             "the MACHINE and the config is a property of the EXPERIMENT: "
             "pinning a machine path inside a shipped experiment is how a "
             "config stops working on the next machine",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="woof global",
        description=(
            "Research-only moist hybrid global spherical-harmonic model, "
            "native WOOF CUDA physics adapter, and one-way regional parent bridge."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pins", help="print the Level-5 arithmetic/coupling identity")
    sub.add_parser("physics-manifest", help="print admitted native physics adapters")

    # The Rust side.  `doctor` reports every door against what is staged;
    # `fetch-doors` stages the eight this package publishes; `obs` is the
    # front door onto the observation streams those eight decode.  Without
    # `obs` the streams are reachable only by importing a module, which by
    # the rule that a capability a user cannot reach does not exist means
    # eight decoders and eleven streams were not shipped.
    from .analysis_fetch import add_fetch_analysis_arguments
    from .doctor import add_doctor_arguments
    from .fetch_doors import add_fetch_doors_arguments
    from .obs_streams import add_stream_commands

    add_doctor_arguments(sub.add_parser(
        "doctor",
        help="what is installed, what is staged, and what stops which command"))
    add_fetch_doors_arguments(sub.add_parser(
        "fetch-doors",
        help="stage the Rust doors this package publishes, verified against "
             "the pins this release carries"))
    add_fetch_analysis_arguments(sub.add_parser(
        "fetch-analysis",
        help="fetch one whole-globe GDAS analysis, the object a global cold "
             "start needs, and print the run command it feeds"))
    observations = sub.add_parser(
        "obs",
        help="the observation streams the assimilation reads: fetch and decode "
             "one stream through its Rust door, cut hourly tables, list the "
             "streams and the background anchors")
    add_stream_commands(observations.add_subparsers(
        dest="obs_command", required=True))

    # The machine seam.  `run-plan` is the versioned front door a desktop
    # application or a terminal workspace drives -- the same contract the
    # engine's own `woof run-plan` publishes, so a client that already drives
    # the engine drives this by changing the module it spawns -- and `sources`
    # is the human view of the registry that door serves as JSON.  Without
    # them the only way to drive this package from a program is to scrape the
    # console output of a command written for a person, which is the shape
    # every one of these documents exists to replace.
    from .runplan import register_cli as register_run_plan
    from .sources import add_sources_arguments

    register_run_plan(sub)
    add_sources_arguments(sub.add_parser(
        "sources",
        help="every source this model initializes from or scores against, "
             "and which authority table answered its mapping, the engine's "
             "or this package's carried copy -- the human view of "
             "`run-plan --sources`"))

    transform = sub.add_parser("transform-check", help="run transform/Parseval controls")
    transform.add_argument("--truncation", type=int, default=15)
    transform.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    transform.add_argument("--precision", choices=("float32", "float64"), default="float64")
    transform.add_argument("--dealias-factor", type=float, default=1.5)

    _add_run_arguments(sub.add_parser("run", help=_RUN_HELP))
    _add_statics_arguments(sub.add_parser("statics", help=_STATICS_HELP))
    _add_assimilate_arguments(
        sub.add_parser("assimilate", help=_ASSIMILATE_HELP))
    _add_cycle_arguments(sub.add_parser("cycle", help=_CYCLE_HELP))
    _add_da_arguments(sub.add_parser("da", help=_DA_HELP))

    inspect = sub.add_parser("inspect", help="validate and print checkpoint metadata")
    inspect.add_argument("checkpoint", type=Path)

    receipt = sub.add_parser("check-receipt", help="validate a self-hashed run receipt")
    receipt.add_argument("receipt", type=Path)

    export = sub.add_parser("export-parent", help="create a complete regular-lat/lon parent export")
    export.add_argument("config", type=config_argument)
    export.add_argument("checkpoint", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--nlat", type=int, required=True)
    export.add_argument("--nlon", type=int, required=True)
    export.add_argument("--overwrite", action="store_true")

    inspect_export = sub.add_parser("inspect-export", help="validate and print parent-export metadata")
    inspect_export.add_argument("input", type=Path)

    # `export` is the spelling a reader of this door wants -- the render-ready
    # tape is the product -- and `export-wrfout` stays as an alias because it
    # is the name in every written instruction that predates this
    # distribution.  A name that vanishes turns a working instruction into an
    # argument error.
    _add_export_wrfout_arguments(
        sub.add_parser("export", aliases=["export-wrfout"],
                       help=_EXPORT_WRFOUT_HELP))
    _add_microwave_arguments(sub.add_parser("microwave", help=_MICROWAVE_HELP))
    _add_abi_score_arguments(sub.add_parser("abi-score", help=_ABI_SCORE_HELP))
    _add_abi_reference_arguments(sub.add_parser("abi-reference", help=_ABI_REFERENCE_HELP))
    _add_abi_fast_model_arguments(sub.add_parser("abi-fast-model", help=_ABI_FAST_MODEL_HELP))

    migrate = sub.add_parser(
        "migrate-level4-checkpoint",
        help="explicitly migrate a Level-4 checkpoint and seed missing moments",
    )
    migrate.add_argument(
        "config", type=config_argument,
        help="target run config, or the name of a shipped experiment")
    migrate.add_argument("input", type=Path)
    migrate.add_argument("output", type=Path)
    migrate.add_argument("--receipt", type=Path, default=None)
    migrate.add_argument("--allow-native-zero-moments", action="store_true")
    migrate.add_argument("--overwrite", action="store_true")

    check_migration = sub.add_parser("check-migration", help="validate a Level-4 migration receipt")
    check_migration.add_argument("receipt", type=Path)

    target = sub.add_parser(
        "make-regional-target",
        help="wrap a prepared target-grid NPZ in the hash-bound target contract",
    )
    target.add_argument("input", type=Path)
    target.add_argument("output", type=Path)
    target.add_argument("--name", required=True)
    target.add_argument("--grid-id", required=True)
    target.add_argument("--source-identity-json", type=Path, default=None)
    target.add_argument("--overwrite", action="store_true")

    inspect_target = sub.add_parser("inspect-regional-target", help="validate regional target metadata")
    inspect_target.add_argument("input", type=Path)

    frame = sub.add_parser("translate-regional-frame", help="translate one global export to a WOOF frame")
    frame.add_argument("parent", type=Path)
    frame.add_argument("target", type=Path)
    frame.add_argument("output", type=Path)
    frame.add_argument("--overwrite", action="store_true")

    inspect_frame = sub.add_parser("inspect-regional-frame", help="validate translated frame metadata")
    inspect_frame.add_argument("input", type=Path)

    series = sub.add_parser("make-parent-series", help="bind two or more translated frames into an LBC series")
    series.add_argument("target", type=Path)
    series.add_argument("output", type=Path)
    series.add_argument("frames", nargs="+", type=Path)
    series.add_argument("--overwrite", action="store_true")

    inspect_series = sub.add_parser("inspect-parent-series", help="validate a translated parent series")
    inspect_series.add_argument("input", type=Path)

    qualify = sub.add_parser("native-qualify", help="run the target-device native physics qualification battery")
    qualify.add_argument("config", type=config_argument)
    qualify.add_argument("--outdir", type=Path, required=True)
    qualify.add_argument(
        "--overwrite", action="store_true",
        help=(
            "replace this door's own artifacts in --outdir "
            "(native-device-evidence.json, native-contract-candidate.json, "
            "continuous/, resumed/); other files are left untouched"
        ),
    )

    check_evidence = sub.add_parser("check-native-evidence", help="validate native device evidence")
    check_evidence.add_argument("input", type=Path)
    check_candidate = sub.add_parser("check-native-candidate", help="validate experimental native adapter candidate")
    check_candidate.add_argument("input", type=Path)

    # Three more doors this distribution adds, beside `doctor` and
    # `fetch-doors` above.  Each earns its place under the rule that a
    # capability a user cannot reach does not exist: `render` and `go` put a
    # command in front of two things the model could already do and nobody
    # could ask for in one line, and `configs` names what shipped.
    add_go_arguments(sub.add_parser("go", help=_GO_HELP))
    add_render_arguments(sub.add_parser("render", help=_RENDER_HELP))
    configs = sub.add_parser("configs", help=_CONFIGS_HELP)
    configs.add_argument(
        "--paths", action="store_true",
        help="print full paths instead of names")
    return parser


def _require_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output {path} exists; pass --overwrite to replace it")


def _npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _door_name(args: argparse.Namespace) -> str:
    """The command a reader can re-run, which is now one command.

    Every sentence these handlers print -- the refusals below and the
    gate-of-record report in :func:`_assimilate` -- is prefixed with a
    command the reader is meant to be able to type again.  While this model
    lived inside the engine there were two entrances and the prefix had to
    say which one had been used, because a message that sends half the
    readers to a command they did not type is worse than no prefix.

    There is one entrance now: :data:`arwen_global.CONSOLE_SCRIPT`.  The
    engine's ``woof global`` still reaches these same handlers through the
    forwarder, and it registers THIS package's parser, so a reader who came
    that way is told the spelling that works everywhere rather than one that
    only works when the engine is installed.

    ``args`` is kept in the signature: it is what a per-command prefix would
    read, and the callers all have it.
    """

    return CONSOLE_SCRIPT


def _size_the_run(args: argparse.Namespace, cfg, door: str):
    """Apply the memory levers, price the card, and hand the run the plan.

    ONE PATH FOR ALL THREE DOORS.  ``run``, ``cycle`` and ``assimilate``
    all build the same model on the same card, and until this existed
    only ``run`` was sized: ``cycle`` priced the card and then threw the
    answer away, and ``assimilate`` never asked.  So a T533 cycle got its
    band count from inside ``build_model_and_cold_state``, which reads the
    card AFTER the Legendre tables are on it -- a different reading, a
    different count, and a receipt describing a run that did not happen.

    Returns the config the run will use.  Raises
    :class:`sizing.GlobalMemoryRefusal` when the card cannot hold it.
    """
    from .sizing import GlobalMemoryRefusal, refusal_sentence, run_memory_gate

    # The levers move BEFORE the run is sized, because the gate prices the
    # transform the run will build and a lever named on the command line
    # and applied after the gate would be a flag the door read and the
    # estimate did not.
    overrides = _memory_lever_overrides(args)
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
        # FLUSHED, as is the verdict below: a rank the door admits and a
        # signal later ends (a peer that never comes and a launcher's
        # timeout, 2026-09-07 13:56Z on the RTX 5070 Ti; a rank 0 ended
        # by its lane when rank 1 was refused, 13:25Z) left a log with no
        # line at all, because a normal exit is what flushes a block
        # buffered stdout and a refusal is the only door verdict that
        # gets one.
        print(f"{door}: memory levers " + ", ".join(
            f"{key}={value}" for key, value in sorted(overrides.items())),
            flush=True)
    # SIZED BEFORE ANYTHING IS ALLOCATED.  The first thing a run does is
    # build the Legendre tables, which at T533 are 2.55 GiB and are the
    # largest single allocation of the whole forecast; a card that cannot
    # hold them used to find out as a CuPy OutOfMemoryError raised from
    # inside `SphericalHarmonicTransform.__post_init__`, after the config
    # had been accepted and the output directory prepared.
    gate = run_memory_gate(cfg)
    print(f"{door}: {gate['verdict']}", flush=True)
    if gate.get("unfitted"):
        raise GlobalMemoryRefusal(gate["unfitted"])
    if gate["refuse"]:
        raise GlobalMemoryRefusal(refusal_sentence(
            gate["estimate"], gate["free_bytes"], config_path=args.config,
            plan=gate.get("plan")))
    # The plan the gate priced is the plan the run takes.  The gate read
    # the card BEFORE anything was allocated; the builder would read it
    # after the Legendre tables are on it and choose differently, and a
    # run whose door priced one schedule and whose builder ran another is
    # a receipt that does not describe the run.
    plan = gate.get("plan")
    chosen = gate.get("latitude_bands")
    if chosen and not int(getattr(cfg, "latitude_bands", 0) or 0):
        cfg = dataclasses.replace(cfg, latitude_bands=int(chosen))
    if (plan is not None and plan.spill_chosen_by == "sizer"
            and str(getattr(cfg, "host_spill", "auto")) == "auto"):
        # The tier's own decision is made again inside the builder, off
        # the state's real arrays; what the door fixes here is the MODE,
        # so a run the door sized with nothing parked cannot acquire a
        # tier later, and one the door sized around the tier cannot lose
        # it.  The slices themselves stay the census's to choose.
        cfg = dataclasses.replace(
            cfg, host_spill="auto" if plan.spill_slices else "off")
    # A card cannot run half a band: the band is the unit a reduction
    # buffer and a waist are written in, so the schedule must have at
    # least one band per card.  The door raises the count and SAYS it did,
    # rather than letting the model refuse after the tables are allocated.
    ranks = int(getattr(cfg, "cards", 1) or 1)
    if ranks > 1 and int(getattr(cfg, "latitude_bands", 0) or 0) < ranks:
        cfg = dataclasses.replace(cfg, latitude_bands=ranks)
        print(
            f"{door}: latitude_bands raised to {ranks} so every card owns "
            "whole bands (a band is the unit a reduction buffer and a "
            "Fourier waist are written in)"
        )
    return cfg, plan


#: What `run` leaves beside its checkpoints so a later command knows what it
#: is looking at.  Two files: a BYTE COPY of the config, because an identity
#: hash cannot be re-loaded and a path can move, and a small index naming it.
RUN_SIDECAR_NAME = "arwen-global-run.json"
RUN_CONFIG_COPY_NAME = "arwen-global-run-config.toml"


def write_run_sidecar(outdir, config_path) -> None:
    """Copy the config into the run directory and index it.

    Costs one small file per run.  What it buys: `woof global render RUNDIR`
    works without re-stating the experiment, the run directory can be moved or
    handed to somebody else and still says what made it, and a receipt whose
    config identity does not match the copy beside it is a visible
    contradiction rather than an unanswerable question.

    Failure here never fails a run.  The forecast is on disk; a sidecar that
    could not be written is a convenience that was not, and saying so is
    better than discarding the run.
    """

    outdir = Path(outdir)
    source = Path(config_path)
    try:
        outdir.mkdir(parents=True, exist_ok=True)
        copy = outdir / RUN_CONFIG_COPY_NAME
        copy.write_bytes(source.read_bytes())
        (outdir / RUN_SIDECAR_NAME).write_text(json.dumps({
            "schema": "gpuwm-global-run-sidecar-v1",
            "config": str(source),
            "config_copy": copy.name,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        pass


def read_run_sidecar(outdir):
    """The config a run directory was run under, or ``None``."""

    outdir = Path(outdir)
    index = outdir / RUN_SIDECAR_NAME
    if not index.is_file():
        return None
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    name = payload.get("config_copy")
    if not name:
        return None
    copy = outdir / name
    return copy if copy.is_file() else None


def _run(args: argparse.Namespace) -> int:
    cfg, plan = _size_the_run(args, load_config(args.config), _door_name(args))
    # The config travels with the run.  Written after the sizer has accepted
    # it, so a directory that carries a config is a directory a run actually
    # started in, and never after a refusal.
    write_run_sidecar(args.outdir, args.config)
    # The machine surface, for anything watching this run rather than reading
    # its console: `go` and `render` wrote one and the FORECAST DOOR, the
    # longest-running command in the distribution, did not.  A workspace
    # driving a day-long integration had a growing log to scrape and no
    # answer to "what stage, how far, and did it end".
    from .status import StatusWriter

    status = StatusWriter(args.outdir, "run", stages=("forecast",))
    status.stage("forecast")
    # A run states its own vertical grid before it starts: the receipt
    # carries the full layer table under "vertical".
    print(f"{_door_name(args)}: {vertical_grid_sentence(cfg)}")
    # ...and the planet its land surface runs on: real WPS_GEOG statics
    # (with the cache it will read) or the declared synthetic constants.
    from .statics import statics_sentence

    print(f"{_door_name(args)}: {statics_sentence(cfg)}")

    def progress(diag):
        line = (
            f"step={diag['step']} time={diag['time_s']:.0f}s "
            f"max_wind={diag['maximum_wind_m_s']:.3f}m/s "
            f"water={diag['global_mean_total_water_kg_m2']:.6f}kg/m2"
        )
        print(line)
        status.step(int(diag["step"]), note=line)

    from .device_memory import device_peak_sentence
    from .runner import RECEIPT_NAME

    try:
        result = run(
            cfg,
            args.outdir,
            restart=args.restart,
            overwrite=args.overwrite,
            progress=progress,
            until_s=getattr(args, "until_s", None),
            profile_steps=int(getattr(args, "profile_steps", 0) or 0),
            profile_warmup=int(getattr(args, "profile_warmup", 2)),
            door_plan=plan,
            metal=bool(getattr(args, "metal", False)),
        )
    except BaseException:
        # The failure receipt carries the peak up to the death; an
        # out-of-memory run is exactly the one whose reader wants the
        # measured number on the console beside the traceback.
        receipt_path = Path(args.outdir) / RECEIPT_NAME
        try:
            failed = json.loads(receipt_path.read_text(encoding="utf-8"))
            row = failed.get("device_memory")
        except (OSError, ValueError):
            row = None
        if isinstance(row, dict):
            print(f"{_door_name(args)}: {device_peak_sentence(row)}")
        status.failed("the run did not finish; its failure receipt is beside "
                      "this file")
        raise
    # The MEASURED peak, after the run; the line printed before the run
    # was the sizing model's prediction and says so.
    print(f"{_door_name(args)}: {device_peak_sentence(result['device_memory'])}")
    print(json.dumps({
        "name": result["name"],
        "status": result["status"],
        "receipt": result["receipt_path"],
        "wall_seconds": result["wall_seconds"],
    }, indent=2, sort_keys=True))
    if result["status"] == "pass":
        status.done(f"receipt {result['receipt_path']}")
        return 0
    status.failed(f"the run ended {result['status']}; "
                  f"receipt {result['receipt_path']}")
    return 1


def _statics(args: argparse.Namespace) -> int:
    from woof.globe.spectral.grid import GaussianGrid

    from .statics import build_statics, cache_paths, write_cache

    cfg = load_config(args.config)
    door = _door_name(args)
    geog_root = getattr(args, "geog_root", None)
    if geog_root is not None:
        cfg = dataclasses.replace(
            cfg, statics=dataclasses.replace(
                cfg.statics, geog_root=str(geog_root)))
    if cfg.statics.source != "real":
        raise ValueError(
            f"{args.config} selects [statics] source = "
            f"{cfg.statics.source!r}; there is nothing to build for a "
            "synthetic planet (set source = \"real\" to build the "
            "WPS_GEOG statics for this grid)")
    grid = GaussianGrid.create(
        cfg.truncation, nlat=cfg.nlat, nlon=cfg.nlon,
        dealias_factor=cfg.dealias_factor,
    )
    default_out, _ = cache_paths(cfg.statics, grid)
    out = default_out if args.out is None else Path(args.out)
    for path in (out, out.with_suffix(".json")):
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"statics cache {path} exists; pass --overwrite to rebuild it")
    print(f"{door}: statics real T{grid.truncation} {grid.nlat}x{grid.nlon} "
          f"({'+'.join(cfg.statics.tokens())}) -> {out}")
    if out != default_out:
        print(f"{door}: note: the run reads {default_out}; point [statics] "
              "cache_dir at this --out directory for it to be found")

    def progress(done, total, seconds):
        print(f"{door}: sector {done}/{total} built in {seconds:.1f} s")

    fields, provenance = build_statics(
        grid, cfg.statics, sector_degrees=args.sector_degrees,
        progress=progress,
    )
    npz_path, sidecar = write_cache(out, fields, provenance, overwrite=args.overwrite)
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    print(json.dumps({
        "cache": str(npz_path),
        "sidecar": str(sidecar),
        "self_sha256": document["self_sha256"],
        "truncation": document["truncation"],
        "nlat": document["nlat"],
        "nlon": document["nlon"],
        "geog_root": document["geog_root"],
        "sectors": len(document["sectors"]),
        "wall_seconds": document["wall_seconds"],
        "arrays": len(document["arrays"]),
    }, indent=2, sort_keys=True))
    return 0


def _assimilation_options_from_args(args: argparse.Namespace):
    """The door's AssimilationOptions with the flags a reader typed applied
    (the defaults live in one place, the dataclass)."""
    import dataclasses

    from .assimilate import AssimilationOptions

    options = AssimilationOptions()
    overrides = {}
    if args.length_scale_km is not None:
        overrides["length_scale_km"] = args.length_scale_km
    if args.max_age_minutes is not None:
        overrides["maximum_age_s"] = args.max_age_minutes * 60.0
    if args.elevation_limit_m is not None:
        overrides["elevation_limit_m"] = args.elevation_limit_m
    if args.wind_balance is not None:
        overrides["wind_balance"] = args.wind_balance
    if getattr(args, "moisture_update", None) is not None:
        overrides["moisture_update"] = args.moisture_update == "on"
    if getattr(args, "humidity_decay_height_m", None) is not None:
        overrides["humidity_decay_height_m"] = args.humidity_decay_height_m
    if overrides:
        options = dataclasses.replace(options, **overrides)
    return options


def _assimilate(args: argparse.Namespace) -> int:
    from .assimilate import assimilate

    # The analysis builds the same model on the same card the forecast
    # does -- it synthesises the background to the grid, repairs
    # positivity and re-analyses -- so it is sized by the same door.
    # Before this it was the one door that never asked, and a T533
    # analysis on a card that could not hold it found out in the
    # allocator.
    cfg, plan = _size_the_run(args, load_config(args.config), _door_name(args))
    options = _assimilation_options_from_args(args)
    report = assimilate(
        cfg,
        args.checkpoint,
        args.obs,
        args.out,
        analysis_time=args.analysis_time,
        options=options,
        overwrite=args.overwrite,
        door_plan=plan,
    )
    print(json.dumps({
        "status": report["status"],
        "analysis": report["analysis"]["path"],
        "report": report["report_path"],
        "assimilated_total": report["assimilated_total"],
        "withheld_total": report["withheld_total"],
        "refused_from_chain": report["rejections"]["already_assimilated"],
        "failed_variables": report["gate_of_record"]["failed_variables"],
        "variables": {
            name: {
                "count": row["count"],
                "o_minus_b_rms": row["o_minus_b"]["rms"],
                "o_minus_a_rms": row["o_minus_a"]["rms"],
                "withheld_count": row["withheld"]["count"],
                "withheld_o_minus_b_rms": row["withheld"]["o_minus_b"]["rms"],
                "withheld_o_minus_a_rms": row["withheld"]["o_minus_a"]["rms"],
            }
            for name, row in report["variables"].items()
        },
    }, indent=2, sort_keys=True))
    if report["status"] != "pass":
        print(
            f"{_door_name(args)}: gate of record failed for "
            f"{report['gate_of_record']['failed_variables']}: an "
            "analysis that fits the observations worse than the "
            "background is not an analysis",
            file=sys.stderr,
        )
        return 1
    return 0


def _cycle(args: argparse.Namespace) -> int:
    door = _door_name(args)
    cfg, plan = _size_the_run(args, load_config(args.config), door)
    print(f"{door}: {vertical_grid_sentence(cfg)}")
    from .statics import statics_sentence

    print(f"{door}: {statics_sentence(cfg)}")
    from .cycle import cycle

    options = _assimilation_options_from_args(args)

    def progress(diag):
        print(
            f"step={diag['step']} time={diag['time_s']:.0f}s "
            f"max_wind={diag['maximum_wind_m_s']:.3f}m/s "
            f"water={diag['global_mean_total_water_kg_m2']:.6f}kg/m2",
            flush=True,
        )

    def analysis_progress(record):
        rows = " ".join(
            f"{name}={row['withheld_o_minus_b_rms']:.3f}>{row['withheld_o_minus_a_rms']:.3f}"
            for name, row in record["variables"].items()
        )
        verdict = "applied" if record["applied"] else (
            f"CARRIED (gate failed: {record['failed_variables']})")
        if record.get("dropped_variables"):
            verdict += f" without {record['dropped_variables']} (failed, withdrawn)"
        print(
            f"analysis step={record['step']} at {record['analysis_time_utc']}: "
            f"{verdict}, {record['assimilated_total']} reports, "
            f"{record['timings_s']['analysis_s']:.1f}s; withheld O-B>O-A rms {rows}",
            flush=True,
        )

    from .device_memory import device_peak_sentence
    from .runner import RECEIPT_NAME

    try:
        result = cycle(
            cfg, args.outdir,
            obs_locations=args.obs, cycles=args.cycles, start_utc=args.start_utc,
            interval_s=args.interval_s, restart=args.restart,
            overwrite=args.overwrite, options=options, until_s=args.until_s,
            keep_backgrounds=args.keep_backgrounds,
            partial_analyses=args.partial_analyses == "on", progress=progress,
            analysis_progress=analysis_progress,
            door_plan=plan,
        )
    except BaseException:
        receipt_path = Path(args.outdir) / RECEIPT_NAME
        try:
            failed = json.loads(receipt_path.read_text(encoding="utf-8"))
            row = failed.get("device_memory")
        except (OSError, ValueError):
            row = None
        if isinstance(row, dict):
            print(f"{door}: {device_peak_sentence(row)}")
        raise
    print(f"{door}: {device_peak_sentence(result['device_memory'])}")
    record = result["cycle"]
    print(json.dumps({
        "name": result["name"],
        "status": result["status"],
        "receipt": result["receipt_path"],
        "wall_seconds": result["wall_seconds"],
        "cycles_completed": record["completed"],
        "cycles_applied": record["applied"],
        "cycles_carried": record["carried"],
        "cycles_partial": record["partial"],
        "wall_seconds_per_model_hour_cycled": record.get(
            "wall_seconds_per_model_hour_cycled"),
        "analyses": [
            {"step": r["step"], "analysis_time_utc": r["analysis_time_utc"],
             "status": r["status"], "analysis": None if r["analysis"] is None
             else r["analysis"]["path"]}
            for r in record["analyses"]
        ],
    }, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


def _da_progress(door: str):
    def progress(diag):
        # This one callable is handed to two different producers.  The model
        # integrator reports a STEP, as a mapping of measured numbers.  The
        # engine's fetch doors -- which `da fresh` calls before there is a
        # model at all, to bring the analysis down -- report a SENTENCE, as a
        # plain string, and `woof.fetch` has done so since it had a progress
        # argument.  A callback that subscripts whatever it is given turned
        # the first of those sentences into `TypeError: string indices must
        # be integers`, which this door's refusal layer then printed as
        # `woof global: string indices must be integers, not 'str'`: a
        # sentence naming neither the command, the stage, nor a remedy, at
        # the first door of the documented `da fresh` quickstart whenever the
        # analysis is fetched rather than handed in with --analysis-grib.
        if not isinstance(diag, dict):
            print(f"{door}: {diag}", flush=True)
            return
        print(
            f"step={diag['step']} time={diag['time_s']:.0f}s "
            f"max_wind={diag['maximum_wind_m_s']:.3f}m/s "
            f"water={diag['global_mean_total_water_kg_m2']:.6f}kg/m2",
            flush=True,
        )

    def analysis_progress(record):
        from .da_scorecard import render_table

        verdict = "applied" if record["applied"] else (
            f"CARRIED (gate failed: {record['failed_variables']})")
        if record.get("dropped_variables"):
            verdict += f" without {record['dropped_variables']} (failed, withdrawn)"
        budget = record.get("budget") or {}
        print(
            f"{door}: analysis step={record['step']} at {record['analysis_time_utc']}: "
            f"{verdict}, {record['assimilated_total']} reports, filter "
            f"{record.get('filter')}; wall {budget.get('wall_s', 0.0):.1f} s of a "
            f"{budget.get('interval_s', 0.0):.0f} s interval "
            f"({budget.get('real_time_fraction', 0.0):.3f} of real time)",
            flush=True,
        )
        if isinstance(record.get("scorecard"), dict):
            print(render_table(record["scorecard"]), flush=True)

    return progress, analysis_progress


def _da_summary(door: str, receipt: dict) -> None:
    handed = receipt.get("analysis_checkpoint")
    print(json.dumps({
        "door": receipt["door"],
        "status": receipt["status"],
        "receipt": receipt["receipt_path"],
        "analysis_checkpoint": None if handed is None else handed["path"],
        "analysis_self_sha256": None if handed is None else handed["self_sha256"],
        "wall_seconds": receipt["wall_seconds"],
        **({"cycles": receipt["cycles"]} if "cycles" in receipt else {}),
        **({"forecast_command": receipt["forecast_command"]}
           if receipt.get("forecast_command") else {}),
    }, indent=2, sort_keys=True))
    scorecards = receipt.get("scorecards")
    if isinstance(scorecards, dict) and scorecards.get("cycles"):
        print(
            f"{door}: scorecard: {scorecards['complete_cycles']} of "
            f"{len(scorecards['cycles'])} cycles engineering-complete"
            + (f"; incomplete: {scorecards['incomplete_cycles']}"
               if scorecards["incomplete_cycles"] else "")
        )
        not_below = [
            f"{source}/{variable}: {entry['o_a_not_below_o_b_cycles']}"
            for source, table in scorecards.get("streams", {}).items()
            for variable, entry in table.items()
            if entry.get("o_a_not_below_o_b_cycles")
        ]
        if not_below:
            print(f"{door}: O-A above O-B (a reading, not a failure) on " + "; ".join(not_below))
    causal = receipt.get("causal")
    if isinstance(causal, dict) and causal.get("latency_classes"):
        print(
            f"{door}: information: {causal['mode']}; cutoff {causal.get('information_cutoff_utc')}; "
            f"latency classes {causal['latency_classes']}"
        )
    budget = receipt.get("budget")
    if isinstance(budget, dict) and budget.get("cycles"):
        print(
            f"{door}: wall budget: mean {budget['mean_wall_s']:.1f} s, max "
            f"{budget['max_wall_s']:.1f} s per {budget['interval_s']:.0f} s cycle "
            f"(max {budget['max_real_time_fraction']:.3f} of real time; "
            f"{'every cycle keeps up' if budget['every_cycle_keeps_up'] else 'a cycle fell behind real time'})"
        )


def _da(args: argparse.Namespace) -> int:
    from . import da_door

    door = f"{_door_name(args)} da {args.da_command}"
    if args.da_command == "init":
        cfg = load_config(args.config)
        receipt = da_door.init(
            cfg, args.outdir, filter_name=args.filter, members=args.members,
            from_checkpoint=args.from_checkpoint, analysis_time_utc=args.analysis_time,
            overwrite=args.overwrite, config_path=args.config,
            ensemble_truncation=args.ensemble_truncation,
            control_options=_control_options_from_args(args),
            additive_inflation_fraction=args.additive_inflation,
            filter_overrides=_filter_overrides_from_args(args),
            filter_settings=_filter_settings_from_args(args),
        )
        _da_summary(door, receipt)
        return 0 if receipt["status"] == "pass" else 1
    if args.da_command == "analyze":
        cfg = load_config(args.config)
        receipt = da_door.analyze(
            cfg, args.checkpoint, args.obs, args.out, analysis_time=args.analysis_time,
            options=_assimilation_options_from_args(args), overwrite=args.overwrite,
            filter_name=args.filter, config_path=args.config,
        )
        _da_summary(door, receipt)
        if receipt["status"] == "fail":
            print(
                f"{door}: gate of record failed for "
                f"{receipt['gate_of_record']['failed_variables']}: an analysis "
                "that fits the withheld observations worse than the background "
                "is not an analysis",
                file=sys.stderr,
            )
            return 1
        return 0
    if args.da_command == "static-covariance":
        from .da_static import estimate_table

        cfg = load_config(args.config)
        pairs = []
        for value in args.pair:
            parts = [part.strip() for part in str(value).split(",")]
            if len(parts) != 2 or not all(parts):
                raise ValueError(f"--pair expects LATER,EARLIER, got {value!r}")
            pairs.append((Path(parts[0]), Path(parts[1])))
        receipt = estimate_table(
            cfg, pairs, args.out, version=args.version, ridge=args.ridge, charts=not args.no_charts,
            overwrite=args.overwrite, progress=lambda line: print(f"{door}: {line}", flush=True),
            config_path=args.config, backend=args.backend, precision=args.precision,
        )
        print(f"{door}: static covariance T{receipt['truncation']} from {receipt['samples']} pairs "
              f"written to {receipt['table']} (sha256 {receipt['sha256'][:16]})")
        return 0 if receipt.get("status") == "pass" else 1
    if args.da_command == "localisation":
        cfg = load_config(args.config)
        receipt = da_door.localisation(
            cfg, args.ensemble, args.out, step=args.step, config_path=args.config,
        )
        for region, rows in receipt["cutoffs"].items():
            for option, row in rows.items():
                residual = row.get("fit_rms_residual")
                note = "no signal above the sampling floor" if row.get("no_signal_above_floor") else f"fit residual {residual:.3f}"
                print(f"{door}: {region}: {option} = {row['cutoff_lnp']:.3f} ln p "
                      f"({row['class']}, {note}, "
                      f"filter carries {receipt['carried_by_filter_options'].get(option)})")
        print(f"{door}: {receipt['members']} members at T{receipt['truncation']}; receipt {args.out}")
        return 0
    if args.da_command == "forecast":
        cfg = load_config(args.config)
        print(f"{door}: {vertical_grid_sentence(cfg)}")
        progress, _ = _da_progress(door)
        receipt = da_door.forecast(
            cfg, args.outdir, analysis=args.analysis, until_s=args.until_s,
            overwrite=args.overwrite, progress=progress, config_path=args.config,
        )
        _da_summary(door, receipt)
        return 0 if receipt["status"] == "pass" else 1
    from .sizing import GlobalMemoryRefusal, refusal_sentence, run_memory_gate

    progress, analysis_progress = _da_progress(door)
    if args.da_command == "cycle":
        cfg = load_config(args.config)
        gate = run_memory_gate(cfg)
        print(f"{door}: {gate['verdict']}")
        if gate["refuse"]:
            raise GlobalMemoryRefusal(refusal_sentence(
                gate["estimate"], gate["free_bytes"], config_path=args.config))
        print(f"{door}: {vertical_grid_sentence(cfg)}")
        receipt = da_door.cycle(
            cfg, args.outdir, obs_locations=args.obs, stream_specs=args.stream,
            cycles=args.cycles, start_utc=args.start_utc, interval_s=args.interval_s,
            restart=args.restart, ensemble=args.ensemble,
            options=_assimilation_options_from_args(args), until_s=args.until_s,
            keep_backgrounds=args.keep_backgrounds,
            partial_analyses=args.partial_analyses == "on", filter_name=args.filter,
            overwrite=args.overwrite, progress=progress, metal=bool(args.metal),
            analysis_progress=analysis_progress, config_path=args.config,
            ensemble_truncation=args.ensemble_truncation,
            control_options=_control_options_from_args(args),
            additive_inflation_fraction=args.additive_inflation,
            filter_settings=_filter_settings_from_args(args),
            observation_bin_s=args.observation_bin_s, anchor_spec=args.anchor,
            increment_application=args.increment_application,
            filter_overrides=_filter_overrides_from_args(args),
        )
        _da_summary(door, receipt)
        return 0 if receipt["status"] in ("pass", "incomplete") else 1
    if args.da_command == "fresh":
        receipt = da_door.fresh(
            args.config, args.outdir, stream_specs=args.stream, obs_locations=args.obs,
            analysis_grib=None if args.analysis_grib is None else str(args.analysis_grib),
            analysis_cycle=args.analysis_cycle, start_utc=args.start_utc,
            until_utc=args.until_utc, forecast_hours=args.forecast_hours,
            interval_s=args.interval_s, filter_name=args.filter, members=args.members,
            options=_assimilation_options_from_args(args),
            observation_latency_s=args.observation_latency_s, overwrite=args.overwrite,
            progress=progress, analysis_progress=analysis_progress,
            fetch_engine=args.fetch_engine, ensemble_truncation=args.ensemble_truncation,
            control_options=_control_options_from_args(args),
            additive_inflation_fraction=args.additive_inflation,
            filter_settings=_filter_settings_from_args(args),
            observation_bin_s=args.observation_bin_s, anchor_spec=args.anchor,
            increment_application=args.increment_application, cutoff_utc=args.cutoff_utc,
            filter_overrides=_filter_overrides_from_args(args),
            defaulted_names=("hybrid_beta",) if getattr(args, "hybrid_beta", None) is None else (),
            keep_backgrounds=args.keep_backgrounds,
        )
        _da_summary(door, receipt)
        return 0 if receipt["status"] in ("pass", "incomplete") else 1
    raise AssertionError(args.da_command)
def _parse_band_files(values, flag: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for value in values or ():
        band, sep, path = str(value).partition("=")
        if not sep or not band.strip().isdigit() or not path:
            raise ValueError(f"{flag} expects BAND=FILE, got {value!r}")
        out[int(band)] = Path(path)
    return out


def _parse_tiles(values):
    from .abi_operator import DEFAULT_TILES

    if not values:
        return DEFAULT_TILES
    tiles = []
    for value in values:
        parts = [part.strip() for part in str(value).split(",")]
        if len(parts) != 5:
            raise ValueError(f"--tile expects LABEL,LATMIN,LATMAX,LONMIN,LONMAX, got {value!r}")
        label, *numbers = parts
        tiles.append((label, *(float(number) for number in numbers)))
    return tuple(tiles)


def _abi_score(args: argparse.Namespace) -> int:
    from .abi_operator import run_case

    cfg = load_config(args.config)
    bands = tuple(int(part) for part in str(args.bands).split(",") if part.strip())
    receipt = run_case(
        cfg, args.checkpoint, start_date=args.start_date, out_dir=args.out,
        goes_rad=_parse_band_files(args.goes_rad, "--goes-rad"), goes_acm=args.goes_acm,
        goes_cmip=_parse_band_files(args.goes_cmip, "--goes-cmip"), tiles=_parse_tiles(args.tile),
        bands=bands, rw_goes=args.rw_goes, simsat_cli=args.simsat_cli, threads=args.threads,
        block=args.block, zenith_max_deg=args.zenith_max, calibrate=args.calibrate,
        charts=not args.no_charts, nlat=args.nlat, nlon=args.nlon, reuse_tapes=not args.no_reuse_tapes,
        received_utc={int(k): str(v) for k, v in _parse_band_files(args.goes_received, "--goes-received").items()},
    )
    summary = {
        "receipt": str(Path(args.out) / "abi-operator-receipt.json"),
        "verdicts": receipt["verdicts"],
        "gate": receipt["gate"],
        "scores": {
            band: {
                key: {k: score["classes"][key][k] for k in ("n", "bias_k", "rmse_k", "rmse_after_k")}
                for key in (f"{receipt['gate']['class']}/{receipt['gate']['zenith']}", "all/all")
                if key in score["classes"]
            }
            for band, score in receipt["scores"].items()
        },
        "operator_entries": {band: (entry["name"] if entry else None)
                             for band, entry in receipt["operator_entries"].items()},
        "calibration": (receipt["calibration"].get("count") if "calibration" in receipt else None),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all(v == "PASS" for v in receipt["verdicts"].values()) else 1


def _abi_reference(args: argparse.Namespace) -> int:
    from . import abi_reference as ref
    from .abi_operator import BANDS, CLEAR_SKY_GATE_K

    blocks_csv = _parse_band_files(args.blocks, "--blocks")
    if args.mode == "columns":
        missing = [name for name in ("config", "checkpoint", "tapes", "valid") if getattr(args, name) is None]
        if missing:
            raise ValueError(f"abi-reference columns needs --{' --'.join(missing)}")
        from .constants import KAPPA, REFERENCE_PRESSURE_PA
        cfg = load_config(args.config)
        blocks = ref.read_block_tables(blocks_csv, zenith_max_deg=args.zenith_max)
        tapes = ref.read_tapes(sorted(glob.glob(args.tapes)))
        if not tapes:
            raise ValueError(f"--tapes {args.tapes!r} matches no file")
        if args.emissivity:
            emissivity = {}
            for part in args.emissivity:
                band, _, value = str(part).partition("=")
                if not value:
                    raise ValueError(f"--emissivity wants BAND=VALUE, got {part!r}")
                emissivity[int(band)] = float(value)
        else:
            emissivity = {int(b): 0.99 for b in blocks["bands"]}
        month = int(args.valid[5:7])
        columns = ref.build_columns(
            blocks, tapes, a_half_pa=cfg.a_half_pa, b_half=cfg.b_half, checkpoint=args.checkpoint, month=month,
            kappa=KAPPA, reference_pressure_pa=REFERENCE_PRESSURE_PA, emissivity_by_band=emissivity)
        provenance = {"config": str(args.config), "config_hash": cfg.config_hash, "checkpoint": str(args.checkpoint),
                      "tapes": [t.path for t in tapes], "blocks": {str(k): str(v) for k, v in blocks_csv.items()},
                      "valid": args.valid, "zenith_max_deg": args.zenith_max,
                      "operator_bands": {str(b): BANDS[b].name for b in blocks["bands"] if b in BANDS}}
        ref.write_columns(args.out, columns, provenance=provenance)
        npz = {k: v for k, v in columns.items() if isinstance(v, np.ndarray)}
        npz["bands"] = np.asarray(blocks["bands"])
        for band in blocks["bands"]:
            npz[f"obs_{band}"] = blocks[f"obs_{band}"]
            npz[f"sim_{band}"] = blocks[f"sim_{band}"]
        npz["n_both_clear"] = blocks["n_both_clear"]
        npz["n_pair"] = blocks["n_pair"]
        np.savez(Path(f"{args.out}.npz"), **npz)
        print(json.dumps({"columns": str(args.out), "sidecar": f"{args.out}.json", "arrays": f"{args.out}.npz",
                          "n": int(columns["lat"].size), "checks": columns["checks"]}, indent=2, sort_keys=True))
        return 0
    missing = [name for name in ("columns", "run", "primary") if getattr(args, name) is None]
    if missing:
        raise ValueError(f"abi-reference score needs --{' --'.join(missing)}")
    saved = np.load(args.columns, allow_pickle=False)
    columns = {k: saved[k] for k in saved.files}
    bands = [int(b) for b in columns.pop("bands")]
    blocks = {"bands": bands, "n_both_clear": columns.pop("n_both_clear")}
    if "n_pair" in columns:
        blocks["n_pair"] = columns.pop("n_pair")
    for band in bands:
        blocks[f"obs_{band}"] = columns.pop(f"obs_{band}")
        blocks[f"sim_{band}"] = columns.pop(f"sim_{band}")
    run_paths = dict(part.split("=", 1) for part in args.run)
    runs = {name: ref.read_crtm_output(path) for name, path in run_paths.items()}
    if args.primary not in runs:
        raise ValueError(f"--primary {args.primary!r} is not one of the runs {sorted(runs)}")
    if args.simsat_emissivity_run and args.simsat_emissivity_run not in runs:
        raise ValueError(f"--simsat-emissivity-run {args.simsat_emissivity_run!r} is not one of the runs {sorted(runs)}")
    if args.table is not None and args.reference_run is None:
        raise ValueError(
            "abi-reference score with --table writes the operator entries, and their Jacobian agreement is measured against "
            f"the reference run: pass --reference-run NAME (one of {sorted(n for n in runs if n != args.primary)})")
    if args.reference_run is not None and (args.reference_run not in runs or args.reference_run == args.primary):
        raise ValueError(f"--reference-run {args.reference_run!r} must be one of the runs other than the primary: "
                         f"{sorted(n for n in runs if n != args.primary)}")
    gate_k = CLEAR_SKY_GATE_K if args.gate_k is None else float(args.gate_k)
    score = ref.score_reference(blocks, columns, runs, primary=args.primary,
                                simsat_emissivity_run=args.simsat_emissivity_run,
                                zenith_gate_deg=args.zenith_gate, gate_k=gate_k)
    score["runs"] = {name: {"channels": run["channels"], "emissivity_mode": run["emissivity_mode"],
                            "refused": run["refused"], "path": run_paths[name]}
                     for name, run in runs.items()}
    score["columns"] = str(args.columns)
    score["reference_run"] = args.reference_run
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.no_charts:
        score["charts"] = ref.write_reference_charts(args.out, score, columns, blocks, runs, primary=args.primary,
                                                     simsat_emissivity_run=args.simsat_emissivity_run)
    entries = None
    if args.table is not None:
        from . import abi_fast_model as fm
        from .abi_operator import fast_operator_entries, four_assessments
        table = fm.read_table(args.table)
        entries = fast_operator_entries(score, table, gate_k=gate_k, operator_run=args.primary, reference_run=args.reference_run)
        admitted = {b: row["admitted_classes"] for b, row in entries["bands"].items()}
        shapes = {b: {c: row["classes"][c]["filter_facing"].get("residual_shape_after_correction")
                      for c in row["admitted_classes"]}
                  for b, row in entries["bands"].items()}
        score["operator_entries"] = str(args.out / "operator-entries.json")
        score["assessments"] = four_assessments(
            engineering={"verdict": "PASS" if all(run["refused"] == 0 for run in runs.values()) else "FAIL",
                         "facts": {"runs": {name: {"columns": run["n"], "refused": run["refused"]} for name, run in runs.items()},
                                   "blocks_scored": int(columns["kept"].sum())}},
            statistical={"verdict": "PASS" if any(admitted.values()) else "FAIL",
                         "facts": {"admitted_classes_by_band": admitted,
                                   "population": score["stream_qc"],
                                   "rule": "a class is admitted when the operator's after-correction rmse against the instrument, "
                                           "on the blocks the stream hands the filter (one block one row, unweighted), is inside "
                                           f"{gate_k} K on at least 1000 blocks; the per-class residuals are the O-B distribution "
                                           "of this stream against this analysis",
                                   "residual_shape_after_correction_by_admitted_class": shapes,
                                   "what_is_not_tested": "Desroziers consistency and the spread need an analysis and an ensemble; "
                                                         "the residual's tails and its correlation with zenith and latitude are "
                                                         "reported above and not gated"}},
            physical={"verdict": "NOT MEASURED",
                      "facts": {"note": "an operator alone changes no state; budgets and imbalance are assessed where the analysis "
                                        "applies its increment"}},
            predictive={"verdict": "NOT MEASURED",
                        "facts": {"note": "forecast skill with this stream assimilated is the DA door's first arm, not the operator's "
                                          "measurement"}},
        )
        ref.write_json(args.out / "operator-entries.json", entries)
    ref.write_json(args.out / "abi-reference-receipt.json", score)
    summary = {
        band: {
            "reference_gate": row["reference_gate"],
            "all": {k: {kk: v.get(kk) for kk in ("n", "bias_k", "rmse_k", "rmse_after_k", "fit_slope")}
                    for k, v in row["classes"]["all"].items()},
            "water": {k: {kk: v.get(kk) for kk in ("n", "bias_k", "rmse_k", "rmse_after_k")}
                      for k, v in row["classes"]["water"].items()},
            "land": {k: {kk: v.get(kk) for kk in ("n", "bias_k", "rmse_k", "rmse_after_k")}
                     for k, v in row["classes"]["land"].items()},
            "jacobians": {k: row["jacobians"][k] for k in ("skin_jacobian_mean", "peak_pressure_hpa_percentiles_5_25_50_75_95",
                                                          "half_sensitivity_band_hpa_median", "sensitivity_below_500hpa_fraction_mean")},
        }
        for band, row in score["bands"].items()
    }
    if entries is not None:
        summary["operator_entries"] = {b: row["admitted_classes"] for b, row in entries["bands"].items()}
        summary["assessments"] = {k: v["verdict"] for k, v in score["assessments"].items()}
    print(json.dumps({"receipt": str(args.out / "abi-reference-receipt.json"), "bands": summary}, indent=2, sort_keys=True))
    if entries is not None:
        return 0 if any(row["admitted_classes"] for row in entries["bands"].values()) else 1
    return 0 if all(row["reference_gate"]["verdict"] == "PASS" for row in score["bands"].values()) else 1


def _abi_fast_model(args: argparse.Namespace) -> int:
    from . import abi_fast_model as fm
    from . import abi_reference as ref

    if args.mode == "forward":
        from .abi_radiance_operator import run_forward
        if args.table is None:
            raise ValueError("abi-fast-model forward needs --table")
        bands = tuple(int(b) for b in str(args.bands).split(",") if b.strip()) if args.bands else None
        record = run_forward(args.columns, args.table, args.out, rw_goes=args.rw_goes, emis_mode=args.emis_mode,
                             bands=bands, threads=args.threads)
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    if args.reference is None:
        raise ValueError("abi-fast-model train needs --reference (a gpuwm-da.abi-crtm.v1 run)")
    saved = np.load(args.columns, allow_pickle=False)
    columns = {k: saved[k] for k in saved.files}
    if "a_half_pa" not in columns:
        raise ValueError(f"{args.columns} carries no vertical coordinate; rebuild it with the v2 columns door")
    reference = ref.read_crtm_output(args.reference)
    planck: dict[int, dict] = {}
    planck_source: dict[str, str] = {}
    for part in args.pack or ():
        band, _, path = str(part).partition("=")
        from .obs_pack import read_goes_pack
        pack = read_goes_pack(path)
        meta = pack.meta if hasattr(pack, "meta") else pack
        row = meta["planck"]
        planck[int(band)] = {k: float(row[k]) for k in ("fk1", "fk2", "bc1", "bc2")}
        planck_source[band] = f"{meta.get('satellite')} {meta.get('sources', [{}])[0].get('filename', path)} planck_* attributes"
    for part in args.planck or ():
        band, _, values = str(part).partition("=")
        fk1, fk2, bc1, bc2 = (float(v) for v in values.split(","))
        planck[int(band)] = {"fk1": fk1, "fk2": fk2, "bc1": bc1, "bc2": bc2}
        planck_source[band] = "given on the command line"
    forms = {13: "linear"}
    for part in args.form or ():
        band, _, form = str(part).partition("=")
        forms[int(band)] = form
    bands = [int(b) for b in reference["channels"]]
    for band in bands:
        if band not in planck:
            raise ValueError(f"band {band} has no Planck row: pass --pack {band}=FILE.goespack or --planck {band}=fk1,fk2,bc1,bc2")
    lon = np.asarray(columns["lon"], dtype=np.float64)
    train = (np.round(lon * 4).astype(int) % 2 == 0)
    test = ~train
    table: dict = {
        "schema": fm.FAST_MODEL_SCHEMA,
        "satellite": args.satellite,
        "sensor": "abi",
        "vertical": fm.coordinate_identity(columns["a_half_pa"], columns["b_half"]),
        "features": {"names": list(fm.FEATURE_NAMES), "definitions": fm.FEATURE_DEFINITIONS, "constants": fm.FEATURE_CONSTANTS},
        "jacobian_steps": fm.JACOBIAN_STEPS,
        "training": {"columns": str(args.columns), "reference_run": str(args.reference), "split": "alternate 0.25-degree longitude columns",
                     "train_columns": int(train.sum()), "test_columns": int(test.sum()),
                     "columns_checks": json.loads(Path(f"{str(args.columns)[:-4]}.json").read_text(encoding="utf-8")).get("checks")
                     if Path(f"{str(args.columns)[:-4]}.json").exists() else None},
        "reference": json.loads(args.provenance.read_text(encoding="utf-8")) if args.provenance else {},
        "bands": {},
    }
    for band in bands:
        od = reference["layer_od"][band]
        form = forms.get(band, "two_term")
        band_table = fm.train_band(band, columns, od, train=train, form=form, planck=planck[band])
        band_table["planck"]["source"] = planck_source.get(str(band), "")
        band_table["emissivity"] = fm.emissivity_table_from_reference(reference, band, columns)
        emis_ref = np.asarray(reference["emissivity"][band], dtype=np.float64)
        band_table["validation"] = fm.validate_band(band_table, columns, od, test=test, emissivity=emis_ref)
        band_table["reference_jacobians"] = ref.jacobian_summary(
            reference["jac_t"][band], reference["jac_q"][band], reference["jac_tskin"][band],
            columns["p_half_hpa"], columns["p_full_hpa"])
        table["bands"][str(band)] = band_table
    fm.write_table(args.out, table)
    print(json.dumps({"table": str(args.out), "bands": {
        b: {"form": t["form"], "validation": {k: t["validation"][k] for k in ("test_rms_k", "test_bias_k", "test_p99_abs_k", "test_max_abs_k", "test_water_rms_k", "test_land_rms_k")},
            "emissivity": {"water": t["emissivity"]["water"], "land_classes_seen": sum(1 for c in t["emissivity"]["land_class_counts"] if c >= 5)}}
        for b, t in table["bands"].items()}}, indent=2, sort_keys=True))
    return 0


def _export_wrfout(args: argparse.Namespace) -> int:
    from .wrfout_export import EXPORT_RECEIPT_NAME, export_wrfout

    cfg = load_config(args.config)
    written = export_wrfout(
        cfg,
        args.checkpoints,
        args.outdir,
        nlat=args.nlat,
        nlon=args.nlon,
        start_date=args.start_date,
        overwrite=args.overwrite,
        bbox=tuple(args.bbox) if args.bbox else None,
    )
    receipt = json.loads(
        (Path(args.outdir) / EXPORT_RECEIPT_NAME).read_text(encoding="utf-8")
    )
    print(json.dumps({
        "tapes": [str(path) for path in written],
        "receipt": str(Path(args.outdir) / EXPORT_RECEIPT_NAME),
        # Every tape says where its T2/Q2/U10/V10 came from.
        "surface_diagnostics": {
            row["tape"]: row["surface_diagnostics"] for row in receipt["tapes"]
        },
    }, indent=2, sort_keys=True))
    return 0


#: Every outcome this model's doors express as a refusal rather than as a
#: crash: a config the loader will not accept, a checkpoint that is not
#: one, an output that already exists, a dependency that is not
#: installed, a device that answered with a NaN.
_REFUSALS = (
    ValueError,
    TypeError,
    RuntimeError,
    FloatingPointError,
    FileNotFoundError,
    FileExistsError,
    ModuleNotFoundError,
    OSError,
)

#: What the `woof global` leg has to convert into a sentence ITSELF.
#:
#: `woof.cli`'s boundary already prints a ValueError as `woof global:
#: <message>` at exit 2 -- with the `--explain` layering, which nothing
#: here can reproduce -- and derives an install remedy from a
#: ModuleNotFoundError's module name.  Both are better answers than this
#: door could compose, so those two rise to it untouched.
#:
#: The rest do NOT reach that boundary: it catches RuntimeError only for
#: the fetch family and catches no OSError at all, so `--overwrite`'s
#: FileExistsError, a missing checkpoint's FileNotFoundError and an
#: unreadable output directory's OSError would each leave `woof global`
#: as a traceback at exit 1 -- the exact shape this tree treats as a
#: defect, and the shape `woof global` has never had.
_DOOR_REFUSALS = tuple(
    error for error in _REFUSALS
    if error not in (ValueError, ModuleNotFoundError))


#: The exit codes this distribution's doors speak, and why each is separate.
#: A workspace, a shell script and a CI job all branch on these and none of
#: them can read a sentence.
#:
#:   0  the command did what it was asked
#:   1  a refusal: this package declined, and the first line of stderr says
#:      why while the rest says what to do about it
#:   2  the command line was not a command line (argparse's own code, kept so
#:      that a mistyped flag never looks like a refusal)
#:   3  a Rust door is missing or fails its pin; stderr names the binary and
#:      the bundle it comes from
#:   4  the device refused admission; stderr carries the measured free bytes
#:      and the bytes the run needs
#:
#: 3 and 4 are split out of 1 because they are the two refusals a caller can
#: ACT on without a human: stage a bundle, or ask for a smaller truncation.
#: Folded into 1 they are indistinguishable from a bad config, and the caller
#: has to parse prose to tell them apart.
EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_ARGUMENTS = 2
EXIT_DOOR = 3
EXIT_DEVICE = 4


def _door_is_missing(exc: BaseException) -> bool:
    """Whether this refusal is a Rust door that is not there.

    Recognised by the microwave bridge's own class plus the two literals the
    two resolvers actually emit: the engine's front-door `require()` and this
    package's own door refusals.  Matching a literal is fragile in general and
    is not fragile here, because both literals are asserted by tests in the
    trees that emit them and the fallback is exit 1, which is a correct
    refusal rather than a wrong answer.
    """

    try:
        from .microwave.atms_bridge import AtmsBridgeMissing
    except Exception:  # pragma: no cover - only if the module cannot import
        AtmsBridgeMissing = ()
    if AtmsBridgeMissing and isinstance(exc, AtmsBridgeMissing):
        return True
    from .doors import DoorMissing

    if isinstance(exc, DoorMissing):
        return True
    text = str(exc)
    return "is not built or not found" in text or "is not staged" in text


def _device_refused(exc: BaseException) -> bool:
    """Whether this refusal is the card declining the run."""

    try:
        from .sizing import GlobalMemoryRefusal
    except Exception:  # pragma: no cover
        return False
    return isinstance(exc, GlobalMemoryRefusal)


def exit_code_for(exc: BaseException) -> int:
    """The exit code one refusal earns."""

    if _device_refused(exc):
        return EXIT_DEVICE
    if _door_is_missing(exc):
        return EXIT_DOOR
    return EXIT_REFUSED


def _door(handler, args: argparse.Namespace) -> int:
    """One leg of `woof global`, under this door's refusal contract."""

    try:
        return handler(args)
    except _DOOR_REFUSALS as exc:
        print(f"{_door_name(args)}: {exc}", file=sys.stderr)
        return exit_code_for(exc)


def register_cli(subparsers) -> argparse.ArgumentParser:
    """Register `woof global` on the main command line.

    The three legs a reader runs -- integrate a forecast, assimilate
    point observations into a checkpoint, write render-ready tapes --
    delegating to the same handlers :func:`main` calls, from the same
    argument declarations.  The rest of the research surface (pins,
    checkpoint and export inspection, Level-4 migration, the one-way
    regional parent bridge, the native physics qualification battery)
    stays on `woof global`, which is unchanged.
    """

    parser = subparsers.add_parser(
        "global",
        help="experimental global spectral model: forecast, point-observation "
             "assimilation, render-ready wrfout export",
        description=(
            "EXPERIMENTAL.  A moist hybrid global spherical-harmonic model "
            "with its own physics suites and its own receipts.  It is a "
            "research door: its output is not a supported product and its "
            "configuration surface can move between releases.  The full "
            "research surface -- pins, checkpoint inspection, Level-4 "
            "migration, the regional parent bridge and the native physics "
            "qualification battery -- is reachable as "
            "`woof global`."),
    )
    commands = parser.add_subparsers(dest="global_command", required=True)

    run_parser = commands.add_parser("run", help=_RUN_HELP)
    _add_run_arguments(run_parser)
    run_parser.set_defaults(func=functools.partial(_door, _run))

    statics_parser = commands.add_parser("statics", help=_STATICS_HELP)
    _add_statics_arguments(statics_parser)
    statics_parser.set_defaults(func=functools.partial(_door, _statics))

    assimilate_parser = commands.add_parser(
        "assimilate", help=_ASSIMILATE_HELP)
    _add_assimilate_arguments(assimilate_parser)
    assimilate_parser.set_defaults(func=functools.partial(_door, _assimilate))

    cycle_parser = commands.add_parser("cycle", help=_CYCLE_HELP)
    _add_cycle_arguments(cycle_parser)
    cycle_parser.set_defaults(func=functools.partial(_door, _cycle))

    da_parser = commands.add_parser("da", help=_DA_HELP)
    _add_da_arguments(da_parser)
    da_parser.set_defaults(func=functools.partial(_door, _da))

    # `export`, not `export-wrfout`: the render-ready tape is the product
    # a reader of this door wants, and the parent export beside it is a
    # step in the regional bridge rather than something to look at.  That
    # one keeps its full name on the module door.
    export_parser = commands.add_parser(
        "export", help=_EXPORT_WRFOUT_HELP)
    _add_export_wrfout_arguments(export_parser)
    export_parser.set_defaults(func=functools.partial(_door, _export_wrfout))

    microwave_parser = commands.add_parser("microwave", help=_MICROWAVE_HELP)
    _add_microwave_arguments(microwave_parser)
    microwave_parser.set_defaults(func=functools.partial(_door, _microwave))
    abi_parser = commands.add_parser("abi-score", help=_ABI_SCORE_HELP)
    _add_abi_score_arguments(abi_parser)
    abi_parser.set_defaults(func=functools.partial(_door, _abi_score))

    reference_parser = commands.add_parser("abi-reference", help=_ABI_REFERENCE_HELP)
    _add_abi_reference_arguments(reference_parser)
    reference_parser.set_defaults(func=functools.partial(_door, _abi_reference))

    fast_parser = commands.add_parser("abi-fast-model", help=_ABI_FAST_MODEL_HELP)
    _add_abi_fast_model_arguments(fast_parser)
    fast_parser.set_defaults(func=functools.partial(_door, _abi_fast_model))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Point this process at the doors this package publishes before anything
    # resolves one.  It sets the engine's own per-binary variables, never
    # overwrites one the operator set, and `doctor` prints what it bound, so
    # a redirected binary is a stated fact rather than a discovered one.
    from .doors import bind_companion_doors

    bind_companion_doors()
    try:
        if args.command == "doctor":
            from .doctor import doctor

            return doctor(args)
        if args.command == "fetch-doors":
            from .fetch_doors import fetch_doors

            return fetch_doors(args)
        if args.command == "fetch-analysis":
            from .analysis_fetch import fetch_analysis_main

            return fetch_analysis_main(args)
        if args.command == "obs":
            return args.func(args)
        if args.command == "run-plan":
            from .runplan import run_plan_main

            return run_plan_main(args)
        if args.command == "sources":
            from .sources import sources_main

            return sources_main(args)
        if args.command == "pins":
            print(json.dumps(pins_receipt(), indent=2, sort_keys=True))
            return 0
        if args.command == "physics-manifest":
            ensure_builtin_global_physics_adapters()
            print(json.dumps(global_physics_manifest(), indent=2, sort_keys=True))
            return 0
        if args.command == "transform-check":
            from woof.globe.spectral.transform import SphericalHarmonicTransform

            transform = SphericalHarmonicTransform.create(
                args.truncation,
                backend=args.backend,
                precision=args.precision,
                dealias_factor=args.dealias_factor,
            )
            print(json.dumps(transform.transform_check(seed=23), indent=2, sort_keys=True))
            return 0
        if args.command == "run":
            return _run(args)
        if args.command == "statics":
            return _statics(args)
        if args.command == "assimilate":
            return _assimilate(args)
        if args.command == "cycle":
            return _cycle(args)
        if args.command == "da":
            return _da(args)
        if args.command == "inspect":
            metadata, _ = read_checkpoint(args.checkpoint)
            print(json.dumps(metadata, indent=2, sort_keys=True))
            return 0
        if args.command == "check-receipt":
            payload = check_receipt(args.receipt)
            print(json.dumps({
                "status": payload["status"],
                "self_sha256": payload["self_sha256"],
            }, indent=2, sort_keys=True))
            return 0 if payload["status"] == "pass" else 1
        if args.command == "export-parent":
            _require_output(args.output, args.overwrite)
            cfg = load_config(args.config)
            path = export_parent(
                cfg, args.checkpoint, args.output, nlat=args.nlat, nlon=args.nlon
            )
            metadata, _ = read_parent_export(path)
            print(json.dumps(metadata, indent=2, sort_keys=True))
            return 0
        if args.command == "inspect-export":
            metadata, _ = read_parent_export(args.input)
            print(json.dumps(metadata, indent=2, sort_keys=True))
            return 0
        if args.command in ("export", "export-wrfout"):
            return _export_wrfout(args)
        if args.command == "go":
            from .go_door import go

            return go(args)
        if args.command == "render":
            from .render_door import render

            return render(args)
        if args.command == "configs":
            names = list_configs()
            root = config_root()
            for name in names:
                print(str(root / f"{name}.toml") if args.paths else name)
            if not names:
                print(f"no experiments ship in this install (looked in {root})",
                      file=sys.stderr)
                return 1
            return 0
        if args.command == "microwave":
            return _microwave(args)
        if args.command == "abi-score":
            return _abi_score(args)
        if args.command == "abi-reference":
            return _abi_reference(args)
        if args.command == "abi-fast-model":
            return _abi_fast_model(args)
        if args.command == "migrate-level4-checkpoint":
            cfg = load_config(args.config)
            checkpoint, migration_receipt = migrate_level4_checkpoint(
                args.input,
                args.output,
                target_config=cfg,
                allow_native_zero_moments=args.allow_native_zero_moments,
                receipt_path=args.receipt,
                overwrite=args.overwrite,
            )
            payload = read_migration_receipt(migration_receipt)
            print(json.dumps({
                "checkpoint": str(checkpoint),
                "receipt": str(migration_receipt),
                "self_sha256": payload["self_sha256"],
            }, indent=2, sort_keys=True))
            return 0
        if args.command == "check-migration":
            print(json.dumps(read_migration_receipt(args.receipt), indent=2, sort_keys=True))
            return 0
        if args.command == "make-regional-target":
            _require_output(args.output, args.overwrite)
            arrays = _npz_arrays(args.input)
            identity = (
                {}
                if args.source_identity_json is None
                else json.loads(args.source_identity_json.read_text(encoding="utf-8"))
            )
            if not isinstance(identity, dict):
                raise ValueError("source identity JSON must be an object")
            path = write_regional_target(
                args.output,
                arrays,
                name=args.name,
                grid_id=args.grid_id,
                source_identity={
                    **identity,
                    "prepared_npz": str(args.input),
                },
            )
            metadata, _ = read_regional_target(path)
            print(json.dumps(metadata, indent=2, sort_keys=True))
            return 0
        if args.command == "inspect-regional-target":
            metadata, _ = read_regional_target(args.input)
            print(json.dumps(metadata, indent=2, sort_keys=True))
            return 0
        if args.command == "translate-regional-frame":
            _require_output(args.output, args.overwrite)
            path = translate_parent_to_regional_frame(
                args.parent, args.target, args.output
            )
            metadata, _ = read_regional_frame(path)
            print(json.dumps(metadata, indent=2, sort_keys=True))
            return 0
        if args.command == "inspect-regional-frame":
            metadata, _ = read_regional_frame(args.input)
            print(json.dumps(metadata, indent=2, sort_keys=True))
            return 0
        if args.command == "make-parent-series":
            _require_output(args.output, args.overwrite)
            if len(args.frames) < 2:
                raise ValueError("make-parent-series requires at least two frames")
            target_metadata, _ = read_regional_target(args.target)
            frames = []
            for frame_path in args.frames:
                metadata, _ = read_regional_frame(frame_path)
                if metadata["target_self_sha256"] != target_metadata["self_sha256"]:
                    raise ValueError(f"frame {frame_path} belongs to a different target")
                frames.append((frame_path, metadata))
            path = write_parent_series(
                args.output,
                frames,
                target_path=args.target,
                target_self_sha256=target_metadata["self_sha256"],
            )
            payload, _ = read_parent_series(path)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "inspect-parent-series":
            payload, resolved = read_parent_series(args.input)
            print(json.dumps({
                **payload,
                "resolved_frames": [str(row[0]) for row in resolved],
            }, indent=2, sort_keys=True))
            return 0
        if args.command == "native-qualify":
            cfg = load_config(args.config)
            evidence, candidate = qualify_native_adapter(
                cfg, args.outdir, overwrite=args.overwrite
            )
            payload = read_native_device_evidence(evidence)
            print(json.dumps({
                "evidence": str(evidence),
                "candidate": None if candidate is None else str(candidate),
                "status": payload["status"],
                "self_sha256": payload["self_sha256"],
            }, indent=2, sort_keys=True))
            return 0
        if args.command == "check-native-evidence":
            payload = read_native_device_evidence(args.input)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload["status"] == "pass" else 1
        if args.command == "check-native-candidate":
            print(json.dumps(
                read_native_contract_candidate(args.input), indent=2, sort_keys=True
            ))
            return 0
    except _REFUSALS as exc:
        print(f"{_door_name(args)}: {exc}", file=sys.stderr)
        return exit_code_for(exc)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
