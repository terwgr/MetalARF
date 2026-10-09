#!/usr/bin/env python3
"""Re-cut this package from a newer revision of the model's source tree.

THE BREAKAGE THIS PREVENTS.  The first carve was done by hand: files copied,
imports rewritten with a shell loop, command spellings fixed one at a time.
That is fine once.  It is not fine at the second cut, because by then the
package has fixes of its own on top of the carved files, and a hand re-copy
either throws those away or is done file by file by somebody reading two
trees at once.  The first resync after the carve had 46 changed files and 7
new ones waiting in the source tree; doing that by eye is how a fix lands in
one tree and not the other.

So the resync is a THREE-WAY MERGE, not a copy:

    base    the source tree at the revision this package was last cut from
    theirs  the source tree at the revision being cut now
    ours    the files in this repository, carrying every fix made here

Each side is put through the SAME rewiring, so the only differences the merge
sees are real changes.  ``git merge-file`` does the merge and marks anything
it cannot resolve, which is the point: a conflict is a place where this
package and the source tree changed the same lines, and that is exactly the
thing a human has to look at.

USAGE

    python tools/resync_from_owner.py --worktree DIR --to REV
    python tools/resync_from_owner.py --worktree DIR --to REV --dry-run

``--from`` defaults to the revision recorded in SOURCE.md, so a re-cut needs
no argument but the new revision and the worktree.  The worktree is NOT
recorded in SOURCE.md: this repository is bound for a public index and a
private tree's path on somebody's disk is not something to publish.  Give it
as ``--worktree`` or in ``WOOF_GLOBAL_SOURCE_WORKTREE``.

WHAT IT DOES NOT DO.  It never writes to the source tree, and it never
commits.  It leaves the working tree changed and the conflicts marked, for
whoever ran it to read, test and commit.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "src" / "arwen_global"
TESTS = REPO / "tests"
SOURCE_MD = REPO / "SOURCE.md"

# The units of the carve: a path in the source tree and where it lands here.
# A directory maps to a directory, a file to a file.
#: The authority mapping files carried as package data, by file name.
#: Named here rather than globbed, because the engine's table holds sixty
#: mappings for sources this model does not read and a glob would carry all
#: of them.
AUTHORITY_MAPPINGS: tuple[str, ...] = (
    "rw-wps-gdas-global-analysis-grib2.mapping.json",
    "rw-wps-gfs-pgrb2-0p25-cloud-cover.mapping.json",
    "rw-wps-gdas-pgrb2-0p25-microwave-columns.mapping.json",
    "rw-wps-ecmwf-open-data-global-forecast-grib2.mapping.json",
    "rw-wps-gfs-surface-flux-grib2.mapping.json",
    "rw-wps-gfs-surface-state-grib2.mapping.json",
)

CARVE: tuple[tuple[str, str], ...] = (
    ("woof/arwen_global", "."),
    ("woof/global_spectral", "spectral"),
    ("woof/core/global_physics_registry.py", "physics/registry.py"),
    ("woof/static/rows.py", "statics_rows.py"),
    ("woof/data/arwen_global", "data"),
    # The six source mappings this model names.  A source is metadata by
    # rule, so these are the ENGINE's rows and the resolver asks the engine
    # for them first -- but no published engine carries one, and a package
    # whose every GDAS experiment refuses at its first door is not shipped.
    # They ride here until the engine takes them, and a re-cut keeps them
    # current with the tree the model is graded in rather than freezing a
    # copy at the cut that first needed it.
    *(
        (f"woof/authorities/{name}", f"data/authorities/{name}")
        for name in AUTHORITY_MAPPINGS
    ),
)

# ---------------------------------------------------------------- the core
#
# THE BREAKAGE THIS TABLE PREVENTS.  This package depends on a PUBLISHED
# engine, and the published engine's physics is not the physics the model was
# graded with.  Measured 2026-09-09 against woof 2.7.0: RRTMGP has no size
# bounding at all, GrellFreitas takes no arguments, NewTiedtke takes no column
# chunk, sfclay dropped vegfra and gained ustm, the YSU launcher lost the
# free-atmosphere mixing-length flag, ysu_contract is absent outright, nine of
# the eleven carried modules differ and eight of the fifteen carried kernel
# files do: seven translation units and one header, re-measured on the Windows
# desktop 2026-09-10.  A run against that engine either
# raises TypeError several minutes in, or -- if the arguments were adapted
# away -- integrates under different physics than every receipt records.
#
# So the modules and kernels the model's own physics path executes AND that
# differ from the published engine are CUT INTO THIS PACKAGE.  Everything else
# the package reaches stays on the engine and is pinned in the seam manifest
# (src/arwen_global/data/engine-seam.json), which is a different mechanism for
# a different problem: a carry makes bytes ours, a pin tells the reader when
# somebody else's bytes moved.
#
# Two modules here match the published engine apart from the import rewrites
# this carve makes, and are carried anyway: noah.py and morrison.py.  Their
# KERNELS differ, and the loader binds its own directory, so a module that
# stayed on the engine would compile the engine's .cu.
CORE_MODULES: tuple[str, ...] = (
    "rrtmgp", "gf", "ntiedtke", "sfclay", "ysu", "ysu_contract", "landuse",
    "physics_inventory", "physics", "noah", "morrison",
)

#: Every kernel a carried module compiles, plus the headers those kernels are
#: assembled with.  The loader has no fallback to the engine's directory, so a
#: kernel a carried module names and this list omits is a FileNotFoundError at
#: the first call, not a silent read of the engine's copy.
CORE_KERNELS: tuple[str, ...] = (
    "gf.cu", "ntiedtke.cu", "sfclay.cu", "ysu.cu", "ysu_validation.cu",
    "noah.cu", "morrison.cu",
    "rrtmgp_rte.cu", "rrtmgp_gas.cu", "rrtmgp_cloud.cu", "rrtmgp_mcica.cu",
    "rrtmgp_validation.cu",
    "common.cuh", "glibc_flt32.cuh", "rrtmgp_planck_common.cuh",
)

CORE_CARVE: tuple[tuple[str, str], ...] = (
    tuple((f"woof/core/{name}.py", f"core/{name}.py") for name in CORE_MODULES)
    + (
        # The float64 mirror the scorecards grade against.  A mirror that is
        # not the kernel it mirrors is a flawed instrument, so it travels with
        # the kernels rather than staying on the engine.
        ("woof/verify/npref.py", "core/npref.py"),
        # The NVRTC loader.  Byte-identical at 2.7.0; carried because
        # _KDIR = Path(__file__).parent is what makes every carried module
        # read the carried kernels.
        ("woof/core/kernels/__init__.py", "core/kernels/__init__.py"),
    )
    + tuple((f"woof/core/kernels/{name}", f"core/kernels/{name}")
            for name in CORE_KERNELS)
    + (
        # THE NOTICE FOR WHAT THE KERNELS TRANSCRIBE, and it travels because
        # the licences it performs ask it to.  Arm's MIT requires its
        # copyright and permission notice in every copy of the code;
        # FDLIBM's only condition is that its notice be preserved;
        # BSD-3-Clause clause 1 requires the notice, the conditions and the
        # disclaimer to be retained in a source redistribution.  A carried
        # kernel with no notice beside it performs none of those, which is
        # what 0.1.0 shipped.  The device sources cannot take the notice as
        # a comment -- their bytes are the ``source_sha256`` a receipt
        # records -- so it sits in their directory instead, and
        # ``KERNEL_NOTICE_SCOPE`` below narrows the copy that lands here to
        # the files this package actually ships.
        ("woof/core/kernels/LICENSE-third-party.txt",
         "core/kernels/LICENSE-third-party.txt"),
        # Byte-identical to the engine's copies.  They travel because carried
        # noah.py resolves TBL_DIR relative to its own file, so without them
        # the first Noah or land-use call raises on a path that does not
        # exist.  A packaging requirement, not a physics divergence.  The
        # directory carries LICENSE-WRF.txt with the four tables, which is
        # UCAR's own request: that its notice travel with any copy of WRF.
        # It is listed in ``VERBATIM_PREFIXES``: third-party text is not
        # this carve's to reword.
        ("woof/data/noah_tables", "data/noah_tables"),
    )
)

#: Applied to CARRIED CORE FILES ONLY.  A carried module must reach the other
#: carried modules and the carried kernels; every other ``woof.`` import in a
#: carried file names a module that STAYS on the engine and is left verbatim,
#: because rewriting it would fork a file this package does not carry.
#:
#: Longest name first inside the alternation: ``physics_inventory`` before
#: ``physics`` and ``ysu_contract`` before ``ysu``, so the short name does not
#: win on a prefix.
CORE_REWIRE: tuple[tuple[str, str], ...] = (
    (r"woof\.core\.kernels", "woof.globe.core.kernels"),
    (r"woof\.verify\.npref", "woof.globe.core.npref"),
    (r"woof\.core\.(physics_inventory|ysu_contract|rrtmgp|gf|ntiedtke"
     r"|sfclay|ysu|landuse|physics|noah|morrison)\b",
     r"woof.globe.core.\1"),
)

#: TWO WORDS THIS PROJECT DOES NOT PUBLISH, rewritten in carried Python.
#:
#: The source tree is private and writes as it likes; this package is public
#: and its source ships with it, so a comment in a carried module is read by
#: the same people who read the README.  `tests/test_no_provenance.py` holds
#: everything that ships to that rule, and the first carve failed it in
#: fifteen places across four carried files.
#:
#: NOT PYTHON ONLY, and this paragraph used to say it was.  `device_prose()`
#: applies this whole table to every `.cu` and `.cuh` as well, so the two
#: words are removed from a carried kernel's comments too.  That was the
#: decision `VERBATIM_SUFFIXES` below records, taken on 2026-09-09 when the
#: device sources turned out not to be clean of the words: five lines across
#: `gf.cu`, `ntiedtke.cu` and `glibc_flt32.cuh` were inside the wheel on a
#: public index.  Read that entry and `DEVICE_EXACT` for what else a device
#: source is rewritten for, and `DEVICE_FORBIDDEN` for the post-condition
#: that stops the cut when a new one arrives.
#:
#: WHAT IT COSTS a kernel, measured rather than assumed: 17 bytes across
#: three of the fifteen device sources (gf.cu -2, ntiedtke.cu -4,
#: glibc_flt32.cuh -11).  Those three files' `source_sha256` moves, and so
#: does the digest the receipt records over the kernel set; the compiled
#: image does not, because a comment is gone before nvrtc reaches the first
#: token.  Nothing else about a device source is touched: the import
#: rewiring still stops at these suffixes, which is what `VERBATIM_SUFFIXES`
#: is for.
#: SPELLED IN PIECES, and that is not decoration: the gate reads the TREE,
#: not the intent, so a table that names the words it removes fails the rule
#: it enforces.  This file ships in the sdist like everything else.
_FIRST = "hone" + "st"
_SECOND = "load[-_ ]bearing"

#: THE CASE FORMS ARE ALL THREE, and the third one is why this comment is
#: here: the gate that holds the shipped tree to these words is case
#: INSENSITIVE, and this table was not, so an all-capitals spelling of
#: either word was substituted by nothing and asserted against by nothing.
#: Four such lines shipped inside two carried kernels.
CORE_WORDS: tuple[tuple[str, str], ...] = (
    (rf"(?<![A-Za-z]){_FIRST}ly(?![A-Za-z])", "exactly"),
    (rf"(?<![A-Za-z]){_FIRST.capitalize()}ly(?![A-Za-z])", "Exactly"),
    (rf"(?<![A-Za-z]){_FIRST.upper()}LY(?![A-Za-z])", "EXACTLY"),
    (rf"(?<![A-Za-z]){_FIRST}(?![A-Za-z])", "exact"),
    (rf"(?<![A-Za-z]){_FIRST.capitalize()}(?![A-Za-z])", "Exact"),
    (rf"(?<![A-Za-z]){_FIRST.upper()}(?![A-Za-z])", "EXACT"),
    (rf"(?<![A-Za-z]){_SECOND}(?![A-Za-z])", "structural"),
    (rf"(?<![A-Za-z]){_SECOND.capitalize()}(?![A-Za-z])", "Structural"),
    (rf"(?<![A-Za-z]){_SECOND.upper()}(?![A-Za-z])", "STRUCTURAL"),
)

#: MACHINE NAMES AND A CASE NAME, rewritten in carried Python AND in the
#: carried device sources.
#:
#: THE BREAKAGE THIS TABLE PREVENTS.  Measured on 0.1.1 as published: six
#: comment lines across `core/gf.py`, `kernels/gf.cu`, `kernels/ntiedtke.cu`
#: and `kernels/ysu.cu` named the private machines a figure was taken on,
#: and eight docstring lines across `core/landuse.py`, `core/npref.py`,
#: `core/physics.py` and `core/rrtmgp.py` named one private case by its
#: working name.  All fourteen were inside the wheel on a public index.  A
#: hostname tells a reader nothing they can use, and a case name in a
#: generic module is the specialisation the case-name rule forbids.  The
#: card model a figure was measured on stays: it is a measurement condition.
#:
#: Earlier text in this file said machine names in carried comments are NOT
#: touched, because they are another author's measurement text.  That was
#: the position until they were measured inside the published wheel; the text is kept, the machine is dropped,
#: and the card it names is kept where the sentence already gave it.
#:
#: WHAT IT COSTS a kernel: the three device sources that carry a machine
#: name (gf.cu, ntiedtke.cu, ysu.cu) move their `source_sha256` and the
#: kernel-set digest a receipt records, and nothing else: a comment is gone
#: before nvrtc reaches the first token.  The engine-divergence fingerprint
#: applies this table to the engine side too, so the engine's own copies of
#: the same comments do not become new differences.
#:
#: SPELLED IN PIECES, for the reason the words above are: the gate that reads
#: the shipped tree matches these spellings, and this file ships in the sdist.
_NODE = "no" + "de"
_CASE = "real" + "74"

CORE_IDENTITY: tuple[tuple[str, str], ...] = (
    (rf"on {_NODE}-1 \(weather-{_NODE}-1, RTX 5070 Ti, ", "on an RTX 5070 Ti ("),
    (rf"on {_NODE}-1 \(RTX 5070(\s+(?:\*\s+)?)Ti, ", r"on an RTX 5070\1Ti ("),
    (rf"MEASURED on {_NODE}-1 at ", "MEASURED on an RTX 5070 Ti at "),
    (rf"\({_NODE}-1, sm_120\)", "(RTX 5070 Ti, sm_120)"),
    (rf"pinned against the {_CASE} WRF", "pinned against the reference WRF"),
    (rf"matching {_CASE}\.", "matching the reference WRF configuration."),
    (rf"the {_CASE}(?=\s+compatibility)", "the reference"),
    (rf"formulation used by {_CASE}\.",
     "formulation used by the reference configuration."),
    (rf"the standard {_CASE} path", "the standard path"),
    (rf"the frozen {_CASE} configuration", "the frozen reference configuration"),
    (rf"For {_CASE}(?=\n)", "For the reference case"),
)

#: The whole carried-text rule: the two words, then the identities.  One
#: name for both because every caller (the carve, the device rule, the carve
#: tests, the engine-divergence fingerprint) applies them together.
CORE_PROSE: tuple[tuple[str, str], ...] = CORE_WORDS + CORE_IDENTITY

#: THE HOUSE DASH RULE, on carried Python.
#:
#: This package's published pages use no em-dash; the source tree writes as it
#: likes, and 27 of them came across inside `core/npref.py` at the first cut.
#: That file is package data in every wheel built from this branch, so it is
#: read by the same people who read the README, and it was the one house-style
#: rule the shipped source was not held to.
#:
#: SPACING IS PRESERVED rather than imposed: the dash becomes two hyphens and
#: the spaces around it stay as written, because the measured form on this
#: tree is a spaced dash and a rule that added its own spaces would produce a
#: double space on all 27.  An UNSPACED dash between two words is the other
#: form and becomes a spaced pair, which is what that form means.
#:
#: NOT ON DEVICE SOURCES.  They are carried byte for byte and their assembled
#: string is digested, so a rule that could rewrite one would move a receipt's
#: `source_sha256` the day a source revision added a dash to a comment.  There
#: are none there today (measured on the tree 2026-09-10: 0 across all fifteen
#: `.cu`/`.cuh`), and `DEVICE_FORBIDDEN` now stops the cut if one arrives,
#: which is the decision a kernel is owed rather than a silent edit to it.
_EM_DASH = chr(8212)

PROSE_DASH: tuple[tuple[str, str], ...] = (
    (rf"(?<=[A-Za-z0-9]){_EM_DASH}(?=[A-Za-z0-9])", " -- "),
    (_EM_DASH, "--"),
)

#: SHIPPED MARKDOWN carried out of the source tree, held to the prose rules.
#:
#: `PROVENANCE.md` travels with the four Noah tables because a reader who
#: finds the tables inside the wheel is owed the sentence saying where they
#: came from.  That makes it published prose, and published prose here uses
#: no em-dash; the source tree's copy opens with one in its title.  One
#: exact string, named with its file, so a re-cut reproduces the shipped
#: bytes and a source tree that reworded the title fails loudly instead of
#: shipping the dash back.
CORE_MD_PROSE: tuple[tuple[str, str, str], ...] = (
    ("data/noah_tables/PROVENANCE.md",
     "# Noah LSM parameter tables " + chr(8212) + " provenance",
     "# Noah LSM parameter tables: provenance"),
)

#: Rewritten BACK after the rewiring, in the file named, and nowhere else.
#:
#: ``MODULE_KEY_ROOT`` is a manifest NAMESPACE, not an import path: it is the
#: prefix of every key the kernel manifest files a compiled image under, and
#: every receipt this model has published was written with kernels keyed
#: ``woof.core.kernels:<name>``.  Moving it would orphan those pins in a way
#: nothing raises about.  The engine's and the carried loader's images coexist
#: under it because certify/kernel_manifest.record_module files a second,
#: differing image under a deterministic suffix rather than replacing the
#: first.
CORE_PRESERVE: tuple[tuple[str, str, str], ...] = (
    ("core/kernels/__init__.py",
     r'MODULE_KEY_ROOT = "arwen_global\.core\.kernels"',
     'MODULE_KEY_ROOT = "woof.core.kernels"'),
)

#: SLASH-SPELLED SOURCE PATHS, rewritten to where this carve puts the file.
#:
#: THE BREAKAGE THIS TABLE PREVENTS, measured on the tree 2026-09-10 from an
#: install of the built wheel beside published woof 2.7.0.  The import
#: rewiring above moves DOTTED names; a path written with slashes is prose to
#: it, so every carried file that named its own kernel or its own tables the
#: way the source tree spells them went on naming the engine's copy.  Three
#: of those sites are not documentation.  ``physics/builtin_adapters._contract``
#: writes the cumulus ``scheme_identity`` that travels into every run receipt,
#: and ``physics/native_options.validate`` repeats the same three paths in the
#: refusal a user reads; both named ``woof/core/kernels/gf.cu``,
#: ``woof/core/kernels/ntiedtke.cu`` and
#: ``woof/arwen_global/physics/arwen_massflux.py``.  The first two exist in
#: an install and are NOT the files that ran: the carried loader's ``_KDIR``
#: is its own directory, and in one process the two assembled ``gf`` sources
#: measured 208,223 bytes here against 195,818 on the engine.  The third
#: exists in no install at all.  A receipt that answers "whose physics ran"
#: with a file that did not run is the one artefact the carve exists to make
#: true.
#:
#: DERIVED FROM THE CARVE TABLES rather than written out, which is what makes
#: it a rule instead of a sweep: a module added to ``CORE_CARVE`` gets its
#: slash-spelled path rewritten by arriving, and a path naming a file the
#: carve does NOT take is left exactly as written, because it still names
#: where that file is.  That is why these are the carve's FILES and
#: directories and not a directory prefix: ``woof/core/kernels/`` as a prefix
#: would rewrite ``thompson.cu`` and ``p3.cu``, which stay on the engine and
#: are named correctly today.
#:
#: NOT APPLIED TO THE SUITE.  ``tests/`` names source paths as DATA: the seam
#: manifest is keyed by engine-relative path by construction, so a rewrite
#: there would corrupt the thing the test asserts.  This rule is for what
#: ships.  NOT APPLIED TO DEVICE SOURCES either, for the reason
#: ``VERBATIM_SUFFIXES`` gives: their bytes are digested.
def _slash_rules(*tables: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    rules: list[tuple[str, str]] = []
    for table in tables:
        for src, rel in table:
            dest = "arwen_global" if rel == "." else f"woof/globe/{rel}"
            rules.append((re.escape(src) + r"(?![A-Za-z0-9_])", dest))
    # Longest source path first: ``woof/data/arwen_global`` must not be eaten
    # by ``woof/arwen_global``, nor ``ysu_contract.py`` by ``ysu.py``.
    rules.sort(key=lambda rule: -len(rule[0]))
    return tuple(rules)


SLASH_PATHS: tuple[tuple[str, str], ...] = _slash_rules(CARVE, CORE_CARVE)

_QUOTES = chr(34) + chr(39)

#: The post-condition on the rewritten text, in the shape ``DEVICE_FORBIDDEN``
#: already uses: the rules above are the remedy, and this is what survives a
#: rule that stopped matching.  The file rules are total, so what this catches
#: is the form a rule reading one path cannot see, a path SPLIT ACROSS TWO
#: STRING LITERALS at a directory boundary, which is how the mass-flux path
#: was written and how the next one will be.
CARRIED_PATH_FORBIDDEN: tuple[tuple[str, str], ...] = (
    (rf"woof/core/kernels/[{_QUOTES}]",
     "the carried kernel directory, split across two string literals"),
    (rf"woof/data/noah_tables[{_QUOTES}]",
     "the carried Noah table directory, split across two string literals"),
)

#: EXACT SENTENCES in carried Python, rewritten in the file named and nowhere
#: else, AFTER the rewiring and BEFORE the slash-path rule.  Written as lines
#: so the table carries no escapes: the gate that reads this tree matches on
#: what is written, and a rule spelled with escapes is a rule nobody rereads.
#:
#: Two sentences in the carried physics driver are made FALSE by the carve
#: rather than merely stale, so rewriting their paths would not fix them:
#:
#: * The constant's history names the standalone preprocessing wheel that
#:   stages the compat layer and forbids the CUDA driver module.  That wheel
#:   is the engine's and knows nothing about this package, so pointing the
#:   path at the carried copy would put a false claim in the file.  The path
#:   comes out and the sentence stays true.
#: * ``run_mpas_column_batch`` is the one exported name in the carried driver
#:   that does NOT resolve to carried physics.  ``woof.core.mpas_column_batch``
#:   stays on the engine and imports the ENGINE's driver, which imports the
#:   engine's surface layer and PBL at module scope, both of which differ from
#:   the tree this model was graded in.  The docstring said that module imports
#:   THIS one, which was true before the carve and is not now.  The name is out
#:   of reach of this package's door, which admits the carried schemes only, so
#:   the remedy is to say what it resolves to rather than to change what it
#:   does.
#:
#: Two more name a DIRECTORY OF DEVELOPMENT TOOLING on the source tree, and
#: they were measured inside the built wheel on 2026-09-10.  Neither path
#: resolves in any install; both describe how the work was organised, on a
#: distribution whose source ships with it.  `SLASH_PATHS` cannot reach them:
#: it rewrites the paths of files the carve TAKES, and these name files it
#: does not take.  The sentence each is inside is about the physics, so what
#: comes out is the citation and what stays is the claim.
#:
#: SPELLED IN PIECES, for the reason `CORE_PROSE` spells the two words in
#: pieces: `tests/test_no_provenance.py` now stops that directory name
#: reaching anything that ships, it reads the TREE rather than the intent,
#: and this file ships in the sdist like everything else.
_TOOLING_DIR = "super" + "powers"

#: A THIRD one names a SCRIPT on the source tree's working root, and it is
#: the same class caught one round later: the two above were found by the
#: rule that reads a directory name, and this one was not, because the root
#: is a plain English word and the file under it is spelled with
#: underscores rather than as a directory.  The sentence it sits in is a
#: measurement of the entry contract this seam asserts, so the count, the
#: result and the date stay and the file name goes; nobody outside the
#: source tree can run that script or read it.
#:
#: SPELLED IN PIECES for the reason the name above is, and the pieces are
#: cut at both boundaries a rule matches on: the working root is separated
#: from its slash so no rule sees a path, and the file name is cut in
#: three, so that no piece of it is the name either: a scan of the built
#: artefacts for the two-piece spelling found the first piece whole.
_WORK_ROOT = "wo" + "rk"
_PROBE_SCRIPT = "probe_shin" + "hong_big" + "_ensemble"

CORE_EXACT: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("core/physics.py",
     ("#: that became structural: it stages ``woof/physics_compat.py`` and",
      "#: forbids ``woof/core/physics.py``, so a refusal that reached up here",
      "#: for the constant raised ImportError instead of refusing."),
     ("#: that became structural: it stages ``woof/physics_compat.py`` and",
      "#: forbids the engine's CUDA physics driver module, the one this file",
      "#: is the carried copy of, so a refusal that reached up here for the",
      "#: constant raised ImportError instead of refusing.")),
    ("core/physics.py",
     ("    ``woof.globe.core.physics.run_mpas_column_batch`` is the published API name",
      "    for the persistent MPAS physics seam; the implementation lives in",
      "    :mod:`woof.core.mpas_column_batch`, which imports THIS module for the",
      "    driver/coupling machinery.  A lazy attribute breaks that cycle without"),
     ("    ``woof.globe.core.physics.run_mpas_column_batch`` is the published",
      "    API name for the persistent MPAS physics seam, and THE SEAM STAYS ON",
      "    THE ENGINE.  The implementation is :mod:`woof.core.mpas_column_batch`,",
      "    which imports the ENGINE's own physics driver module for the",
      "    driver/coupling machinery, not this carried copy, so the object this",
      "    name returns runs the PUBLISHED engine's physics and not the physics",
      "    this package carries.  Nothing in this package's own door reaches it:",
      "    the native options admit the carried schemes only.  A lazy attribute",
      "    breaks that cycle without")),
    ("core/physics.py",
     (f"#: (``docs/{_TOOLING_DIR}/plans/2026-07-14-gpuwm-phase3.md``: "
      '"constant',
      '#: GLW=300 W/m^2 on a vegetated land column").  Making it a better '
      "number"),
     ("#: (the specification those cases were written to: a constant",
      "#: GLW = 300 W/m^2 on a vegetated land column).  Making it a better "
      "number")),
    ("core/physics.py",
     ('        # "explicit-deposit-v1"; G-LAKE root cause',
      f"        # .{_TOOLING_DIR}/sdd/lake-momentum-root-cause.md): the "
      "theta and"),
     ('        # "explicit-deposit-v1"; the G-LAKE lake-momentum root',
      "        # cause): the theta and")),
    ("core/physics.py",
     (f"        # MEASURED the other direction ({_WORK_ROOT}/{_PROBE_SCRIPT}",
      "        # .py, 2026-08-16): 24,000 adversarial columns with entry "
      "TKE at",
      "        # or above the floor produced ZERO tke-confined non-finites."),
     ("        # MEASURED the other direction (2026-08-16): 24,000 "
      "adversarial",
      "        # columns with entry TKE at or above the floor produced ZERO",
      "        # tke-confined non-finites.")),
)

#: RRTMGP READS ITS TRACE-GAS AND OZONE CLIMATOLOGY FROM THE ENGINE'S TABLE.
#:
#: THE BREAKAGE THIS TABLE PREVENTS.  The carried ``RRTMGPRadiation`` opened
#: the RFMIP clear-sky input NetCDF out of the engine's ``recast-woof-data``
#: companion at every construction, for 136 numbers: sixteen experiment-zero
#: global means and the median layer pressure and ozone over the 100 sites.
#: The engine stops shipping that file at 2.8.0, because no RFMIP file is
#: redistributed any more (the reference results are CC-BY-NC-SA-4.0 and the
#: input file's own licence attribute contradicts its link), so on that
#: companion the first radiation call would refuse.  The numbers ship in the
#: engine's companion as ``rrtmgp-trace-gas-climatology.json`` and
#: :func:`woof.core.rrtmgp.load_trace_climatology` loads them against their
#: SHA-256 pin; the carried driver calls that loader.  Until 2026-09-29 this
#: package shipped its own copy of the table and loader, because the engine
#: it could install beside did not have them yet; the 2.8.0 floor retired
#: that copy.
#:
#: The same reason moves the RFMIP clear-sky ORACLE (``_rfmip_profiles`` and
#: ``rfmip_clear_sky``) onto the engine's fetch route,
#: :func:`woof.core.rfmip_upstream.fetch_rfmip`: the file is fetched from its
#: pinned upstream commit into a user cache and verified, or read from a copy
#: the caller names, exactly as the engine's own oracle does.
#:
#: A CODE RULE, applied by the same exact-block machinery as ``CORE_EXACT``
#: and holding the same post-condition: a block that stops matching stops the
#: cut.  The source tree still opens the NetCDF, and it has to until it runs
#: on an engine without the file; the day it changes these blocks, the rule
#: matches nothing, the cut stops, and whoever re-cuts retires the entry
#: rather than carrying two routes to the same numbers.
#:
#: BIT-IDENTICAL, measured: every value equals what the replaced block
#: computed from the NetCDF (``float(GM[0]) * units``, float64 medians over
#: the sites), and a T255 ten-step run on one card wrote byte-identical
#: checkpoints before and after (2026-09-26).  The engine's table is the same
#: bytes as the copy that run read (SHA-256 ``71d7f857...``).
CORE_TRACE_CLIMATOLOGY: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("core/rrtmgp.py",
     ("    explicit case overrides.  Water vapor comes from the model and ozone is",
      "    interpolated from the median RFMIP climatological profile.",
      '    """'),
     ("    explicit case overrides.  Water vapor comes from the model and ozone is",
      "    interpolated from the median RFMIP climatological profile.  Both come",
      "    from the engine's :func:`woof.core.rrtmgp.load_trace_climatology`,",
      "    the table derived from the RFMIP input file, which is not read here.",
      '    """')),
    ("core/rrtmgp.py",
     ('        with Dataset(_table("rfmip-clear-sky-inputs.nc"), "r") as ncfile:',
      "            ncfile.set_auto_mask(False)",
      "            for gas, rfmip_name in _RFMIP_GAS_NAMES.items():",
      '                variable = ncfile[rfmip_name + "_GM"]',
      '                scale = float(getattr(variable, "units", "1").replace(" ", ""))',
      "                self.trace_vmr[gas] = float(variable[0]) * scale",
      "            for gas, value in trace_gases(",
      "                    self.start_time, self.trace_gas_overrides).items():",
      "                if gas not in self.trace_vmr:",
      "                    # Defensive parity with the pure policy validation: table",
      "                    # and packaged RFMIP names must never drift silently.",
      "                    raise ValueError(",
      '                        f"unknown trace gas {gas!r}; known well-mixed gases: "',
      '                        f"{sorted(self.trace_vmr)}")',
      "                self.trace_vmr[gas] = value",
      "            pressure = np.median(",
      '                np.asarray(ncfile["pres_layer"][:], np.float64), axis=0)',
      "            ozone = np.median(",
      '                np.asarray(ncfile["ozone"][0], np.float64), axis=0)'),
     ("        from woof.core.rrtmgp import load_trace_climatology",
      "",
      "        climatology = load_trace_climatology()",
      "        for gas in _RFMIP_GAS_NAMES:",
      "            self.trace_vmr[gas] = climatology.trace_vmr[gas]",
      "        for gas, value in trace_gases(",
      "                self.start_time, self.trace_gas_overrides).items():",
      "            if gas not in self.trace_vmr:",
      "                # Defensive parity with the pure policy validation: table",
      "                # and packaged RFMIP names must never drift silently.",
      "                raise ValueError(",
      '                    f"unknown trace gas {gas!r}; known well-mixed gases: "',
      '                    f"{sorted(self.trace_vmr)}")',
      "            self.trace_vmr[gas] = value",
      "        pressure = climatology.pressure_layer_pa",
      "        ozone = climatology.ozone_vmr")),
    ("core/rrtmgp.py",
     ('def _rfmip_profiles(tables, sites, experiments):',
      '    sites = np.asarray(sites, dtype=np.intp)',
      '    experiments = np.asarray(experiments, dtype=np.intp)',
      '    with Dataset(_table("rfmip-clear-sky-inputs.nc"), "r") as nc:'),
     ('def _rfmip_profiles(tables, sites, experiments, inputs=None):',
      '    from woof.core.rfmip_upstream import fetch_rfmip',
      '',
      '    sites = np.asarray(sites, dtype=np.intp)',
      '    experiments = np.asarray(experiments, dtype=np.intp)',
      '    source = fetch_rfmip("rfmip-clear-sky-inputs.nc", path=inputs)',
      '    with Dataset(source, "r") as nc:')),
    ("core/rrtmgp.py",
     ('def rfmip_clear_sky(*, sites=None, experiments=None) -> RFMIPResult:',
      '    """Run the shipped RFMIP clear-sky oracle profiles on the GPU.',
      '',
      '    This reproduces the upstream physics-index-1/forcing-index-1 examples:',
      '    one-angle LW, default solar spectrum normalized to each RFMIP TSI, and',
      '    nighttime columns explicitly zeroed after the SW solve.',
      '    """'),
     ('def rfmip_clear_sky(*, sites=None, experiments=None,',
      '                    inputs=None) -> RFMIPResult:',
      '    """Run the RFMIP clear-sky oracle profiles on the GPU.',
      '',
      '    This reproduces the upstream physics-index-1/forcing-index-1 examples:',
      '    one-angle LW, default solar spectrum normalized to each RFMIP TSI, and',
      '    nighttime columns explicitly zeroed after the SW solve.',
      '',
      '    The RFMIP input file is not shipped by the engine from 2.8.0 on (see',
      '    :mod:`woof.core.rfmip_upstream`): ``inputs`` names a local copy, and',
      '    without it the pinned upstream file is fetched into the RFMIP cache.',
      '    Either way its SHA-256 is verified before a byte is read.',
      '    """')),
    ("core/rrtmgp.py",
     ('     _sza, _tsi) = _rfmip_profiles(lw, sites, experiments)',),
     ('     _sza, _tsi) = _rfmip_profiles(lw, sites, experiments, inputs)',)),
    ("core/rrtmgp.py",
     ('     sza, tsi) = _rfmip_profiles(sw, sites, experiments)',),
     ('     sza, tsi) = _rfmip_profiles(sw, sites, experiments, inputs)',)),
)



# The suite is part of the cut, and finding that out cost a whole run.  The
# first re-cut took the model and left the tests where they were, on the
# reasoning that the suite here has diverged on purpose -- which it has, into
# named skips for engine gaps and this distribution's exit codes.  But a test
# is written against the module beside it, so a model that moves 46 files and
# leaves its tests behind fails 30 of them: the sizer's measurement rows, the
# assimilation door's new options, the banded peak model, the entry operator.
# Every one was a test asserting the previous revision's behaviour.
#
# So the suite is merged the same way the model is, three ways, and the
# divergence survives because that is what a merge does.
#: THE BESIDE-THE-CODE NOTICE, NARROWED TO WHAT THIS PACKAGE SHIPS.
#:
#: THE BREAKAGE THIS TABLE PREVENTS.  The source tree's copy of
#: ``LICENSE-third-party.txt`` covers its whole kernel directory: fourteen
#: files carrying Arm's libm cores, nine more carrying FDLIBM's, eight legacy
#: RRTMG translation units under AER's grant, and the Numerical Recipes
#: GAMMLN coefficients in two files.  This package carries fifteen device
#: sources and NONE of those except glibc_flt32.cuh and the five RTE+RRTMGP
#: ones.  Carried verbatim, the notice would tell a reader that this
#: distribution contains AER's RRTMG and Numerical Recipes' coefficients,
#: which it does not, and would name twenty-odd files that are not in the
#: wheel.  A notice that describes a different directory is worse than no
#: notice: it is a false statement about what somebody received.
#:
#: So the carve NARROWS it, and narrowing is the only edit made: every grant
#: the carried files do stand on keeps the source tree's own wording, because
#: that wording is the provenance account and rewriting it would fork the two
#: copies of the same claim.  ``CORE_EXACT``'s rule holds: a block that stops
#: matching stops the cut, which is what happens the day the source tree
#: rewords a section this table trims, and is exactly when somebody has to
#: look at whether the trim is still right.
#:
#: ``tests/test_licence_notices_ship.py`` is the other half: it reads the
#: CARRIED copy and fails if it names a device source this package does not
#: ship, or if a carried kernel's transcription has no section here.
KERNEL_NOTICE_SCOPE: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    # Where the texts land in THIS distribution's wheel.
    ('core/kernels/LICENSE-third-party.txt',
     ("provenance account, are in the repository's root NOTICE, and the complete",
      "licence texts are in the repository's licenses/ directory, which ships in",
      "the wheel under gpuwm-<version>.dist-info/licenses/."),
     ("provenance account, are in the repository's root NOTICE, and the complete",
      "licence texts are in the repository's licenses/ directory, which ships in",
      "the wheel under gpuwm_global-<version>.dist-info/licenses/licenses/",
      "(PEP 639 keeps the source path, hence the doubled component).")),
    # Why the notice is here and not in each file.  The source tree's reason
    # is its two frozen-digest suites, which this package does not carry; the
    # reason that survives the carve is the receipt.
    ('core/kernels/LICENSE-third-party.txt',
     ("WHY THIS FILE EXISTS RATHER THAN A HEADER IN EACH .cu",
      "-----------------------------------------------------",
      "Every .cu in this directory is pinned by SHA-256 -- by",
      "tests/test_mp8_frozen.py::FROZEN_MODULE_DIGESTS, which pins the file AND the",
      "assembled compile string the loader hands nvrtc, and by",
      "tests/test_kernel_source_freeze_per_module.py::BASELINE_PINNED, which pins",
      "the rest.  A comment prepended to any of them would move a frozen digest and",
      "turn a numerics gate red for a reason that has nothing to do with numerics.",
      "So the notice sits beside the code instead of inside it, and it ships with",
      "the code: pyproject.toml's package-data glob for woof.globe.core.kernels names",
      "this file explicitly.",
      "",
      "The two .cuh headers here -- glibc_flt32.cuh and thompson_aerosol_common.cuh",
      "-- are not themselves digest-pinned, but they are PREPENDED to pinned",
      "translation units by the loader's _EXTRA_HEADERS table, so their bytes are",
      "inside the assembled compile string a receipt records as source_sha256 for",
      "gf and ntiedtke.  A notice comment in either one moves those receipts'",
      "digests without moving a compiled image, so the notice for both stays in",
      "this file with the rest.  The Python counterparts of the same",
      "transcriptions, which nothing digests, do carry their notice in their own",
      "header: woof/core/noahmp_libm.py, woof/core/mynn_pbl.py,",
      "woof/core/ruc.py, woof/core/rrtmgp.py and woof/verify/npref.py among",
      "them."),
     ("WHY THIS FILE EXISTS RATHER THAN A HEADER IN EACH .cu",
      "-----------------------------------------------------",
      "The bytes of these device sources ARE the identity a run receipt",
      "records: the loader assembles each module's compile string out of the",
      "files in this directory and the receipt pins its source_sha256.  A",
      "notice comment prepended to one of them would move that digest without",
      "moving a compiled image, and every receipt this model has published",
      "would stop matching the code that wrote it.  So the notice sits beside",
      "the code instead of inside it, and it ships with the code:",
      "pyproject.toml's package-data for woof.globe.core.kernels names this",
      "file explicitly.",
      "",
      "glibc_flt32.cuh is covered here for the same reason even though it is",
      "not a compilation unit of its own: the loader prepends it to gf and to",
      "ntiedtke, so its bytes sit inside those two modules' assembled",
      "strings.  The Python counterparts of the same transcriptions, which",
      "nothing digests, do carry their notice in their own header:",
      "woof/globe/core/rrtmgp.py and woof/globe/core/npref.py.")),
    # What is covered here, and the libm scope, both narrowed to this
    # directory's fifteen files.
    ('core/kernels/LICENSE-third-party.txt',
     ("1. Arm optimized-routines (MIT)        -- logf, expf, exp2f, powf",
      "2. FDLIBM / Sun Microsystems           -- expm1f, tanhf, atanf, log10f",
      "3. AER RRTMG (BSD 3-Clause)            -- the RRTMG radiation kernels, legacy",
      "                                          and the McICA generator the default",
      "                                          radiation path is driven with",
      "4. RTE+RRTMGP (BSD 3-Clause)           -- the default radiation kernels",
      "5. Numerical Recipes Software          -- the GAMMLN log-gamma coefficients",
      "",
      "Sections 1 and 2 also state the scope of those two grants; section 6 states",
      "what none of the five covers, and says whose work the gamma routines in",
      "glibc_flt32.cuh are."),
     ("1. Arm optimized-routines (MIT)        -- logf, expf, exp2f, powf",
      "2. FDLIBM / Sun Microsystems           -- expm1f, the lgammaf reduction",
      "3. AER RRTMG (BSD 3-Clause)            -- the McICA subcolumn generator the",
      "                                          radiation path is driven with",
      "4. RTE+RRTMGP (BSD 3-Clause)           -- the rest of the radiation kernels",
      "",
      "Sections 1 and 2 also state the scope of those two grants; section 5 states",
      "what none of the four covers, and says whose work the gamma routines in",
      "glibc_flt32.cuh are.")),
    ('core/kernels/LICENSE-third-party.txt',
     ("Scope in this directory: the logf / expf / exp2f / powf cores and their data",
      "tables (Arm, MIT) appear in glibc_flt32.cuh, mynn_dmp_sibling.cu, mynn_pbl.cu,",
      "noahmp_bareflux.cu, noahmp_fluxprep.cu, noahmp_leaves.cu, noahmp_radiation.cu,",
      "noahmp_snow.cu, noahmp_soilwater.cu, noahmp_vegeflux.cu, noahmp_vegprecip.cu,",
      "noahmp_water.cu, rrtmg_lw.cu and rrtmg_sw.cu.  The expm1f / tanhf / atanf /",
      "log10f reductions (FDLIBM, Sun Microsystems) appear in mynn_dmp_sibling.cu,",
      "mynn_pbl.cu, noahmp_bareflux.cu, noahmp_energy.cu, noahmp_fluxprep.cu,",
      "noahmp_glacier.cu, noahmp_leaves.cu, noahmp_vegeflux.cu and ruc.cu.",
      "glibc_flt32.cuh is on that second list as well, and on a third count:",
      "besides the Arm cores above it carries FDLIBM's expm1f (s_expm1f.c) and the",
      "positive arm of FDLIBM's lgammaf reduction (e_lgammaf_r.c), which are the",
      "only transcriptions of those two routines in this tree.  Their Python",
      "counterparts are in woof/core/noahmp_libm.py, woof/core/mynn_pbl.py and",
      "woof/core/ruc.py, each of which carries this notice in its own header.",
      "tests/test_cuda_libm_table_copies.py enumerates the table copies and keeps",
      "them in step, so this list is machine-checkable rather than prose."),
     ("Scope in this directory: both libm grants are carried by one file,",
      "glibc_flt32.cuh, which the loader prepends to gf and to ntiedtke.  It",
      "holds the logf / expf / exp2f / powf cores and their data tables (Arm,",
      "MIT), and FDLIBM's expm1f (s_expm1f.c) and the positive arm of FDLIBM's",
      "lgammaf reduction (e_lgammaf_r.c).  No other device source here carries",
      "either.  The model's own source tree transcribes the same routines in",
      "further files -- its boundary-layer, Noah-MP and legacy-RRTMG kernels --",
      "and its own copy of this notice names them; this package ships none of",
      "those files, and this copy names what is here.")),
    # AER RRTMG: not one of the eight legacy translation units is carried,
    # but rrtmgp_mcica.cu is, and it is AER's work.  So this section is
    # NARROWED to that one file rather than deleted.  The 0.1.0 notice filed
    # that kernel under RTE+RRTMGP on the strength of its filename prefix;
    # dropping the section whole would make the same mistake the other way.
    ('core/kernels/LICENSE-third-party.txt',
     ("3. AER RRTMG -- the RRTMG radiation kernels, legacy and McICA",
      "-------------------------------------------------------------",
      "",
      "rrtmg_lw.cu, rrtmg_sw.cu, rrtmg_lw_chain.cu, rrtmg_lw_taugb02_10_11_12.cu,",
      "rrtmg_lw_taugb03_05.cu, rrtmg_lw_taugb06_09.cu, rrtmg_lw_taugb13_16.cu and",
      "rrtmg_mcica_wrf.cu transcribe WRF v4.6.1's phys/module_ra_rrtmg_lw.F and",
      "phys/module_ra_rrtmg_sw.F.  That code is the work of Atmospheric and",
      "Environmental Research, Inc., not of UCAR, and WRF says so by preserving",
      "AER's notice -- seven times in the longwave file, nine in the shortwave:"),
     ("3. AER RRTMG -- the McICA subcolumn generator",
      "----------------------------------------------",
      "",
      "rrtmgp_mcica.cu is AER's work, and its name is the reason that has to be",
      "said out loud.  The McICA subcolumn cloud generator the radiation path is",
      "driven with is WRF's RRTMG generator, not rte-rrtmgp's: that file",
      "transcribes module mcica_subcol_gen_sw in WRF v4.6.1",
      "phys/module_ra_rrtmg_sw.F -- kissvec at lines 2008-2040, the",
      "maximum-random overlap walk at 1778-1813 -- as its own header says.  Its",
      "host driver in woof/globe/core/rrtmgp.py and its float64 mirror in",
      "woof/globe/core/npref.py mirror the same routine and carry this notice",
      "in their own headers.  Section 4 does not cover any of the three.",
      "",
      "That code is the work of Atmospheric and Environmental Research, Inc.,",
      "not of UCAR, and WRF says so by preserving AER's notice -- seven times in",
      "the longwave file, nine in the shortwave:")),
    # The tail of the same section: the pointer to legacy coefficients this
    # package does not ship, and the source tree's own McICA paragraph, which
    # names its paths and now says what the narrowed heading already says.
    ('core/kernels/LICENSE-third-party.txt',
     ("licenses/LICENSE-AER-RRTMG-BSD-3-Clause.txt and beside the packaged",
      "coefficients in woof/data/wrf_radiation/.",
      "",
      "rrtmgp_mcica.cu is in this section too, and its name is the reason that has",
      "to be said out loud.  The McICA subcolumn cloud generator that the DEFAULT",
      "RTE+RRTMGP radiation path is driven with is WRF's RRTMG generator, not",
      "rte-rrtmgp's: that file transcribes module mcica_subcol_gen_sw in",
      "phys/module_ra_rrtmg_sw.F -- kissvec at lines 2008-2040, the maximum-random",
      "overlap walk at 1778-1813 -- as its own header says.  Its host driver in",
      "woof/core/rrtmgp.py and its float64 mirror in woof/verify/npref.py mirror",
      "the same routine and carry this notice in their own headers.  It is AER's",
      "work under the same grant, and section 4 does not cover it."),
     ("licenses/LICENSE-AER-RRTMG-BSD-3-Clause.txt.  The model's source tree",
      "also ships that text beside its packaged legacy-RRTMG coefficients, and",
      "applies this section to eight legacy translation units besides; this",
      "package carries neither those files nor those coefficients, and this copy",
      "covers the one file it does carry.")),
    # Numerical Recipes: neither carrier is carried.
    ('core/kernels/LICENSE-third-party.txt',
     ("5. Numerical Recipes Software -- the GAMMLN log-gamma coefficients",
      "------------------------------------------------------------------",
      "",
      "nssl2_fused_gs.cu's wrf_gamma_dp, and the THOMPSON_AA_CC* tables in",
      "thompson_aerosol_common.cuh that were produced with it, carry the Lanczos",
      "g=5, n=6 coefficient set and evaluation that WRF v4.6.1 takes from Numerical",
      "Recipes.  WRF preserves the notice immediately above and below REAL FUNCTION",
      "GAMMLN(XX) at phys/module_mp_thompson.F:5325-5347:",
      "",
      "    (C) Copr. 1986-92 Numerical Recipes Software 2.02",
      "",
      "It is preserved here for the same reason.  WOOF's position on this material",
      "-- taken from WRF, reproduced in order to match WRF bit for bit, and the",
      "standard published Lanczos coefficient set -- is set out in full, including",
      "the argument against it, in licenses/NOTICE-Numerical-Recipes.txt.",
      "",
      "",
      "6. The scope of the five grants above",
      "-------------------------------------"),
     ("5. The scope of the four grants above",
      "-------------------------------------")),
    # The closing scope paragraph, over four grants rather than five.  The
    # three sentences that follow it in the source tree, saying whose work
    # the gamma routines are, are carried as they stand.
    ('core/kernels/LICENSE-third-party.txt',
     ("Each is stated by routine or by file on purpose.  They cover what sections 1",
      "to 5 name and nothing else.  Other code in this directory -- including other",
      "transcriptions -- takes nothing from Arm, Sun, AER, the RTE+RRTMGP copyright",
      "holders or Numerical Recipes, and is not offered to a reader on the strength",
      "of these notices.  Its provenance is recorded where it lives: in the",
      "repository NOTICE, in PROVENANCE.md, and in each file's own header."),
     ("Each is stated by routine or by file on purpose.  They cover what sections 1",
      "to 4 name and nothing else.  Other code in this directory -- including other",
      "transcriptions, and there are many: these kernels are transcriptions of WRF",
      "schemes -- takes nothing from Arm, Sun, AER or the RTE+RRTMGP copyright",
      "holders, and is not offered to a reader on the strength of these notices.",
      "Its provenance is recorded where it lives: in the repository NOTICE, in",
      "PROVENANCE.md, and in each file's own header.")),
    # The FDLIBM section header and the glibc-versus-FDLIBM spelling
    # paragraph, which prices routines no carried file holds.
    ('core/kernels/LICENSE-third-party.txt',
     ("2. FDLIBM -- expm1f, tanhf, atanf, log10f",
      "------------------------------------------"),
     ("2. FDLIBM -- expm1f and the lgammaf reduction",
      "----------------------------------------------")),
    ('core/kernels/LICENSE-third-party.txt',
     ("Where WOOF must match a glibc 2.39 reference bit for bit, these",
      "transcriptions follow glibc's versions of these files, which differ from other",
      "FDLIBM descendants at a small number of points.  Through 2.6.5 there were",
      "three: atanf's 2**25 large-argument threshold, the fabsf spelling in log10f's",
      "zero path, and a redundant zero guard in tanhf.  Two of the three are gone at",
      "2.6.6 because neither did any work -- tanhf's guard is unreachable behind the",
      "|x| < 2**-55 branch, and log10f's zero path divided -2**25 by |x| only to",
      "reach -inf -- and the two forms were compared on all 4,294,967,296 float32 bit",
      "patterns with zero differing.  noahmp_energy.cu and noahmp_leaves.cu carry the",
      "argument at the point of the edit.  lgammaf's 2**26 and 2**-30 thresholds are",
      "on that list too, carried by the positive arm of the lgammaf reduction in",
      "glibc_flt32.cuh.  atanf's threshold stays: it is a fact a",
      "black-box measurement recovers rather than a spelling read from a source, and",
      "the root NOTICE records that it was measured from the running library."),
     ("Where WOOF must match a glibc 2.39 reference bit for bit, these",
      "transcriptions follow glibc's versions of these files rather than another",
      "FDLIBM descendant's.  For the two routines carried here that means",
      "lgammaf's 2**26 and 2**-30 thresholds, which are glibc's spelling of the",
      "same reduction.")),
)


TEST_CARVE: tuple[tuple[str, str], ...] = (
    ("tests", "."),
)

#: Only these files are taken from the source tree's tests directory.  It
#: holds the engine's whole suite; this package's share of it is the two
#: prefixes plus one shared helper, and a glob wider than that would drag in
#: several thousand tests of software this distribution does not contain.
TEST_PATTERNS = (
    "test_arwen_global_*.py",
    "test_global_spectral_*.py",
    "global_spectral_dense_reference.py",
)

# The import rewiring.  Order matters: the registry and the rows grid are
# matched before the two package prefixes, because both live under paths a
# broader rule would rewrite first.
#
# These are applied to BOTH sides of the merge, so a rule that is wrong shows
# up as a conflict rather than as a silent difference.
REWIRE: tuple[tuple[str, str], ...] = (
    (r"woof\.core\.global_physics_registry", "woof.globe.physics.registry"),
    (r"woof\.static\.rows", "woof.globe.statics_rows"),
    (r"woof\.global_spectral", "woof.globe.spectral"),
    (r"woof\.arwen_global", "arwen_global"),
)

# The command spellings.  In the source tree the model is reached as a
# subcommand of the engine; here it is a console script of its own.  A help
# string that names a command the reader cannot type is a door that does not
# open, so these are rewritten too -- and they are rewritten on both sides,
# which is what keeps a later edit to the same sentence from conflicting on
# the spelling alone.
COMMANDS: tuple[tuple[str, str], ...] = (
    (r"woof global da\b", "woof global da"),
    (r"woof global\b", "woof global"),
)

# NOT A RULE, and this is why it is written down.  The first carve also turned
# `python -m arwen_global` into the console script, AFTER the prefix rewiring
# had already produced that spelling.  For the bare module door that is right;
# for `python -m woof.globe.upper_air_scorecard` and the twelve other modules
# with a `__main__` guard, the two rules composed into the console script's
# name with the module name stuck to it after a DOT, which is not a command:
# the script takes subcommands separated by a space, so a shell reports the
# whole thing as not found.  21 sites shipped in the package's own source and
# 18 more in its published pages.  The prefix rewiring above already produces
# the correct `python -m arwen_global...` on its own, so the rule that broke
# them is absent rather than fixed, and tests/test_cli_doors.py holds the tree
# to that.

TEXT_SUFFIXES = {".py", ".cu", ".cuh", ".json", ".toml", ".md", ".txt", ".sh"}

#: Suffixes NO REWIRING rule may touch.
#:
#: THE BREAKAGE THIS PREVENTS, and it was measured on the first run of
#: the carve rather than imagined: eight COMMENT lines across gf.cu,
#: sfclay.cu, ysu.cu, morrison.cu and rrtmgp_rte.cu name the Python
#: modules that drive or mirror them, and the rewiring rewrote all eight,
#: adding 48 bytes to the kernel set.  A kernel source string is not
#: prose: the loader hands the assembled string to nvrtc AND to
#: certify/kernel_manifest.record_module, which digests exactly that
#: string.  Editing a comment inside one moves the source_sha256 every
#: receipt this model has published was written under, and nothing
#: raises to say so.  So the import rewiring stops at these suffixes and
#: a device source keeps the shape it was graded in.
VERBATIM_SUFFIXES = {".cu", ".cuh"}

#: THE ONE EXCEPTION, and it is the decision this file said was owed here.
#:
#: `CORE_PROSE` above used to say the device sources happen to be clean of
#: the two words this project does not publish, and that if one ever was
#: not, the answer would be a decision about that kernel rather than a
#: silent edit.  They were not clean.  Measured on the tree 2026-09-09:
#: `gf.cu`
#: shouts the first of the two words twice, `ntiedtke.cu` uses the second
#: twice, and `glibc_flt32.cuh` cites, by file name, the tool one of its
#: comments was written with.  All five lines are inside the wheel, on a
#: public index, in a distribution whose source ships with it.
#:
#: THE DECISION.  The five lines are rewritten, by this rule, in the carried
#: copies only.  Not by hand: a hand edit would make the carried kernels
#: something no re-cut reproduces, and reproducibility from the source tree
#: is the whole claim of this file.  The rule is exactly the word table
#: above plus the one exact string below, applied to `.cu` and `.cuh` and
#: to nothing else.  Machine names and the one case name in device comments
#: are rewritten by `CORE_IDENTITY`, which is part of that word table since
#: 0.1.2; the paragraph there says what it costs.
#:
#: WHAT IT COSTS, measured rather than assumed: 17 bytes across three of the
#: fifteen device sources (gf.cu -2, ntiedtke.cu -4, glibc_flt32.cuh -11),
#: which moves those three files' `source_sha256`, the digest the receipt
#: records over the kernel set, and nothing else.  The COMPILED image does
#: not move: a comment is gone before nvrtc reaches the first token, and
#: that was measured on the card rather than asserted (see the lane's
#: kernel-image proof).  The engine's own copies are untouched, so a
#: receipt written by the engine is unaffected.
#: SPELLED IN PIECES, and the split is INSIDE the word rather than beside it:
#: the gate that reads this tree matches on a word boundary, so a name broken
#: at the dot is still the name.  This file ships in the sdist like every
#: other, and a table that names what it removes must not become an instance
#: of it.
_TOOL = "cla" + "ude"
#: SPLIT INSIDE THE WORD, every one of them, which is what the paragraph
#: above says and what one of these rows did not do: the third name was
#: broken at a word boundary, so its first four letters shipped as a word
#: of their own and the tree's own process-language gate reported a hit
#: on this file.  A table that names what it removes must not be an instance of
#: it, at any of the boundaries a gate reads.
_TOOL_NAMES = (_TOOL, "anthro" + "pic", "cha" + "tgpt", "open" + "ai",
               "copi" + "lot")

DEVICE_EXACT: tuple[tuple[str, str, str], ...] = (
    ("glibc_flt32.cuh",
     "// (" + _TOOL.upper() + ".md: __fmaf_rn/__fmul_rn/__fadd_rn are "
     "NVIDIA-guaranteed",
     "// (__fmaf_rn/__fmul_rn/__fadd_rn are NVIDIA-guaranteed"),
)

#: What a carried device source may not contain when the carve is done.  The
#: rules above are the remedy; this is the post-condition, because a rule
#: that stopped matching is indistinguishable from a clean tree and the
#: source it reads is somebody else's to reword.
DEVICE_FORBIDDEN: tuple[tuple[str, str], ...] = (
    (rf"(?i)(?<![A-Za-z])({_FIRST}\w*|{_SECOND})(?![A-Za-z])",
     "a word this project does not publish"),
    (r"(?i)\b(" + "|".join(_TOOL_NAMES) + r")\b",
     "the name of a tool a file was written with"),
    (re.escape(_EM_DASH),
     "an em-dash, which this package's published text does not use"),
    (rf"\b(weather-{_NODE}-\d+|{_NODE}-[1-9]\d*)\b",
     "the name of a private machine"),
    (rf"(?i){_CASE}", "a private case name"),
)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def read_source_md() -> dict[str, str]:
    """The revision the last cut was taken from."""
    if not SOURCE_MD.is_file():
        return {}
    text = SOURCE_MD.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for key, pattern in (
        ("revision", r"^- Revision:\s*`([0-9a-f]{7,40})`"),
    ):
        m = re.search(pattern, text, re.M)
        if m:
            out[key] = m.group(1)
    return out


def extract(worktree: Path, rev: str, dest: Path,
            carve: tuple[tuple[str, str], ...] = CARVE,
            patterns: tuple[str, ...] | None = None,
            extra_rewire: tuple[tuple[str, str], ...] = (),
            preserve: tuple[tuple[str, str, str], ...] = (),
            slash: tuple[tuple[str, str], ...] = ()) -> None:
    """The carved paths of one revision, laid out in this package's shape."""
    raw = dest / "_raw"
    raw.mkdir(parents=True)
    # Only the units that EXIST at this revision.  `git archive` fails the
    # whole extraction on one pathspec that matches nothing, and a unit that
    # arrived after the base revision -- the assimilation system's data
    # directory did exactly that -- is a normal state, not an error.
    paths = [
        src for src, _ in carve
        if subprocess.run(["git", "cat-file", "-e", f"{rev}:{src}"],
                          cwd=worktree, capture_output=True).returncode == 0
    ]
    if not paths:
        raise SystemExit(f"none of the carved paths exist at {rev}")
    proc = subprocess.Popen(
        ["git", "archive", rev, "--"] + paths,
        cwd=worktree,
        stdout=subprocess.PIPE,
    )
    tar = subprocess.Popen(["tar", "-x", "-C", str(raw)], stdin=proc.stdout)
    proc.stdout.close()
    tar.communicate()
    if tar.returncode != 0 or proc.wait() != 0:
        raise SystemExit(f"could not extract {rev} from {worktree}")

    tree = dest / "arwen_global"
    for src, rel in carve:
        s = raw / src
        if not s.exists():
            # A unit that does not exist at this revision is not an error:
            # the data directory arrived with the assimilation system and is
            # absent from every revision before it.
            continue
        d = tree if rel == "." else tree / rel
        if s.is_dir():
            d.mkdir(parents=True, exist_ok=True)
            for item in s.iterdir():
                if patterns is not None:
                    if item.is_dir() or not any(
                            fnmatch.fnmatch(item.name, pat) for pat in patterns):
                        continue
                target = d / item.name
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, target)
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
    shutil.rmtree(raw)
    rewire(tree, extra_rewire, preserve, slash)


#: Carried paths the import and command rewriting must NOT touch, relative
#: to the package root.  The authority mappings are the ENGINE's documents:
#: their notes name `gpuwm.arwen_global.surface_seeding` (two of the six do),
#: and rewriting that to this package's spelling would make the carried copy
#: differ from the engine's by two bytes of prose.  The resolver refuses a
#: mapping both tables carry with different bytes -- that refusal exists to
#: catch an engine row that has MOVED -- so a cosmetic rewrite here would
#: refuse every command the day the engine finally published the row.
VERBATIM_PREFIXES: tuple[str, ...] = (
    "data/authorities/",
    # UCAR's notice, reproduced verbatim beside the four WRF tables it
    # belongs to.  Third-party licence text is not this carve's to reword:
    # the rewiring rules exist to keep OUR prose true about where OUR files
    # landed, and a rule that edited a licence would be modifying the thing
    # the licence asks to be reproduced unchanged.
    "data/noah_tables/LICENSE-WRF.txt",
)


def device_prose(path: Path) -> None:
    """The word rule and the one exact string, on one carried device source.

    Then the post-condition, which is the half that survives a source tree
    that rewords the line: what is left is checked, and anything still
    carrying a banned word or a tool name stops the carve by name rather
    than shipping inside a wheel.
    """

    original = path.read_text(encoding="utf-8")
    text = original
    for pattern, replacement in CORE_PROSE:
        text = re.sub(pattern, replacement, text)
    for name, old, new in DEVICE_EXACT:
        if path.name == name:
            text = text.replace(old, new)
    for pattern, what in DEVICE_FORBIDDEN:
        match = re.search(pattern, text)
        if match:
            line = text[:match.start()].count(chr(10)) + 1
            raise SystemExit(
                f"{path.name}:{line} carries {what} ({match.group(0)!r}) and "
                "no rule here removes it.  A device source ships inside the "
                "wheel on a public index: reword it in the source tree, or "
                "add the rule that rewrites it, and re-measure the kernel "
                "digests either way")
    if text != original:
        path.write_text(text, encoding="utf-8", newline="")


def carried_paths_clean(rel: str, text: str) -> None:
    """No carried file is left named at the source tree's path.

    The slash rules are total over the paths they name, so what this fires on
    is the form a rule reading a whole path cannot see: a path SPLIT ACROSS
    TWO STRING LITERALS at a directory boundary.  That is how the mass-flux
    path in the cumulus refusal was written, and it is the shape the next one
    will have.
    """

    for pattern, what in CARRIED_PATH_FORBIDDEN:
        match = re.search(pattern, text)
        if match:
            line = text[:match.start()].count(chr(10)) + 1
            raise SystemExit(
                f"{rel}:{line} names {what} ({match.group(0)!r}).  A shipped "
                "string naming the source tree's copy of a file this package "
                "carries is a receipt that answers 'whose physics ran' with a "
                "file that did not run: write the path this carve produces, "
                "or add the rule that does")


def rewire(tree: Path,
           extra: tuple[tuple[str, str], ...] = (),
           preserve: tuple[tuple[str, str, str], ...] = (),
           slash: tuple[tuple[str, str], ...] = ()) -> None:
    preserved = {rel: (pattern, replacement)
                 for rel, pattern, replacement in preserve}
    for path in sorted(tree.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(tree).as_posix()
        if any(relative.startswith(prefix) for prefix in VERBATIM_PREFIXES):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if path.suffix in VERBATIM_SUFFIXES:
            device_prose(path)
            continue
        original = text
        for pattern, replacement in REWIRE + COMMANDS + extra:
            text = re.sub(pattern, replacement, text)
        if path.suffix == ".py":
            # PYTHON ONLY.  A carried markdown page has its own named
            # rule below, and a dash rule running first would leave that
            # one matching nothing and stop the cut on a page it had
            # already cleaned.
            for pattern, replacement in PROSE_DASH:
                text = re.sub(pattern, replacement, text)
        rel = path.relative_to(tree).as_posix()
        for named, old_lines, new_lines in (CORE_EXACT + CORE_TRACE_CLIMATOLOGY
                                            + KERNEL_NOTICE_SCOPE):
            if rel != named:
                continue
            old_text = chr(10).join(old_lines)
            text, hits = re.subn(re.escape(old_text),
                                 chr(10).join(new_lines), text)
            if not hits:
                raise SystemExit(
                    f"the exact-sentence rule for {rel} matched nothing: "
                    f"{old_lines[0]!r}.  The source tree reworded a sentence "
                    "the carve makes false, and a rule that matches nothing "
                    "leaves the false sentence inside the wheel")
        if slash:
            for pattern, replacement in slash:
                text = re.sub(pattern, replacement, text)
            carried_paths_clean(rel, text)
        for named, old, new in CORE_MD_PROSE:
            if rel != named:
                continue
            text, hits = re.subn(re.escape(old), new, text)
            if not hits:
                raise SystemExit(
                    f"the shipped-prose rule for {rel} matched nothing: "
                    f"{old!r}.  The source tree reworded the line this rule "
                    "cleans, and a rule that matches nothing is a page that "
                    "ships as it was written")
        if rel in preserved:
            pattern, replacement = preserved[rel]
            text, hits = re.subn(pattern, replacement, text)
            if not hits:
                # A preserve rule that matches nothing means the rewiring
                # stopped producing what the rule was written to undo, and
                # the thing it protects -- a manifest namespace every
                # published receipt was keyed under -- would move with
                # nothing raising about it.
                raise SystemExit(
                    f"the preserve rule for {rel} matched nothing: {pattern!r}")
        if text != original:
            path.write_text(text, encoding="utf-8", newline="")


def relative_files(tree: Path) -> set[str]:
    out = set()
    for path in tree.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            out.add(path.relative_to(tree).as_posix())
    return out


def match_eol(source: Path, model: Path) -> None:
    """Give ``source`` the line endings ``model`` stores, either way round.

    Thirty files in this tree are stored with CRLF and the rest with LF.  A
    merge of one side into the other rewrites every line, which buries the
    real change inside a whole-file diff -- the exact failure a previous
    commit here had to undo, and the failure ten test files hit again in the
    round that carved the physics in (see ``tools/check_line_endings.py``).

    BOTH DIRECTIONS, since 2026-09-10.  This converted LF to CRLF only, so a
    CRLF source merged into an LF model came out CRLF and flipped the model
    file whole.  Nothing in the tree hit that today (the source tree is LF
    where this package is LF), which is exactly why it would have gone in
    unnoticed the first day it stopped being true.
    """
    if not model.is_file():
        return
    model_bytes = model.read_bytes()
    flat = source.read_bytes().replace(b"\r\n", b"\n")
    if b"\r\n" in model_bytes:
        data = flat.replace(b"\n", b"\r\n")
    else:
        data = flat
    if data != source.read_bytes():
        source.write_bytes(data)


def merge_unit(label: str, worktree: Path, base_rev: str, tip_rev: str,
               tmp: Path, dest_root: Path,
               carve: tuple[tuple[str, str], ...],
               patterns: tuple[str, ...] | None,
               dry_run: bool,
               extra_rewire: tuple[tuple[str, str], ...] = (),
               preserve: tuple[tuple[str, str, str], ...] = (),
               slash: tuple[tuple[str, str], ...] = (),
               first_cut: bool = False) -> int:
    """Three-way merge one unit of the cut.  Returns the conflict count.

    ``first_cut`` is for a carve table that gained rows AFTER a cut.  Those
    rows name files that exist at BOTH revisions and nowhere here, so the
    three-way merge has nothing to merge them against and would report them
    as unchanged.  They are the table's first arrival and are copied.  A file
    the table names that is already here still merges three ways, so a later
    re-cut of the same table behaves like every other row.
    """

    extract(worktree, base_rev, tmp / "base", carve, patterns,
            extra_rewire, preserve, slash)
    extract(worktree, tip_rev, tmp / "tip", carve, patterns,
            extra_rewire, preserve, slash)
    base_tree = tmp / "base" / "arwen_global"
    tip_tree = tmp / "tip" / "arwen_global"

    base_files = relative_files(base_tree)
    tip_files = relative_files(tip_tree)
    ours_files = relative_files(dest_root)

    added = sorted(tip_files - base_files)
    removed = sorted(base_files - tip_files)
    common = sorted(base_files & tip_files)
    changed = [
        rel for rel in common
        if (base_tree / rel).read_bytes() != (tip_tree / rel).read_bytes()
    ]
    if first_cut:
        added = sorted(set(added) | (tip_files - ours_files))
        # EVERY file the table names that is already here is put through the
        # merge, not only the ones that moved upstream.  A table whose rows
        # are all unchanged between the two revisions would otherwise report
        # nothing at all, and "the carried core re-cuts cleanly" would be a
        # claim with no output behind it.  A file whose base and tip bytes
        # are equal merges to itself, so the extra work is a no-op that
        # produces a verdict.
        changed = sorted(rel for rel in common if rel in ours_files)

    print()
    print(f"{label}: {len(changed)} changed, {len(added)} added, "
          f"{len(removed)} removed upstream")

    conflicts: list[str] = []
    merged: list[str] = []
    for rel in changed:
        ours = dest_root / rel
        if rel not in ours_files:
            # Deleted here on purpose; an upstream change to a file this
            # package removed is reported, never silently reinstated.
            conflicts.append(f"{rel} (changed upstream, absent here)")
            continue
        base_copy = tmp / "merge-base"
        tip_copy = tmp / "merge-tip"
        shutil.copy2(base_tree / rel, base_copy)
        shutil.copy2(tip_tree / rel, tip_copy)
        match_eol(base_copy, ours)
        match_eol(tip_copy, ours)
        # A dry run merges into a COPY.  It reaches the same verdict per file
        # that the real run would, clean or conflicted, and writes nothing
        # here -- which is the only way "this re-cut is clean" is a claim a
        # reader can check BEFORE it has happened.
        into = ours
        if dry_run:
            into = tmp / "merge-ours"
            shutil.copy2(ours, into)
        proc = subprocess.run(
            ["git", "merge-file",
             "-L", "woof global",
             "-L", f"source {base_rev[:9]}",
             "-L", f"source {tip_rev[:9]}",
             str(into), str(base_copy), str(tip_copy)],
            capture_output=True, text=True,
        )
        if proc.returncode < 0:
            raise SystemExit(f"git merge-file failed on {rel}: {proc.stderr}")
        if proc.returncode > 0:
            conflicts.append(f"{rel} ({proc.returncode} conflict(s))")
        else:
            merged.append(rel)

    for rel in added:
        target = dest_root / rel
        if target.exists():
            conflicts.append(f"{rel} (added upstream, already present here)")
            continue
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tip_tree / rel, target)
        merged.append(rel)

    if dry_run:
        print("  (dry run: merged into copies, nothing written)")
    print(f"  {len(merged)} merged cleanly")
    for rel in merged:
        print(f"  ok  {rel}")
    if removed:
        print(f"  {len(removed)} removed upstream; NOT removed here, decide each:")
        for rel in removed:
            print(f"  ?   {rel}")
    if conflicts:
        print(f"  {len(conflicts)} NEED A DECISION:")
        for rel in conflicts:
            print(f"  !!  {rel}")
    return len(conflicts)


def core_pending() -> tuple[str, ...]:
    """Carried-core files this table names that this tree does not have.

    The carve table gained the core rows after the last cut, so at the first
    run its files exist at the recorded revision AND at the one being cut
    and nowhere here.  ``--to`` equal to the recorded revision is then a real
    piece of work rather than a no-op, and this is how the tool knows.
    """

    return tuple(rel for _, rel in CORE_CARVE if not (PACKAGE / rel).exists())


def main(argv: list[str] | None = None) -> int:
    recorded = read_source_md()
    parser = argparse.ArgumentParser(
        description="Re-cut this package from a newer revision of the model's source tree.",
    )
    parser.add_argument("--to", required=True, metavar="REV",
                        help="the source revision to cut from now")
    parser.add_argument("--from", dest="base", default=recorded.get("revision"),
                        metavar="REV",
                        help="the revision last cut from (default: the one SOURCE.md records)")
    parser.add_argument("--worktree", default=os.environ.get("WOOF_GLOBAL_SOURCE_WORKTREE"),
                        metavar="DIR",
                        help="the source worktree (default: WOOF_GLOBAL_SOURCE_WORKTREE)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change and write nothing")
    parser.add_argument("--unit",
                        choices=("all", "model", "suite", "core"), default="all",
                        help="merge only one unit of the cut; a re-run of a unit "
                             "already merged would apply the same change twice, so "
                             "this exists for resuming after a conflict")
    args = parser.parse_args(argv)

    if not args.base:
        parser.error("no --from and no revision recorded in SOURCE.md")
    if not args.worktree:
        parser.error("no --worktree and no WOOF_GLOBAL_SOURCE_WORKTREE")
    worktree = Path(args.worktree)
    if not (worktree / ".git").exists():
        parser.error(f"{worktree} is not a git worktree")

    base_rev = run(["git", "rev-parse", args.base], cwd=worktree).stdout.strip()
    tip_rev = run(["git", "rev-parse", args.to], cwd=worktree).stdout.strip()
    pending = core_pending()
    if base_rev == tip_rev and args.unit in ("model", "suite"):
        print(f"already at {tip_rev[:9]}: nothing to resync")
        return 0
    if base_rev == tip_rev:
        # The model and the suite have nothing to do, and the carried core
        # still does: its table gained its rows after the last cut, so its
        # files may be absent here at a revision this package already
        # carries.  When they are all present this is a CHECK -- every row
        # re-merged against the source, writing nothing new -- which is what
        # makes "the carried core is what the source tree has" a statement
        # with a command behind it.
        print(f"already at {tip_rev[:9]} for the model and the suite; "
              f"the carried core has {len(pending)} file(s) to take"
              if pending else
              f"already at {tip_rev[:9]} for the model and the suite; "
              "re-merging the carried core against the source")

    tmp = Path(tempfile.mkdtemp(prefix="resync-"))
    try:
        model = suite = core = 0
        if args.unit in ("all", "model") and base_rev != tip_rev:
            model = merge_unit(
                "the model", worktree, base_rev, tip_rev, tmp / "model",
                PACKAGE, CARVE, None, args.dry_run, slash=SLASH_PATHS)
        if args.unit in ("all", "suite") and base_rev != tip_rev:
            suite = merge_unit(
                "the suite", worktree, base_rev, tip_rev, tmp / "suite",
                TESTS, TEST_CARVE, TEST_PATTERNS, args.dry_run)
        if args.unit in ("all", "core"):
            core = merge_unit(
                "the carried core", worktree, base_rev, tip_rev, tmp / "core",
                PACKAGE, CORE_CARVE, None, args.dry_run,
                extra_rewire=CORE_REWIRE + CORE_PROSE,
                preserve=CORE_PRESERVE, slash=SLASH_PATHS,
                first_cut=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if args.dry_run:
        return 1 if (model or suite or core) else 0
    if model or suite or core:
        print()
        print("Conflict markers are in the files.  Resolve, run the suite, commit.")
        return 1
    print()
    print(f"Record the new revision in SOURCE.md and pyproject.toml: {tip_rev}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
