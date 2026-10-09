"""The third-party notices this distribution owes, held to what it ships.

THE BREAKAGE THIS FILE PREVENTS, measured on the live 0.1.0 wheel on
2026-09-12.  That wheel carried sixteen files under
`woof/globe/core/kernels/` and four WRF parameter tables verbatim.  Among
them: five RTE+RRTMGP transcriptions and a shared header, under a
BSD-3-Clause whose clause 1 asks a source redistribution to retain the
notice, the conditions and the disclaimer; `glibc_flt32.cuh`, carrying Arm's
logf/expf/exp2f/powf under an MIT that asks for its copyright and permission
notice in every copy, and FDLIBM's expm1f and lgammaf reduction under a
notice whose one condition is that it be preserved; and four tables UCAR
asks to travel with its own notice.  The wheel's NOTICE said "Nothing from that
engine is copied into this distribution" and "This distribution ships none of
that code and none of those tables", had no section for any of the four, and
`license-files` named `LICENSE` and `NOTICE` only, so not one licence text
was inside the artefact.  Four conditions, none performed, on a public index.

So this file asserts the INVENTORY, and the assertions run against the tree
rather than against a list somebody keeps in prose:

1. Every licence text NOTICE points at is in `licenses/` and is not empty.
2. Every path NOTICE and the beside-the-code notice name as carrying a
   transcription exists, so a reader who follows a reference lands on code.
3. The beside-the-code notice names no device source this package does not
   ship, which is what the source tree's own copy of that file would do if
   the carve stopped narrowing it.
4. Every carried device source is assigned a section, and the assignment
   agrees with what the file's own header says it transcribes.  A file with
   no section fails by name, which is the question 0.1.0 never asked.
5. The two inline Python headers survive, and the notice beside the four
   WRF tables is there.
6. The gamma routines at the end of glibc_flt32.cuh are this project's own
   work, both notices say so, and nothing that ships calls them a
   transcription of a C library.  The probe is the DEFINITION of the three
   functions, not their names, so the routines themselves are what is held
   present.

The scan is text only: no import, no build, no CUDA device.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "arwen_global"
KERNELS = PACKAGE / "core" / "kernels"
NOTICE = ROOT / "NOTICE"
KERNEL_NOTICE = KERNELS / "LICENSE-third-party.txt"
TABLE_NOTICE = PACKAGE / "data" / "noah_tables" / "LICENSE-WRF.txt"

#: The texts, and the work each one is the licence for.  Named rather than
#: globbed: a text that stops shipping has to fail here, and a text that
#: arrives has to be given a reason.
LICENCE_TEXTS: dict[str, str] = {
    "LICENSE-RTE-RRTMGP-BSD-3-Clause.txt":
        "the RTE+RRTMGP radiation transcriptions",
    "LICENSE-Arm-optimized-routines-MIT.txt":
        "Arm's logf, expf, exp2f and powf cores in glibc_flt32.cuh",
    "LICENSE-FDLIBM-SunPro.txt":
        "FDLIBM's expm1f and lgammaf reduction in glibc_flt32.cuh",
    "LICENSE-WRF-public-domain.txt":
        "the four WRF parameter tables and the transcribed WRF schemes",
    "LICENSE-AER-RRTMG-BSD-3-Clause.txt":
        "AER's RRTMG McICA generator in rrtmgp_mcica.cu, its host driver "
        "and its float64 mirror",
    "NOTICE-AER-RRTMG-as-distributed-with-WRF.txt":
        "AER's own notice as WRF preserves it, all three year spans",
}

#: EVERY device source this package ships, and the work whose notice covers
#: it.  The map is the converse of the question 0.1.0 asked and got wrong.
#: Asking "is each section's list of files present" passes a file that sits
#: in the WRONG section, which is how `rrtmgp_mcica.cu` -- a transcription
#: of WRF's RRTMG generator, AER's work -- shipped under the RTE+RRTMGP
#: BSD-3-Clause with no AER text in the distribution at all.  Asking "which
#: section covers this file" fails a file with no row, and the evidence rule
#: below fails a row that disagrees with the file's own header.
#:
#: `"own"` is not a grant: it is this project's own code, written against a
#: transcription rather than transcribed, and the notice's closing scope
#: section is what says the four grants do not reach it.
DEVICE_WORK: dict[str, str] = {
    "rrtmgp_mcica.cu": "aer",
    "rrtmgp_gas.cu": "rte",
    "rrtmgp_cloud.cu": "rte",
    "rrtmgp_rte.cu": "rte",
    "glibc_flt32.cuh": "libm",
    "gf.cu": "wrf",
    "morrison.cu": "wrf",
    "noah.cu": "wrf",
    "ntiedtke.cu": "wrf",
    "sfclay.cu": "wrf",
    "ysu.cu": "wrf",
    "common.cuh": "own",
    "ysu_validation.cu": "own",
    "rrtmgp_validation.cu": "own",
    "rrtmgp_planck_common.cuh": "own",
}

#: For the two radiation grants, the section of the beside-the-code notice
#: that MUST name the file and the one that must NOT.  Those two are the
#: pair 0.1.0 crossed; the rest of the map has no section requirement here
#: because their scope is stated once in the notice's preamble
#: (`glibc_flt32.cuh`, checked by its own test below), in the root NOTICE's
#: WRF section, or in the closing scope section that says the grants do not
#: reach this project's own code.
WORK_SECTION: dict[str, tuple[str, str] | None] = {
    "aer": ("3. AER RRTMG", "4. RTE+RRTMGP"),
    "rte": ("4. RTE+RRTMGP", "3. AER RRTMG"),
    "libm": None,
    "wrf": None,
    "own": None,
}

#: What a file's OWN header has to say for its row to stand.  A header that
#: names WRF's RRTMG modules cannot sit in the RTE+RRTMGP section however
#: its filename is spelled, and that is the whole of 0.1.0's mistake.
#: Patterns, because the same upstream is named two ways in the headers
#: that cite it: rrtmgp_gas.cu and rrtmgp_cloud.cu write the repository name,
#: rrtmgp_rte.cu writes the project name.
WORK_EVIDENCE: dict[str, tuple[str, ...]] = {
    "aer": (r"module_ra_rrtmg",),
    "rte": (r"rte-rrtmgp|RTE\+RRTMGP",),
    "libm": (r"glibc",),
    "wrf": (r"WRF v4\.6\.1",),
    "own": (),
}


@pytest.fixture(scope="module")
def notice() -> str:
    return NOTICE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def kernel_notice() -> str:
    return KERNEL_NOTICE.read_text(encoding="utf-8")


def test_every_licence_text_ships_and_is_not_empty() -> None:
    directory = ROOT / "licenses"
    assert directory.is_dir(), "licenses/ is where NOTICE sends every reader"
    present = sorted(path.name for path in directory.glob("*.txt"))
    assert present == sorted(LICENCE_TEXTS), present
    for name, covers in LICENCE_TEXTS.items():
        body = (directory / name).read_text(encoding="utf-8").strip()
        assert body, f"{name} is empty, and it covers {covers}"


def test_the_notice_points_at_every_text_it_names(notice) -> None:
    """A reference to a file nobody shipped is worse than no reference."""

    for name in LICENCE_TEXTS:
        assert f"licenses/{name}" in notice, name


def test_the_notices_sit_beside_the_code(notice) -> None:
    assert KERNEL_NOTICE.is_file(), (
        "the kernels ship with no notice beside them, which is the state "
        "0.1.0 was published in")
    assert TABLE_NOTICE.is_file(), (
        "UCAR asks that its notice travel with any copy of WRF, and the "
        "four tables are verbatim copies")
    for path in (KERNEL_NOTICE, TABLE_NOTICE):
        assert path.read_text(encoding="utf-8").strip()
    assert "core/kernels/LICENSE-third-party.txt" in notice
    assert "data/noah_tables/LICENSE-WRF.txt" in notice


def test_the_python_transcriptions_open_with_their_notice() -> None:
    """The two carried modules that transcribe RTE+RRTMGP say so in place.

    Nothing digests a Python file here, so unlike the device sources these
    two can carry the notice inline, and BSD-3-Clause clause 1 is about the
    source a reader receives.
    """

    for name in ("rrtmgp.py", "npref.py"):
        head = (PACKAGE / "core" / name).read_text(encoding="utf-8")[:4000]
        assert "THIRD-PARTY NOTICE" in head, name
        assert "BSD 3-Clause" in head, name
        assert "licenses/LICENSE-RTE-RRTMGP-BSD-3-Clause.txt" in head, name
        # Both drive and mirror WRF's RRTMG McICA generator as well, and a
        # header that named only rte-rrtmgp offered a reader the wrong grant
        # over that half of the file.
        assert "licenses/LICENSE-AER-RRTMG-BSD-3-Clause.txt" in head, name
        assert "Atmospheric & Environmental Research" in head, name


def test_the_kernel_notice_names_no_file_this_package_does_not_ship(
        kernel_notice) -> None:
    """The carve narrows the source tree's copy; this is the post-condition.

    The source tree's own copy of this notice covers its whole kernel
    directory, including twenty-odd files that stay there.  Carried verbatim
    it would tell a reader this distribution contains the eight legacy RRTMG
    translation units and the Numerical Recipes coefficients, which it does
    not.  It does contain AER's McICA generator, and section 3 of the
    narrowed copy performs that notice over the one file that is here.  A
    notice that describes a different directory is a false statement about
    what somebody received, so a name in it that is not a file here fails.
    """

    here = {path.name for path in KERNELS.iterdir() if path.is_file()}
    named = set()
    for word in kernel_notice.replace(",", " ").replace("(", " ").split():
        token = word.strip(".;:)'\"")
        if token.endswith(".cu") or token.endswith(".cuh"):
            named.add(token)
    assert named, "the notice names no device source at all"
    assert named <= here, sorted(named - here)


def _sections(text: str) -> list[tuple[str, str]]:
    """The notice split at its numbered headings, heading and body."""

    out: list[tuple[str, str]] = []
    heading = "preamble"
    body: list[str] = []
    for line in text.splitlines():
        if re.match(r"^\d+\. ", line):
            out.append((heading, "\n".join(body)))
            heading, body = line, []
        else:
            body.append(line)
    out.append((heading, "\n".join(body)))
    return out


def test_every_carried_device_source_has_a_covering_section(
        kernel_notice) -> None:
    """The converse question, asked of the directory rather than of a list.

    A file with no row fails by name.  A file whose row names a section it
    is not named in fails.  A row that disagrees with the file's own header
    fails, which is what stops a second `rrtmgp_*.cu` transcribed from WRF
    from being filed under the RTE+RRTMGP grant by its prefix.
    """

    here = sorted(path.name for path in KERNELS.iterdir()
                  if path.suffix in (".cu", ".cuh"))
    assert here, "no device sources at all"
    unassigned = [name for name in here if name not in DEVICE_WORK]
    assert unassigned == [], (
        "a device source ships with no section assigned to it: "
        + ", ".join(unassigned)
        + ".  Give it a row in DEVICE_WORK and, if the work it transcribes "
          "has no section yet, a section and a licence text with it.")
    stale = sorted(set(DEVICE_WORK) - set(here))
    assert stale == [], f"DEVICE_WORK names files that are gone: {stale}"

    sections = _sections(kernel_notice)
    for name in here:
        work = DEVICE_WORK[name]
        head = (KERNELS / name).read_text(encoding="utf-8")[:2500]
        # Comment leaders and the wrap column are not evidence either way:
        # gf.cu's header breaks "WRF v4.6.1" across two comment lines.
        head = re.sub(r"[\s/*]+", " ", head)
        for token in WORK_EVIDENCE[work]:
            assert re.search(token, head), (
                f"{name} is filed under {work!r} but its own header does not "
                f"say {token!r}")
        if work != "aer":
            assert "module_ra_rrtmg" not in head, (
                f"{name} transcribes WRF's RRTMG, which is AER's work, and "
                f"it is filed under {work!r}")
        where = WORK_SECTION[work]
        if where is None:
            continue
        wanted, forbidden = where
        holders = [heading for heading, body in sections
                   if name in body and re.match(r"^\d+\. ", heading)]
        assert any(heading.startswith(wanted) for heading in holders), (
            f"{name} is not named in section {wanted!r}, which is the "
            f"section that performs the grant it stands on; the sections "
            f"naming it are {holders}")
        # A mention in the other section is allowed in one shape only: the
        # disclaimer that says the grant does not reach this file.  Section 4
        # names rrtmgp_mcica.cu on purpose, because the prefix invites the
        # reader to assume the opposite.
        for heading, body in sections:
            if not heading.startswith(forbidden) or name not in body:
                continue
            marker = "NOT covered by this section"
            flat = re.sub(r"\s+", " ", body)
            assert marker in flat and flat.index(name) > flat.index(marker), (
                f"{name} is named in section {forbidden!r} without the "
                f"disclaimer that the grant there does not cover it")

    assert "Arm optimized-routines" in kernel_notice
    assert "SPDX-License-Identifier: MIT" in kernel_notice
    assert "Sun Microsystems" in kernel_notice
    assert "earth-system-radiation/rte-rrtmgp" in kernel_notice
    assert "Atmospheric & Environmental Research" in kernel_notice


def test_the_root_notice_covers_the_same_works(notice) -> None:
    """The root NOTICE carries a section per grant, and names the carrier.

    0.1.0's root NOTICE had no AER section at all, so a reader who never
    opened the kernels directory was told the radiation was one work when it
    is two.
    """

    for heading in ("RRTMG (Atmospheric and Environmental Research, Inc.)",
                    "RTE+RRTMGP",
                    "FP32 libm transcriptions -- Arm optimized-routines",
                    "WRF / NCAR / UCAR"):
        assert heading in notice, heading
    aer = notice.split(
        "RRTMG (Atmospheric and Environmental Research, Inc.)", 1)[1]
    aer = aer.split("RTE+RRTMGP", 1)[0]
    assert "rrtmgp_mcica.cu" in aer, (
        "the AER section does not name the file it covers")
    assert "licenses/LICENSE-AER-RRTMG-BSD-3-Clause.txt" in aer
    assert "licenses/NOTICE-AER-RRTMG-as-distributed-with-WRF.txt" in aer


def test_the_libm_transcriptions_are_where_the_notice_says_they_are() -> None:
    """One file carries both libm grants, and the notice says exactly that.

    If a second device source ever gains a copy of those tables -- which is
    how the source tree ended up with fourteen -- the notice's scope
    paragraph stops being true, and this is what says so.
    """

    carriers = sorted(
        path.name for path in KERNELS.iterdir()
        if path.suffix in (".cu", ".cuh")
        and "GFK_EXP2F_TAB" in path.read_text(encoding="utf-8"))
    assert carriers == ["glibc_flt32.cuh"], carriers


#: Wording withdrawn on 2026-09-12: the gamma block was once described, on
#: the engine's record, as a transcription of a C library under a copyleft
#: grant.  That finding was withdrawn; the block is this project's own work.
#: None of these may return to anything that ships.  SPELLED IN PIECES,
#: split inside each word, because this file ships in the sdist and a table
#: that names what it forbids must not be an instance of it.
WITHDRAWN_CLAIMS: tuple[str, ...] = (
    "LG" + "PL", "gamma_pro" + "ductf", "e_gam" + "maf_r",
    "transcription of gl" + "ibc", "no notice cu" + "res", "F" + "SF",
)

#: The three sentences both notices carry instead.
OWN_WORK = "this project's own work"
OWN_WORK_ROUTINES = "gfk_gamma_product, gfk_gammaf_positive and gfk_tgamma"
NOT_A_TRANSCRIPTION = "not a transcription of any C library"


def test_the_gamma_block_is_this_project_s_own_work_and_both_notices_say_so(
        notice, kernel_notice) -> None:
    """The gamma routines are present, owned, and described as owned.

    THE PROBE IS THE DEFINITION, NOT THE NAME.  Both notices name the three
    routines in prose, so a name search cannot tell the record from the
    code; a ``__device__`` definition is the thing that ships.
    """

    header = (KERNELS / "glibc_flt32.cuh").read_text(encoding="utf-8")
    for name in ("gfk_gamma_product", "gfk_gammaf_positive", "gfk_tgamma"):
        assert f"__device__ float {name}(" in header, (
            f"{name} is not defined in glibc_flt32.cuh; the gamma this "
            "package was graded with is supposed to be here")
    for text, where in ((notice, "NOTICE"), (kernel_notice, "the kernels notice")):
        assert OWN_WORK_ROUTINES in text, where
        assert OWN_WORK in text, where
        assert NOT_A_TRANSCRIPTION in text, where


def test_no_shipped_text_calls_the_gamma_a_transcription_of_a_c_library(
        notice, kernel_notice) -> None:
    """The withdrawn wording stays out of the notices and the two kernels."""

    texts = {
        "NOTICE": notice,
        "the kernels notice": kernel_notice,
        "glibc_flt32.cuh": (KERNELS / "glibc_flt32.cuh").read_text(encoding="utf-8"),
        "gf.cu": (KERNELS / "gf.cu").read_text(encoding="utf-8"),
    }
    hits = [f"{where}: {claim!r}" for where, text in texts.items()
            for claim in WITHDRAWN_CLAIMS if claim in text]
    assert hits == [], (
        "wording withdrawn on 2026-09-12 is back in something that ships: "
        + ", ".join(hits))
