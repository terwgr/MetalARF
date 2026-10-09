"""Read the Metal ABI from its canonical native header."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_HEADER = ROOT / "woof" / "metal" / "native" / "woof_metal.h"
_ABI_DEFINE = re.compile(
    r"^\s*#define\s+WOOF_METAL_ABI_VERSION\s+"
    r"\(\(\s*(\d+)u\s*<<\s*16\s*\)\s*\|\s*(\d+)u\s*\)\s*$",
    re.MULTILINE,
)


def canonical_metal_abi_version(header: Path = CANONICAL_HEADER) -> int:
    """Return the packed ABI integer declared in ``woof_metal.h``.

    This deliberately accepts only the canonical major/minor macro form so
    packaging fails closed if the header's representation changes.
    """

    try:
        source = header.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"cannot read canonical Metal ABI header {header}: {error}") from error
    matches = list(_ABI_DEFINE.finditer(source))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one canonical WOOF_METAL_ABI_VERSION definition in {header}")
    major, minor = (int(value) for value in matches[0].groups())
    if major > 0xFFFF or minor > 0xFFFF:
        raise RuntimeError(f"Metal ABI major/minor outside 16-bit range in {header}")
    return (major << 16) | minor
