"""``recast-woof-data`` -- the bulk physics reference tables ``woof`` reads.

This distribution exists for one reason, and it is a channel limit rather
than a design preference: PyPI rejects any file over 100 MiB, and the
2.5.0 ``woof`` wheel measured 103.62 MiB.  Two directories of pure
reference data accounted for 64.21 MiB of it -- the RRTMGP k-distribution
and cloud-optics NetCDFs, and the Thompson microphysics lookup tables --
so they ship here instead and ``woof`` depends on this package at the
exact same version.

It is NOT an optional extra.  ``pip install recast-woof`` installs it, because
a bare install that cannot run radiation or Thompson microphysics is not
an install.  Nothing here is platform-specific: every byte is a table, so
this wheel is ``py3-none-any`` and one file serves every platform
``woof`` runs on.

Layout mirrors where the files used to live, exactly::

    woof_data/data/rrtmgp/...          was woof/data/rrtmgp/...
    woof_data/data/thompson/tables/... was woof/data/thompson/tables/...

That is deliberate: :mod:`woof.data_assets` resolves a
``woof/data``-relative path against one root or the other, so the bytes
reaching every call site are the same bytes at the same relative path,
and a directory that moves later needs one entry in
``woof.data_assets.COMPANION_TREES`` and no other edit.

One directory never lived under ``woof/data``: ``woof_data/data/basemap``
holds the Natural Earth and US Census shapefiles the renderer draws
coastlines, borders, state lines and counties from.  A platform ``woof``
wheel carries the renderer but has no room for them, and this
distribution arrives with every install, so ``woof.rustwx`` hands this
directory to the renderer (``woof.data_assets.companion_basemap_dir``).

Nothing in here imports ``woof``.  A data package that needed its
consumer installed to answer where its own files are would be a cycle
with no purpose.
"""

from __future__ import annotations

from pathlib import Path

#: Distribution name on PyPI, spelled once.
DISTRIBUTION_NAME = "recast-woof-data"

try:  # installed distribution metadata, never a second literal
    from importlib.metadata import version as _distribution_version

    __version__ = _distribution_version(DISTRIBUTION_NAME)
except Exception:  # pragma: no cover - a source tree that was never installed
    __version__ = "0+unknown"


def data_root() -> Path:
    """Directory holding the moved trees, laid out as ``woof/data`` was.

    Returned as a real filesystem :class:`~pathlib.Path` because that is
    what the consumers need: ``netCDF4.Dataset`` opens a path,
    ``numpy.fromfile`` opens a path, and the Thompson loader hashes one.
    This distribution is a plain directory install -- no zip import, no
    extension modules -- so the path always exists.
    """

    return Path(__file__).resolve().parent / "data"


__all__ = ["DISTRIBUTION_NAME", "__version__", "data_root"]
