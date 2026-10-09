"""Explicit latitude rows on a uniform longitude ring: the crate's ``rows``
grid kind, for target grids no WPS map projection describes.

A global spectral model's Gaussian grid is one: its rows sit at the
Gauss-Legendre nodes, its columns are uniform in longitude, and it has
no map projection at all.  This class carries that description across
the seam to the Rust static-field builder (``static-fields``
``projection::rows``) exactly as :class:`~woof.static.projection.
ProjectedGrid` carries a Lambert or Mercator domain: the row latitudes
are handed over as DATA (the caller owns the quadrature; nothing here
re-derives a node), and the crate does every transform and every byte of
sampling.

There is deliberately no numpy transform body here.  Every other grid
in this package keeps one as the byte-parity reference for the WPS
path; this grid has no WPS reference to be parity with, and the Python
static build (``WOOF_STATIC_PYTHON=1``) refuses it by name rather than
approximating it.
"""
from __future__ import annotations

import math
import weakref

import numpy as np

from woof.static.projection import EARTH_RADIUS_M, _free_rust_grid_handle


class RowsGrid:
    """A ring of ``nlon`` uniform columns over explicit row latitudes.

    ``latitude_deg`` are the ascending mass-row centre latitudes; column
    ``i`` (1-based) sits at ``lon0_deg + (i - 1) * dlon_deg``.  The mass
    grid is ``(nlat, nlon)``; ``e_we``/``e_sn`` follow the WPS staggered
    convention every other grid uses (``mass = (e_sn - 1, e_we - 1)``).
    """

    map_proj = "rows"
    _translation_reference = None
    _translation_offset = (0, 0)

    def __init__(self, latitude_deg, lon0_deg: float, dlon_deg: float,
                 nlon: int):
        lat = np.ascontiguousarray(np.asarray(latitude_deg, dtype=np.float64))
        if lat.ndim != 1 or lat.size < 2:
            raise ValueError(
                "RowsGrid needs a one-dimensional table of at least two "
                f"row latitudes, got shape {lat.shape}")
        if not np.all(np.isfinite(lat)) or np.any(np.diff(lat) <= 0.0):
            raise ValueError(
                "RowsGrid row latitudes must be finite and strictly ascending")
        if float(lat[0]) < -90.0 or float(lat[-1]) > 90.0:
            raise ValueError("RowsGrid row latitudes must lie in [-90, 90]")
        nlon = int(nlon)
        if nlon < 1:
            raise ValueError(f"RowsGrid needs at least one column, got {nlon}")
        dlon = float(dlon_deg)
        if not math.isfinite(dlon) or dlon <= 0.0:
            raise ValueError(f"RowsGrid dlon_deg must be positive, got {dlon}")
        self.latitude_deg = lat
        self.lon0_deg = float(lon0_deg)
        self.dlon_deg = dlon
        self.nlat = int(lat.size)
        self.nlon = nlon
        self.e_we = nlon + 1
        self.e_sn = self.nlat + 1
        # The crate's gcell-ratio and precision selector: one equatorial
        # column width on the WPS sphere.
        self.dx = 2.0 * math.pi * EARTH_RADIUS_M / nlon
        self.dy = self.dx

    @classmethod
    def from_gaussian(cls, grid) -> "RowsGrid":
        """The rows description of a ``GaussianGrid`` (its own nodes, its
        own first longitude, ``360 / nlon`` spacing)."""
        lon = np.asarray(grid.longitude_deg, dtype=np.float64)
        return cls(grid.latitude_deg, float(lon[0]), 360.0 / int(grid.nlon),
                   int(grid.nlon))

    @property
    def closes(self) -> bool:
        """Does the ring cover exactly 360 degrees?"""
        return abs(self.dlon_deg * self.nlon - 360.0) < 1.0e-9 * 360.0

    # -- the Rust seam --------------------------------------------------------

    def _rust_spec(self) -> dict:
        """This grid as the crate's ``GridSpec`` JSON document.  The WPS
        parameters are carried at their neutral values; the crate reads
        only ``lat_deg``/``lon0_deg``/``dlon_deg`` for this kind."""
        return {
            "kind": self.map_proj,
            "ref_lat": 0.0, "ref_lon": 0.0,
            "truelat1": 0.0, "truelat2": 0.0, "stand_lon": 0.0,
            "dx": self.dx, "dy": self.dy,
            "e_we": self.e_we, "e_sn": self.e_sn,
            "known_x": 1.0, "known_y": 1.0,
            "moad_cen_lat": 0.0, "moad_cen_lon": 0.0,
            "lat_deg": [float(v) for v in self.latitude_deg],
            "lon0_deg": self.lon0_deg,
            "dlon_deg": self.dlon_deg,
        }

    def _rust_handle(self, bridge) -> int:
        """The cached crate grid handle (reference handle + integer
        offset for a sector, exactly as ``ProjectedGrid._rust_handle``)."""
        handle = self.__dict__.get("_rust_grid_handle")
        if handle is not None:
            return handle
        base = self._translation_reference
        if base is None:
            handle = bridge.grid_new(self._rust_spec())
        else:
            di, dj = self._translation_offset
            handle = bridge.grid_translated(
                base._rust_handle(bridge), di, dj, self.e_we, self.e_sn)
        self.__dict__["_rust_grid_handle"] = handle
        weakref.finalize(self, _free_rust_grid_handle, handle)
        return handle

    def _rust_sampling_handle(self, bridge) -> int:
        """The handle the engine's static sampler reads (from woof 2.8.0).

        THE BREAKAGE THIS PREVENTS.  The 2.8 engine's static build asks the
        grid for a SAMPLING handle, which keeps a nested grid's integer
        ancestry, instead of the coordinate handle, and a grid without the
        method raised AttributeError on the first sector of every global
        statics build.  A rows grid has no nest ancestry: a whole grid is
        its own spec and a sector is its reference plus an integer column
        offset, which is exactly what :meth:`_rust_handle` already hands
        the crate, so the two handles are the same one.
        """
        return self._rust_handle(bridge)

    # -- sectors --------------------------------------------------------------

    def sector(self, first_column: int, columns: int) -> "RowsGrid":
        """Columns ``first_column .. first_column + columns - 1`` (0-based)
        of this ring as a translated grid.

        The sector DELEGATES its transforms to this ring at an exact
        integer column offset (the crate's ``translated`` rule), so a
        source pixel bins onto the same column whether the ring is built
        whole or in sectors; sectors exist only to bound the source window
        a whole-globe build would otherwise read at once.
        """
        if self._translation_reference is not None:
            raise ValueError("a sector is cut from the reference ring, not "
                             "from another sector")
        first = int(first_column)
        count = int(columns)
        if first < 0 or count < 1 or first + count > self.nlon:
            raise ValueError(
                f"sector columns {first}..{first + count - 1} fall outside "
                f"the ring's {self.nlon} columns")
        piece = RowsGrid.__new__(RowsGrid)
        piece.__dict__.update({
            key: value for key, value in self.__dict__.items()
            if key != "_rust_grid_handle"})
        piece.nlon = count
        piece.e_we = count + 1
        piece._translation_reference = self
        piece._translation_offset = (first, 0)
        return piece

    def sectors(self, degrees: float) -> list["RowsGrid"]:
        """The ring cut into sectors of about ``degrees`` of longitude
        each (the last one takes the remainder)."""
        width = float(degrees)
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError(f"sector width must be positive, got {degrees}")
        # At least two: the crate refuses a source window wider than the
        # global dataset, and a whole-ring window plus its margin is.
        count = max(2, int(math.ceil(self.dlon_deg * self.nlon / width - 1e-9)))
        columns = max(1, int(math.ceil(self.nlon / count)))
        out = []
        start = 0
        while start < self.nlon:
            count = min(columns, self.nlon - start)
            out.append(self.sector(start, count))
            start += count
        return out


__all__ = ["RowsGrid"]
