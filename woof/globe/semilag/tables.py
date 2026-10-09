"""The grid tables a semi-Lagrangian gather reads, built once per grid.

Everything here is geometry: the extended latitude table that lets a
four-point meridional stencil cross a pole, the reciprocal Lagrange
denominators of the true Gauss-Legendre nodes, and the bracket lookup that
turns a departure latitude into a stencil start in constant time.  No field
data and no time step reach this module.

The two facts that make the extension exact rather than an approximation:

*   A point at latitude ``phi < -pi/2`` is the point at latitude
    ``-pi - phi`` and longitude ``lambda + pi``.  So the row one step south
    of the southernmost Gaussian ring IS that ring, read half a grid around
    in longitude, and its latitude is ``-pi - lat[0]``.  The same holds at
    the north pole with ``pi - lat[nlat-1]``.
*   A geocentric CARTESIAN component of a tangent vector field is a plain
    scalar function on the sphere, so it crosses the pole with no sign
    change and no basis rotation.  A local east-north component does not.
    That is why :mod:`woof.globe.semilag.trajectory` advects the
    wind as Cartesian components: the sign flip a local basis needs is easy
    to get wrong and shows up nowhere except in the two outermost rings,
    which are 0.26 percent of a T255 grid.

Two stencil widths share one table object.  The four-point tables carry
two reflected rows beyond each pole and serve the cubic gather; the
six-point tables (``lat_ext6``, ``mrden6``, ``rowoff6``, ``rowshift6``,
``mlut6``) carry three and serve the quintic gather of the dynamical
bundle (``[semilag] horizontal_interpolation = "quintic_lagrange"``).  They
are separate arrays rather than one wider table because the cubic gather's
bracket, its row offsets and its reciprocal denominators are indexed from
the cubic table's own origin, and the cubic path is pinned bit for bit by
the gates that were measured on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from woof.globe.spectral.grid import GaussianGrid


#: Bins of the bracket lookup per extended latitude row.  The in-kernel
#: correction loops make any value correct; four keeps them at zero or one
#: step for a Gauss-Legendre distribution, which is nearly uniform in angle.
LOOKUP_BINS_PER_ROW = 4


def _six_point_tables(lat: np.ndarray, nlat: int, nlon: int):
    """The extended latitude table, row maps, reciprocal Lagrange
    denominators and bracket lookup of the six-point meridional stencil."""
    if nlat < 6:
        raise ValueError(
            f"a six-point meridional stencil needs nlat >= 6, got {nlat}"
        )
    ghost = 3
    lat_ext = np.empty(nlat + 2 * ghost, dtype=np.float64)
    for g in range(ghost):
        lat_ext[ghost - 1 - g] = -math.pi - lat[g]
        lat_ext[nlat + ghost + g] = math.pi - lat[nlat - 1 - g]
    lat_ext[ghost:nlat + ghost] = lat
    if not np.all(np.diff(lat_ext) > 0.0):
        raise ValueError(
            "the three-row pole-extended latitude table is not strictly "
            "ascending; the grid's outermost rings sit closer to a pole than "
            "the reflection of the rings inside them, which no Gaussian "
            "grid does and which the bracket search cannot represent"
        )
    rows = np.arange(nlat + 2 * ghost, dtype=np.int64) - ghost
    data_row = np.where(
        rows < 0, -1 - rows,
        np.where(rows >= nlat, 2 * nlat - 1 - rows, rows),
    )
    shifted = (rows < 0) | (rows >= nlat)
    rowoff = (data_row * nlon).astype(np.int32)
    rowshift = np.where(shifted, nlon // 2, 0).astype(np.int32)
    starts = np.arange(nlat + 1, dtype=np.int64)
    nodes = lat_ext[starts[:, None] + np.arange(6)[None, :]]
    mrden = np.empty((nlat + 1, 6), dtype=np.float64)
    for m in range(6):
        den = np.ones(nlat + 1, dtype=np.float64)
        for q in range(6):
            if q == m:
                continue
            den *= nodes[:, m] - nodes[:, q]
        mrden[:, m] = 1.0 / den
    lookup_bins = LOOKUP_BINS_PER_ROW * (nlat + 2 * ghost)
    lut_scale = lookup_bins / math.pi
    centres = -0.5 * math.pi + (np.arange(lookup_bins) + 0.5) / lut_scale
    mlut = np.searchsorted(lat_ext, centres, side="right") - 1
    mlut = np.clip(mlut, 2, nlat + 2).astype(np.int32)
    return lat_ext, rowoff, rowshift, mrden, mlut


@dataclass(frozen=True)
class SphericalGridTables:
    """Device-resident geometry for one Gaussian grid and one dtype."""

    nlat: int
    nlon: int
    radius_m: float
    dlam: float
    inv_dlam: float
    lut_scale: float
    lookup_bins: int
    lat_ext: Any
    mrden: Any
    mlut: Any
    rowoff: Any
    rowshift: Any
    xp: Any
    dtype: Any
    metal_context: Any | None
    #: The host float64 copies, kept because the numpy reference path and
    #: the tests read them and because a float32 table is not the
    #: specification of anything.
    lat_ext_host: np.ndarray
    mrden_host: np.ndarray
    rowoff_host: np.ndarray
    rowshift_host: np.ndarray
    #: The six-point tables of the quintic gather (module docstring).
    lat_ext6: Any
    mrden6: Any
    mlut6: Any
    rowoff6: Any
    rowshift6: Any
    lookup_bins6: int
    lut_scale6: float
    lat_ext6_host: np.ndarray | None
    mrden6_host: np.ndarray | None
    rowoff6_host: np.ndarray | None
    rowshift6_host: np.ndarray | None
    #: Metal copies of the immutable gather lookup tables, keyed by stencil
    #: order.  The table object and its context have the same lifetime.
    _metal_gather_resources: dict = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    @classmethod
    def create(cls, grid: GaussianGrid, *, xp, dtype, metal_context=None) -> "SphericalGridTables":
        nlat = int(grid.nlat)
        nlon = int(grid.nlon)
        if nlat < 4:
            raise ValueError(
                f"a four-point meridional stencil needs nlat >= 4, got {nlat}"
            )
        if nlon % 2:
            raise ValueError(
                f"nlon must be even so the polar reflection is an exact "
                f"shift of nlon/2 columns, got {nlon}"
            )
        lat = np.asarray(grid.lat_rad, dtype=np.float64)
        if lat.shape != (nlat,):
            raise ValueError(f"lat_rad must have shape ({nlat},)")
        if not np.all(np.diff(lat) > 0.0):
            raise ValueError(
                "lat_rad must be strictly ascending; the bracket search and "
                "the pole reflection both read it as a monotone table"
            )
        lon = np.asarray(grid.lon_rad, dtype=np.float64)
        dlam = 2.0 * math.pi / nlon
        drift = float(np.max(np.abs(lon - np.arange(nlon) * dlam)))
        if drift > 1e-12:
            raise ValueError(
                "the zonal weights are the equispaced cubic Lagrange weights, "
                f"which needs lon[i] = i*2*pi/nlon; this grid drifts by "
                f"{drift:.3e} rad"
            )

        lat_ext = np.empty(nlat + 4, dtype=np.float64)
        lat_ext[0] = -math.pi - lat[1]
        lat_ext[1] = -math.pi - lat[0]
        lat_ext[2:nlat + 2] = lat
        lat_ext[nlat + 2] = math.pi - lat[nlat - 1]
        lat_ext[nlat + 3] = math.pi - lat[nlat - 2]
        if not np.all(np.diff(lat_ext) > 0.0):
            raise ValueError(
                "the pole-extended latitude table is not strictly ascending; "
                "the grid's outermost rings sit closer to a pole than the "
                "reflection of the ring inside them, which no Gaussian grid "
                "does and which the bracket search cannot represent"
            )

        rows = np.arange(nlat + 4, dtype=np.int64) - 2
        data_row = np.where(
            rows < 0, -1 - rows,
            np.where(rows >= nlat, 2 * nlat - 1 - rows, rows),
        )
        shifted = (rows < 0) | (rows >= nlat)
        rowoff = (data_row * nlon).astype(np.int32)
        rowshift = np.where(shifted, nlon // 2, 0).astype(np.int32)

        # Reciprocal Lagrange denominators for every stencil start.
        starts = np.arange(nlat + 1, dtype=np.int64)
        nodes = lat_ext[starts[:, None] + np.arange(4)[None, :]]
        mrden = np.empty((nlat + 1, 4), dtype=np.float64)
        for m in range(4):
            den = np.ones(nlat + 1, dtype=np.float64)
            for q in range(4):
                if q == m:
                    continue
                den *= nodes[:, m] - nodes[:, q]
            mrden[:, m] = 1.0 / den

        lookup_bins = LOOKUP_BINS_PER_ROW * (nlat + 4)
        lut_scale = lookup_bins / math.pi
        centres = -0.5 * math.pi + (np.arange(lookup_bins) + 0.5) / lut_scale
        mlut = np.searchsorted(lat_ext, centres, side="right") - 1
        mlut = np.clip(mlut, 1, nlat + 1).astype(np.int32)

        # The six-point tables: three reflected rows beyond each pole, the
        # reciprocal denominators of every six-node Lagrange stencil, and a
        # bracket lookup over the wider table.  A stencil start ``sst`` in
        # ``[0, nlat]`` reads rows ``sst .. sst+5`` of the table, and the
        # bracket ``m0`` (``lat_ext6[m0] <= p < lat_ext6[m0+1]``) for any
        # latitude inside ``[-pi/2, pi/2]`` lies in ``[2, nlat+2]``, so
        # ``sst = m0 - 2`` puts the departure latitude between the third
        # and fourth nodes of its stencil, which is where a six-point
        # Lagrange interpolant is best conditioned.
        # A grid too small for six rows (nlat < 6) keeps its cubic tables
        # and the quintic gather refuses it by name (interpolate.gather_batch).
        if nlat >= 6:
            lat_ext6, rowoff6, rowshift6, mrden6, mlut6 = _six_point_tables(
                lat, nlat, nlon
            )
        else:
            lat_ext6 = rowoff6 = rowshift6 = mrden6 = mlut6 = None

        return cls(
            nlat=nlat,
            nlon=nlon,
            radius_m=float(grid.radius_m),
            dlam=dlam,
            inv_dlam=1.0 / dlam,
            lut_scale=lut_scale,
            lookup_bins=int(lookup_bins),
            lat_ext=xp.asarray(lat_ext, dtype=dtype),
            mrden=xp.asarray(np.ascontiguousarray(mrden.reshape(-1)),
                             dtype=dtype),
            mlut=xp.asarray(mlut, dtype=np.int32),
            rowoff=xp.asarray(rowoff, dtype=np.int32),
            rowshift=xp.asarray(rowshift, dtype=np.int32),
            xp=xp,
            dtype=dtype,
            metal_context=metal_context,
            lat_ext_host=lat_ext,
            mrden_host=mrden,
            rowoff_host=rowoff,
            rowshift_host=rowshift,
            lat_ext6=None if lat_ext6 is None else xp.asarray(lat_ext6, dtype=dtype),
            mrden6=None if mrden6 is None else xp.asarray(
                np.ascontiguousarray(mrden6.reshape(-1)), dtype=dtype),
            mlut6=None if mlut6 is None else xp.asarray(mlut6, dtype=np.int32),
            rowoff6=None if rowoff6 is None else xp.asarray(rowoff6, dtype=np.int32),
            rowshift6=None if rowshift6 is None else xp.asarray(rowshift6, dtype=np.int32),
            lookup_bins6=0 if mlut6 is None else int(mlut6.shape[0]),
            lut_scale6=0.0 if mlut6 is None else float(mlut6.shape[0] / math.pi),
            lat_ext6_host=lat_ext6,
            mrden6_host=mrden6,
            rowoff6_host=rowoff6,
            rowshift6_host=rowshift6,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.nlat, self.nlon

    def equatorial_spacing_m(self) -> float:
        """``a * 2*pi / nlon``: one zonal cell at the equator.

        The Lipschitz diagnostic uses this as its single reference length,
        so its vertical and horizontal entries share units without any
        term inheriting the polar collapse of the local zonal spacing.
        """
        return self.radius_m * self.dlam
