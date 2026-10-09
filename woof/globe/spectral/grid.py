"""Gaussian latitude/longitude grid used by the spectral transform."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import math

import numpy as np

from .constants import EARTH_RADIUS_M


@dataclass(frozen=True)
class GaussianGrid:
    truncation: int
    nlat: int
    nlon: int
    sin_lat: np.ndarray
    lat_rad: np.ndarray
    lon_rad: np.ndarray
    quadrature_weights: np.ndarray
    cos_lat: np.ndarray
    radius_m: float = EARTH_RADIUS_M

    @staticmethod
    def shape_for(
        truncation: int,
        *,
        nlat: int | None = None,
        nlon: int | None = None,
        dealias_factor: float = 1.5,
    ) -> tuple[int, int]:
        """The ``(nlat, nlon)`` a truncation implies, WITHOUT building one.

        Split out of :meth:`create` so a caller that only needs the grid
        SHAPE -- the memory estimator, which prices arrays it is refusing
        to let anything allocate -- can ask for it without paying for the
        Gauss-Legendre nodes.  A second copy of these formulas living in
        the estimator is the failure this split exists to prevent: the
        estimator would then price a grid the transform does not build,
        and a dealias change would move one and not the other.
        """

        if isinstance(truncation, bool) or not isinstance(
            truncation, (int, np.integer)
        ):
            raise ValueError("truncation must be an integer")
        t = int(truncation)
        if t < 1:
            raise ValueError(f"truncation must be >= 1, got {truncation}")
        if isinstance(dealias_factor, bool) or not math.isfinite(dealias_factor) \
                or dealias_factor < 1.0:
            raise ValueError("dealias_factor must be finite and >= 1")
        representation_lat = t + 1
        representation_lon = 2 * t + 1
        required_lat = max(
            representation_lat, int(math.ceil(dealias_factor * (t + 1)))
        )
        required_lon = max(
            representation_lon,
            int(math.ceil(2.0 * dealias_factor * (t + 1))),
        )
        if nlat is None:
            nl = required_lat
        else:
            if isinstance(nlat, bool) or not isinstance(nlat, (int, np.integer)):
                raise ValueError("nlat must be an integer")
            nl = int(nlat)
            if nl < required_lat:
                raise ValueError(
                    f"nlat={nl} is too small for T{t} with "
                    f"dealias_factor={dealias_factor:g}; need at least "
                    f"{required_lat}"
                )
        if nlon is None:
            no = required_lon
            if no % 2:
                no += 1
        else:
            if isinstance(nlon, bool) or not isinstance(nlon, (int, np.integer)):
                raise ValueError("nlon must be an integer")
            no = int(nlon)
            if no % 2:
                raise ValueError(
                    "explicit nlon must be even so positive/negative FFT "
                    "frequencies have one unambiguous layout"
                )
            if no < required_lon:
                raise ValueError(
                    f"nlon={no} is too small for T{t} with "
                    f"dealias_factor={dealias_factor:g}; need at least "
                    f"{required_lon} (rounded up to the next even value)"
                )
        return nl, no

    @classmethod
    def create(
        cls,
        truncation: int,
        *,
        nlat: int | None = None,
        nlon: int | None = None,
        dealias_factor: float = 1.5,
        radius_m: float = EARTH_RADIUS_M,
    ) -> "GaussianGrid":
        nl, no = cls.shape_for(
            truncation, nlat=nlat, nlon=nlon, dealias_factor=dealias_factor
        )
        t = int(truncation)
        if not math.isfinite(radius_m) or radius_m <= 0:
            raise ValueError("radius_m must be finite and positive")
        mu, weights = np.polynomial.legendre.leggauss(nl)
        lat = np.arcsin(mu)
        lon = np.arange(no, dtype=np.float64) * (2.0 * np.pi / no)
        coslat = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
        return cls(
            truncation=t,
            nlat=nl,
            nlon=no,
            sin_lat=mu.astype(np.float64),
            lat_rad=lat.astype(np.float64),
            lon_rad=lon,
            quadrature_weights=weights.astype(np.float64),
            cos_lat=coslat.astype(np.float64),
            radius_m=float(radius_m),
        )

    @classmethod
    def for_shape(cls, nlat: int, nlon: int, *, radius_m: float = EARTH_RADIUS_M) -> "GaussianGrid":
        """The grid of a product living on ``nlat x nlon`` Gaussian nodes.

        Its truncation is the grid's own zonal content, ``nlon/2 - 1``
        (T1535 on the 1536x3072 GFS grid); :meth:`truncated_to` then caps
        that to the truncation an analysis can afford.
        """
        return cls.create(
            int(nlon) // 2 - 1, nlat=nlat, nlon=nlon, dealias_factor=1.0,
            radius_m=radius_m,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.nlat, self.nlon

    @property
    def zonal_content_truncation(self) -> int:
        """``T_in = nlon/2 - 1``: the degree a product on this grid carries.

        A field made on this grid (a model product, a nonlinear term) has
        zonal wavenumbers up to ``nlon/2 - 1`` and, being a spherical
        field, latitudinal content of the same degree.
        """
        return self.nlon // 2 - 1

    @property
    def exact_analysis_truncation(self) -> int:
        """The largest T this grid analyses a product of its own exactly.

        A Gaussian grid with ``nlat`` nodes integrates polynomials in
        sin(lat) exactly to degree ``2*nlat - 1``, so projecting content of
        degree ``T_in`` onto degree ``T`` is exact -- no aliasing -- when
        ``T_in + T <= 2*nlat - 1``; and the retained orders ``m <= T`` must
        exist in the real FFT, ``T <= T_in``.  On 1536x3072 every T up to
        1535 is exact; on the default 1.5x-dealiased grid of T533
        (801x1602) it is 800.
        """
        t_in = self.zonal_content_truncation
        return min(t_in, 2 * self.nlat - 1 - t_in)

    def truncated_to(self, truncation: int) -> "GaussianGrid":
        """This grid's nodes under a truncation the grid did not derive.

        Refuses when the projection of this grid's own content onto
        ``truncation`` would alias, with the condition spelt out, because
        an aliased analysis returns coefficients that look right and are
        wrong in every degree the quadrature could not integrate.
        """
        if isinstance(truncation, bool) or not isinstance(
            truncation, (int, np.integer)
        ):
            raise ValueError("truncation must be an integer")
        t = int(truncation)
        if t < 1:
            raise ValueError(f"truncation must be >= 1, got {truncation}")
        t_in = self.zonal_content_truncation
        if t > t_in:
            raise ValueError(
                f"T{t} exceeds the zonal content of a {self.nlat}x{self.nlon} "
                f"grid: the real FFT of {self.nlon} longitudes carries orders "
                f"m <= nlon/2 - 1 = {t_in}, so order {t} does not exist there"
            )
        if t_in + t > 2 * self.nlat - 1:
            raise ValueError(
                f"T{t} on a {self.nlat}x{self.nlon} grid is not an exact "
                f"projection: the exactness condition T_in + T <= 2*nlat - 1 "
                f"with T_in = nlon/2 - 1 = {t_in} reads {t_in} + {t} = "
                f"{t_in + t} > {2 * self.nlat - 1}, so degrees above "
                f"{2 * self.nlat - 1 - t_in} would alias; "
                f"T <= {self.exact_analysis_truncation} is exact on this grid"
            )
        return dataclasses.replace(self, truncation=t)

    @property
    def latitude_deg(self) -> np.ndarray:
        return np.rad2deg(self.lat_rad)

    @property
    def longitude_deg(self) -> np.ndarray:
        return np.rad2deg(self.lon_rad)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        lon, lat = np.meshgrid(self.lon_rad, self.lat_rad)
        return lat, lon

    def global_mean(self, field) -> float:
        arr = np.asarray(field, dtype=np.float64)
        if arr.shape[-2:] != self.shape:
            raise ValueError(f"field has shape {arr.shape[-2:]}, expected {self.shape}")
        lon_mean = arr.mean(axis=-1)
        return float(0.5 * np.sum(lon_mean * self.quadrature_weights, axis=-1))

    def global_integral(self, field) -> float:
        return 4.0 * np.pi * self.global_mean(field)

    def rms(self, field) -> float:
        arr = np.asarray(field, dtype=np.float64)
        return math.sqrt(max(0.0, self.global_mean(arr * arr)))

    def degree_length_m(self, degree: int) -> float:
        n = int(degree)
        return math.inf if n == 0 else 2.0 * np.pi * self.radius_m / n
