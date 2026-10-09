"""The YSU free-atmosphere mixing-length modes, runtime-free.

``kernels/ysu.cu`` (WRF v4.6.1 ``phys/physics_mmm/bl_ysu.F90``) sets the
asymptotic mixing length of its local, Richardson-number diffusivity
above the boundary layer from the LAYER THICKNESS: ``rlamdz =
min(max(0.1 dz, 30 m), 300 m)`` (bl_ysu.F90:1003, ysu.cu ``rlamdz``), so
the diffusivity ``K = l^2 |dV/dz| f(Ri)`` grows with the square of the
layer thickness for the same resolved shear and Richardson number, and
the momentum flux across an interface with the square of the spacing.
Measured on WOOF global's 40-level stack, whose layers across the jet
are 55 hPa (about 1500 m) thick: the length reads 150 m where a regional
column with 300 m layers reads 30 m, a 25x diffusivity for the same
shear, and the physics drains the 100 to 400 km kinetic energy at
237 hPa at 0.7 to 1.2 per day (woof.globe.pbl_free_atmosphere).

Two modes, selected per launch and never mixed inside a column:

* ``"wrf-layer"``: WRF's own rule, bit for bit.  The regional model and
  every parity gate against the frozen Fortran use it.
* ``"fixed"``: the asymptotic length is the scheme's own ``rlam`` (30 m,
  the value WRF's rule gives every layer thinner than 300 m) at every
  spacing, so the free-atmosphere flux for a resolved shear no longer
  grows with the layer thickness.  Everything inside the boundary layer
  (the profile K, the countergradient and entrainment terms) is untouched.
  Selectable in the WOOF global native suite
  (``ysu_free_atmosphere_mixing_length = "fixed"``), not its default: the
  option's note in ``woof.globe.physics.native_options`` carries
  the grade that kept WRF's rule as the default.
"""
from __future__ import annotations

#: mode name -> the integer the kernel and the float64 mirror switch on.
YSU_FREE_ATMOSPHERE_MIXING_LENGTHS: dict[str, int] = {
    "wrf-layer": 0,
    "fixed": 1,
}

#: The asymptotic length of the ``"fixed"`` mode: WRF's ``rlam`` (bl_ysu.F90:143).
YSU_FIXED_ASYMPTOTIC_LENGTH_M = 30.0


def free_atmosphere_mixing_length_flag(mode: str) -> int:
    """The kernel flag for ``mode``; refuses any name outside the table."""
    try:
        return YSU_FREE_ATMOSPHERE_MIXING_LENGTHS[str(mode)]
    except KeyError:
        raise ValueError(
            "free_atmosphere_mixing_length must be one of "
            f"{sorted(YSU_FREE_ATMOSPHERE_MIXING_LENGTHS)}, got {mode!r}"
        ) from None


__all__ = [
    "YSU_FIXED_ASYMPTOTIC_LENGTH_M",
    "YSU_FREE_ATMOSPHERE_MIXING_LENGTHS",
    "free_atmosphere_mixing_length_flag",
]
