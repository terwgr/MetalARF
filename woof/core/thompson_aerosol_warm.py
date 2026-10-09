"""Thompson aerosol-aware warm and frozen source-network launchers."""

from __future__ import annotations

import numpy as np

from woof.core.thompson_aerosol_launch import aerosol_kernel
from woof.core.state import DTYPE
from woof.core.thompson_aerosol_launch import (
    WARM_MODULE,
    launch_grid,
    validate_fields,
    validate_fp64_fortran_table,
)

#: WRF's ordered rain/snow collision table names (module_mp_thompson.F
#: qr_acr_qsV2 cache), matching ``woof.core.thompson_runtime``'s
#: ``DeviceClassicColdSourceTables.rain_snow_tables`` order exactly.
RAIN_SNOW_TABLE_NAMES = (
    "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
    "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
    "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2",
)

#: WRF's ordered rain/graupel collision table names (qr_acr_qg_V4 cache).
RAIN_GRAUPEL_TABLE_NAMES = (
    "tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr",
)

RAIN_SNOW_TABLE_SHAPE = (37, 9, 37, 37)
RAIN_GRAUPEL_TABLE_SHAPE = (37, 37, 1, 37, 37)
#: The operational WRF 3.9 fork's shapes (thompson_version = "wrf_39_noaa",
#: woof.core.thompson_contract.FORK_GENERATED_TABLE_FILES).
WRF39_RAIN_SNOW_TABLE_SHAPE = (28, 9, 37, 37)
WRF39_RAIN_GRAUPEL_TABLE_SHAPE = (28, 28, 37, 37)


def _rain_snow_shape():
    from woof.core.thompson_aerosol_launch import active_thompson_version
    return (WRF39_RAIN_SNOW_TABLE_SHAPE
            if active_thompson_version() == "wrf_39_noaa"
            else RAIN_SNOW_TABLE_SHAPE)


def _rain_graupel_shape():
    from woof.core.thompson_aerosol_launch import active_thompson_version
    return (WRF39_RAIN_GRAUPEL_TABLE_SHAPE
            if active_thompson_version() == "wrf_39_noaa"
            else RAIN_GRAUPEL_TABLE_SHAPE)
EFFICIENCY_TABLE_SHAPE = (100, 100)


def _arrays_overlap(left, right) -> bool:
    """Return whether two contiguous device arrays share any storage.

    Transcribed from ``woof/core/thompson.py:22-48``; that module is
    read-only for this port and ``_arrays_overlap`` is a private name there.
    """
    if left is right:
        return True

    def interval(value):
        interface = getattr(value, "__cuda_array_interface__", None)
        address_space = "cuda"
        if interface is None:
            interface = getattr(value, "__array_interface__", None)
            address_space = "host"
        if interface is None:
            return None
        pointer = int(interface["data"][0] or 0)
        nbytes = int(value.nbytes)
        return address_space, pointer, pointer + nbytes

    left_interval = interval(left)
    right_interval = interval(right)
    if left_interval is None or right_interval is None:
        return False
    left_space, left_start, left_end = left_interval
    right_space, right_start, right_end = right_interval
    return (left_space == right_space
            and left_start < right_end
            and right_start < left_end)


def _validate_dt(dt: float) -> None:
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")


def _validate_collision_tables(rain_snow_tables, rain_graupel_tables):
    rain_snow_values = tuple(rain_snow_tables)
    rain_graupel_values = tuple(rain_graupel_tables)
    if len(rain_snow_values) != len(RAIN_SNOW_TABLE_NAMES):
        raise ValueError(
            "rain_snow_tables must contain "
            f"{len(RAIN_SNOW_TABLE_NAMES)} arrays, "
            f"got {len(rain_snow_values)}")
    if len(rain_graupel_values) != len(RAIN_GRAUPEL_TABLE_NAMES):
        raise ValueError(
            "rain_graupel_tables must contain "
            f"{len(RAIN_GRAUPEL_TABLE_NAMES)} arrays, "
            f"got {len(rain_graupel_values)}")
    for name, table in zip(
            RAIN_SNOW_TABLE_NAMES, rain_snow_values, strict=True):
        validate_fp64_fortran_table(name, table, _rain_snow_shape())
    for name, table in zip(
            RAIN_GRAUPEL_TABLE_NAMES, rain_graupel_values, strict=True):
        validate_fp64_fortran_table(
            name, table, _rain_graupel_shape())
    return rain_snow_values, rain_graupel_values


def launch_aerosol_warm_source_network(
        qc, qr, nr, qs, qg, graupel_number_shadow,
        graupel_melt_marker, snow_melt_marker,
        temperature, pressure, qv,
        nc_entry, nwfa_entry, nifa_entry,
        ncten, nwfaten, nifaten,
        rain_cloud_efficiency, snow_cloud_efficiency,
        rain_snow_tables, rain_graupel_tables, dt: float) -> None:
    """Apply WRF-ordered warm-level sources with a prognostic droplet number.

    ``nc_entry``/``nwfa_entry``/``nifa_entry`` are the FROZEN per-kilogram
    entry state for the whole mp=28 call and are never written.  The droplet
    and aerosol tendencies are ADDED to ``ncten``/``nwfaten``/``nifaten``
    (per kilogram per second), which a terminal state kernel applies once
    with WRF's clamps.  Hydrometeor mass, rain number, graupel number,
    vapour and temperature are updated in place exactly as the mp=8 warm
    network updates them.

    ``graupel_melt_marker`` must arrive carrying the held ``T >= 273.15 K``
    entry mask; the kernel consumes it and overwrites it with WRF's held
    ``prr_gml > 0`` decision.  ``snow_melt_marker`` receives the independent
    ``prr_sml > 0`` decision.

    This launcher does NOT run the ncten balance limiter.  Call
    :func:`launch_ncten_balance` once, after this and after the cold
    network, and before the saturation adjustment.
    """
    from woof.metal.array import MetalArray
    if isinstance(qc, MetalArray):
        from woof.core.thompson_aerosol_launch import active_thompson_version
        from woof.metal.thompson_aerosol_warm_family import (
            submit_warm_source_network,
        )
        rain_snow_values = tuple(rain_snow_tables)
        rain_graupel_values = tuple(rain_graupel_tables)
        submit_warm_source_network(
            qc, qr, nr, qs, qg, graupel_number_shadow,
            graupel_melt_marker, snow_melt_marker,
            temperature, pressure, qv,
            nc_entry, nwfa_entry, nifa_entry,
            ncten, nwfaten, nifaten,
            rain_cloud_efficiency, snow_cloud_efficiency,
            *rain_snow_values, *rain_graupel_values, dt, qc.size,
            wrf39=active_thompson_version() == "wrf_39_noaa")
        return
    _, size = validate_fields({
        "qc": qc,
        "qr": qr,
        "nr": nr,
        "qs": qs,
        "qg": qg,
        "graupel_number_shadow": graupel_number_shadow,
        "graupel_melt_marker": graupel_melt_marker,
        "snow_melt_marker": snow_melt_marker,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "nc_entry": nc_entry,
        "nwfa_entry": nwfa_entry,
        "nifa_entry": nifa_entry,
        "ncten": ncten,
        "nwfaten": nwfaten,
        "nifaten": nifaten,
    })
    if _arrays_overlap(graupel_melt_marker, snow_melt_marker):
        raise ValueError(
            "snow_melt_marker must not alias graupel_melt_marker")
    # The accumulator contract is the whole point of this package: an
    # accumulator aliasing its own entry state would make the "read-only
    # entry state" guarantee silently false.
    for entry_name, entry in (("nc_entry", nc_entry),
                              ("nwfa_entry", nwfa_entry),
                              ("nifa_entry", nifa_entry)):
        for accum_name, accum in (("ncten", ncten),
                                  ("nwfaten", nwfaten),
                                  ("nifaten", nifaten)):
            if _arrays_overlap(entry, accum):
                raise ValueError(
                    f"{accum_name} must not alias {entry_name}; mp=28 entry "
                    "state is read-only for the whole call")
    for name, accum in (("nwfaten", nwfaten), ("nifaten", nifaten)):
        if _arrays_overlap(ncten, accum):
            raise ValueError(f"{name} must not alias ncten")
    if _arrays_overlap(nwfaten, nifaten):
        raise ValueError("nifaten must not alias nwfaten")
    validate_fp64_fortran_table(
        "rain_cloud_efficiency", rain_cloud_efficiency,
        EFFICIENCY_TABLE_SHAPE)
    validate_fp64_fortran_table(
        "snow_cloud_efficiency", snow_cloud_efficiency,
        EFFICIENCY_TABLE_SHAPE)
    rain_snow_values, rain_graupel_values = _validate_collision_tables(
        rain_snow_tables, rain_graupel_tables)
    _validate_dt(dt)

    # Smaller blocks admit more resident warps without changing cell arithmetic.
    grid, block = launch_grid(size, threads=64)
    aerosol_kernel(WARM_MODULE, "thompson_aa_warm_source_network")(
        grid, block,
        (qc, qr, nr, qs, qg, graupel_number_shadow,
         graupel_melt_marker, snow_melt_marker,
         temperature, pressure, qv,
         nc_entry, nwfa_entry, nifa_entry,
         ncten, nwfaten, nifaten,
         rain_cloud_efficiency, snow_cloud_efficiency,
         *rain_snow_values, *rain_graupel_values,
         DTYPE(dt), np.int32(size)))


def launch_aerosol_warm_source_network_from_owner(
        qc, qr, nr, qs, qg, graupel_number_shadow,
        graupel_melt_marker, snow_melt_marker,
        temperature, pressure, qv,
        nc_entry, nwfa_entry, nifa_entry,
        ncten, nwfaten, nifaten,
        table_owner, dt: float) -> None:
    """Launch the warm network from one verified classic table owner.

    mp=28 reuses the four classic Thompson caches unchanged; only
    ``CCN_ACTIVATE.BIN`` is new, and the warm network never reads it.
    """
    from woof.core.thompson_runtime import DeviceClassicTableSet

    if (not isinstance(table_owner, DeviceClassicTableSet)
            or not table_owner.roundtrip_verified):
        raise TypeError(
            "table_owner must be a verified DeviceClassicTableSet")
    tables = table_owner.cold_source_tables
    launch_aerosol_warm_source_network(
        qc, qr, nr, qs, qg, graupel_number_shadow,
        graupel_melt_marker, snow_melt_marker,
        temperature, pressure, qv,
        nc_entry, nwfa_entry, nifa_entry,
        ncten, nwfaten, nifaten,
        tables.rain_cloud_efficiency, table_owner.t_Efsw,
        tables.rain_snow_tables, tables.rain_graupel_tables, dt)


def launch_ncten_balance(
        qc_entry, qc_after, nc_entry, density, ncten, dt: float) -> None:
    """Apply WRF's cloud water mass/number balance ONCE.

    ``module_mp_thompson.F:2996-3019``.  Keeps the mass-weighted mean droplet
    size between 1 and 100 microns and the total below ``Nt_c_max``, by
    OVERWRITING ``ncten`` wherever a clamp fires -- the limiter backs the
    tendency out against ``nc1d*rho``, it does not add to it.

    Sequencing is part of the contract.  WRF runs this once per column after
    every ``ncten`` source has been accumulated (autoconversion, accretion,
    Bigg freezing, snow riming, graupel riming), and before the saturation
    adjustment's ``pnc_wcd``.  Calling it from inside both the warm and the
    cold network double-applies it: the result stays finite and plausible and
    is wrong, which no per-kernel unit test would catch.

    ``qc_entry`` is the frozen entry cloud mixing ratio, ``qc_after`` the
    current one (WRF's ``qc1d + qcten*dtsave``), and ``density`` the ENTRY
    air density (``module_mp_thompson.F:1802``) -- not a density rediagnosed
    from the mutated temperature and vapour.
    """
    _, size = validate_fields({
        "qc_entry": qc_entry,
        "qc_after": qc_after,
        "nc_entry": nc_entry,
        "density": density,
        "ncten": ncten,
    })
    if _arrays_overlap(ncten, nc_entry):
        raise ValueError(
            "ncten must not alias nc_entry; mp=28 entry state is read-only "
            "for the whole call")
    if _arrays_overlap(ncten, qc_entry) or _arrays_overlap(ncten, qc_after):
        raise ValueError("ncten must not alias the cloud water fields")
    if _arrays_overlap(qc_entry, qc_after):
        raise ValueError(
            "qc_after must not alias qc_entry; the limiter needs both the "
            "entry and the post-source cloud mass")
    _validate_dt(dt)

    from woof.metal.array import MetalArray
    if isinstance(ncten, MetalArray):
        from woof.metal.thompson_aerosol_warm_family import (
            submit_ncten_balance,
        )
        submit_ncten_balance(
            qc_entry, qc_after, nc_entry, density, ncten, dt, size)
        return

    grid, block = launch_grid(size)
    aerosol_kernel(WARM_MODULE, "thompson_aa_ncten_balance")(
        grid, block,
        (qc_entry, qc_after, nc_entry, density, ncten,
         DTYPE(dt), np.int32(size)))




#: Order of the doubles ``thompson_aa_probe_frozen_constants`` writes.  These
#: are WRF's REAL(4) graupel exponents and prefactors; the Fortran oracle
#: prints the same quantities as ``WP07F_*`` so the comparison is against
#: WRF's own digits, never a second transcription.




#: Fields ``probe_warm_frozen_rates`` returns as float32 (WRF REAL) and as
#: float64 (WRF DOUBLE PRECISION), in the order the kernel writes them.




__all__ = [
    "EFFICIENCY_TABLE_SHAPE",
    "RAIN_GRAUPEL_TABLE_NAMES",
    "RAIN_GRAUPEL_TABLE_SHAPE",
    "RAIN_SNOW_TABLE_NAMES",
    "RAIN_SNOW_TABLE_SHAPE",
    "launch_aerosol_warm_source_network",
    "launch_aerosol_warm_source_network_from_owner",
    "launch_ncten_balance",
]
