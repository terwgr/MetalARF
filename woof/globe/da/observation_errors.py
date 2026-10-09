"""The observation-error calibration the filter lays over the rows' assigned errors.

The observation doors assign each row an error when they decode it (a
METAR 2 m temperature 1.5 K, a station pressure 100 Pa, a motion vector
4 m/s, a refractivity row the Kuo et al. (2004) fraction of its own value
by tangent height, ...).  Those are the doors' stated instrument errors
and they are kept in the tables; what the filter needs is the error of
the report AS THE FILTER SEES IT, instrument plus representativeness
against the member's grid, and Desroziers et al. (2005) estimates that
from the record: ``E[d_oa d_ob] = R`` on a stream whose gain was near the
optimal one.  :data:`DESROZIERS_ERROR_TABLE` is that estimate read off the
six hourly analyses of the final grade of 2026-09-06 (the GDAS 2026-09-01
18Z start cycled to 2026-09-02 00Z at T255 over 32 T127 members, the
control's card, global region, every cycle with at least 100 rows in the
cell; the value is the root of the cycle-mean estimated error variance,
rounded to the assigned error's precision).  The surface classes are the
record's directly; the aloft classes are the same record's reading and
are stated here with it, so a reader sees which rows the filter weights
differently from the doors and by how much.

A stream whose assigned error is a PROFILE (one value per row, the
refractivity fraction of the row's own value) is calibrated by a SCALE
laid over the row's own error instead of a constant, in
:data:`DESROZIERS_ERROR_SCALE_TABLE`: the root of the cycle-mean ratio of
the Desroziers-estimated error variance to the assigned one, read off the
six analyses of the completed system's grade of record (2026-09-07, the
same start, the semi-Lagrangian control, the hybrid at beta 0.75): 1.97,
1.85, 1.91, 2.01, 2.02, 2.02 on 2,445 to 3,493 rows a cycle, mean 1.965,
root 1.40.  Before the scale, 1,389 to 2,208 of the 8,000 to 11,000
refractivity rows offered a cycle (16 to 20 percent) failed the 4-sigma
background check against an error half the diagnosed one, and the streams
lane's own six cycles read the same shape too small by 1.67 to 1.89.

What the calibration changed against the doors' values (assigned to
estimated): METAR temperature 1.5 to 1.75 K, dewpoint 1.5 to 1.9 K,
station pressure 100 to 80 Pa, wind 2.5 to 2.3 m/s; NDBC temperature 1.5
to 1.95 K, dewpoint 2.0 to 1.85 K, pressure 100 to 85 Pa, wind 2.9 to 2.55
m/s; IGRA2 temperature 1.0 to 1.35 K, wind 2.5 to 3.1 m/s, dewpoint 2.5
to 3.4 K, surface pressure 100 to 90 Pa; GOES derived-motion winds 4.0 to
2.8 m/s (the innovation rms itself, 2.86 m/s, bounds the sum of the
observation and background errors, so 4.0 m/s could not have been the
observation's); radio-occultation refractivity on both routes
(``cdaac-ro`` live, ``gnss-ro`` retrospective) the Kuo fraction times
1.4.

The table is applied by name (``FilterOptions.observation_error_calibration``)
before quality control, so the background check and the solve see one
error; a (stream, variable) neither table names keeps the row's own
error, and the receipt records the assigned error, the calibrated
constant or scale, and the calibrated error's rms per stream and
variable.
"""
from __future__ import annotations

import numpy as np

#: (stream, variable) -> observation error standard deviation in the
#: variable's units, the Desroziers estimate of the 2026-09-06 record.
DESROZIERS_ERROR_TABLE: dict[tuple[str, str], float] = {
    ("iem-metar", "temperature_k"): 1.75,
    ("iem-metar", "dewpoint_k"): 1.9,
    ("iem-metar", "surface_pressure_pa"): 80.0,
    ("iem-metar", "wind_u_m_s"): 2.3,
    ("iem-metar", "wind_v_m_s"): 2.3,
    ("ndbc", "temperature_k"): 1.95,
    ("ndbc", "dewpoint_k"): 1.85,
    ("ndbc", "surface_pressure_pa"): 85.0,
    ("ndbc", "wind_u_m_s"): 2.55,
    ("ndbc", "wind_v_m_s"): 2.55,
    ("igra2", "temperature_k"): 1.35,
    ("igra2", "dewpoint_k"): 3.4,
    ("igra2", "surface_pressure_pa"): 90.0,
    ("igra2", "wind_u_m_s"): 3.1,
    ("igra2", "wind_v_m_s"): 3.1,
    ("goes-dmw", "wind_u_m_s"): 2.8,
    ("goes-dmw", "wind_v_m_s"): 2.8,
}

#: (stream, variable) -> the factor laid over each row's OWN assigned
#: error for the streams whose error is a profile: the root of the
#: cycle-mean Desroziers error-variance ratio of the completed system's
#: grade of record (2026-09-07; six analyses, 1.97 to 2.02, mean 1.965).
DESROZIERS_ERROR_SCALE_TABLE: dict[tuple[str, str], float] = {
    ("cdaac-ro", "refractivity_n"): 1.4,
    ("gnss-ro", "refractivity_n"): 1.4,
}

CALIBRATIONS = {
    "desroziers-2026-09-06": {
        "table": DESROZIERS_ERROR_TABLE,
        "scales": DESROZIERS_ERROR_SCALE_TABLE,
        "record": (
            "the final grade of 2026-09-06: woof global da fresh from the GDAS 2026-09-01 18Z analysis, "
            "six hourly letkf analyses to 2026-09-02 00Z at T255 over 32 T127 members, the control's card, "
            "global region; E[d_oa d_ob] per stream and variable, cycle mean over the cycles with at least "
            "100 rows, rooted and rounded; the refractivity scale from the completed system's grade of "
            "record of 2026-09-07 (the same start on the semi-Lagrangian control under the hybrid at beta "
            "0.75), the root of the cycle-mean ratio of the estimated to the assigned error variance over "
            "its six analyses"
        ),
    },
}


def calibrated_error(name: str | None, stream: str, variable: str,
                     assigned: np.ndarray) -> tuple[np.ndarray, float | dict | None]:
    """The error array a batch carries under calibration ``name``: the
    table's constant for ``(stream, variable)`` broadcast over the rows,
    the scale table's factor times each row's own assigned error for a
    profile-error cell, or the assigned errors when neither table names
    the cell (or ``name`` is None).  Returns ``(errors, table_entry)``
    with the entry a float for a constant, ``{"scale": factor}`` for a
    scale and ``None`` when the rows keep their own errors."""
    assigned = np.asarray(assigned, dtype=np.float64)
    if name is None:
        return assigned, None
    try:
        calibration = CALIBRATIONS[name]
    except KeyError as exc:
        raise ValueError(f"unknown observation-error calibration {name!r}; this tree carries {sorted(CALIBRATIONS)}") from exc
    key = (str(stream), str(variable))
    value = calibration["table"].get(key)
    if value is not None:
        return np.full(assigned.shape, float(value), dtype=np.float64), float(value)
    scale = calibration.get("scales", {}).get(key)
    if scale is not None:
        return assigned * float(scale), {"scale": float(scale)}
    return assigned, None


__all__ = ["CALIBRATIONS", "DESROZIERS_ERROR_SCALE_TABLE", "DESROZIERS_ERROR_TABLE", "calibrated_error"]
