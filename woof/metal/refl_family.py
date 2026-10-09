"""Metal Kessler and Morrison reflectivity submissions.

The CUDA reflector uses binary64 intermediates and COMPLEX*16 mixing. The
Metal shader carries those values as software-binary64 bit patterns; radar
initialization values are uploaded once per call without downcasting.
"""

from __future__ import annotations

import numpy as np

from .array import MetalArray


# [k_w, m_w.re/im, m_i.re/im, xam_r/s/g, pi, pi5, lambda4, r_d, 0.622,
# qv_floor, mass_threshold, nested_volume_floor, melt_outside, gamma4/7,
# dry_ice_factor, 900, 1000, 1e18, 10, 273.15, then xxds, xdts, xxdg,
# xdtg, and Simpson weights; each bin vector has 50 float64 words.
_TABLE_CONSTANTS = 25
_TABLE_WORDS = _TABLE_CONSTANTS + 5 * 50


def _table_words(morr_rimed_ice: int) -> np.ndarray:
    from woof.core.refl import radar_init

    rc = radar_init(int(morr_rimed_ice))
    constants = np.asarray((
        rc.k_w, rc.m_w_0.real, rc.m_w_0.imag,
        rc.m_i_0.real, rc.m_i_0.imag,
        rc.xam_r, rc.xam_s, rc.xam_g,
        3.1415926535897932384626434, rc.pi5, rc.lamda4,
        287.0, 0.622, 1.0e-10, 1.0e-9,
        1.0e-10, 0.9, 6.0, 720.0, 0.176 / 0.93,
        900.0, 1000.0, 1.0e18, 10.0, 273.15,
    ), dtype=np.float64)
    values = np.concatenate((constants, rc.xxds, rc.xdts, rc.xxdg,
                             rc.xdtg, rc.simpson)).astype(np.float64,
                                                          copy=False)
    if values.shape != (_TABLE_WORDS,) or not np.isfinite(values).all():
        raise ValueError("Morrison radar_init table has an invalid layout")
    return np.ascontiguousarray(values).view(np.uint64)


def submit_refl10cm_morrison(qv, qr, nr, qs, ns, qg, ng, theta, pii,
                              pressure, refl, *, morr_rimed_ice: int = 1,
                              tables: MetalArray | None = None):
    """Submit Morrison's output-due REFL_10CM diagnostic on Metal.

    `theta * pii` is rounded in FP32 before promotion, matching the adapter's
    CUDA scratch construction; pressure is the unchanged prepared field.
    Every species field and output uses C-order `(nz, ny, nx)` FP32 storage.
    """
    arrays = {"qv": qv, "qr": qr, "nr": nr, "qs": qs, "ns": ns,
              "qg": qg, "ng": ng, "theta": theta, "pii": pii,
              "p": pressure, "refl": refl}
    shape = tuple(refl.shape)
    if len(shape) != 3 or shape[0] < 1 or shape[1] < 1 or shape[2] < 1:
        raise ValueError(f"Morrison reflectivity needs a nonempty 3-D field, got {shape}")
    if shape[0] > 256:
        raise ValueError("Morrison reflectivity supports at most 256 levels")
    if int(np.prod(shape, dtype=np.int64)) > 0x7FFFFFFF:
        raise ValueError("Morrison reflectivity size exceeds the int32 ABI")
    if any(not isinstance(value, MetalArray)
           or value.dtype != np.dtype(np.float32)
           or value.shape != shape
           or not value.flags.c_contiguous for value in arrays.values()):
        raise TypeError("Morrison reflectivity fields must be matching contiguous FP32 MetalArrays")
    context = refl.buffer.context
    if any(value.buffer.context is not context for value in arrays.values()):
        raise ValueError("Morrison reflectivity fields must share one Metal context")

    table_words = _table_words(morr_rimed_ice)
    if tables is None:
        tables = MetalArray(context.buffer(table_words.nbytes),
                            table_words.shape, np.uint64)
    elif (not isinstance(tables, MetalArray)
          or tables.dtype != np.dtype(np.uint64)
          or tables.shape != (_TABLE_WORDS,)
          or tables.buffer.context is not context):
        raise TypeError("Morrison radar tables must be a 275-word uint64 MetalArray")
    tables.buffer.write(table_words.tobytes())
    nz, ny, nx = shape
    return context.submit_kernel(
        "refl10cm_morrison_column",
        {**arrays, "tables": tables},
        {"nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=32)


def submit_refl10cm_kessler(qv, qr, t, pressure, refl):
    """Submit the independent-cell Kessler rain-only fallback on Metal."""
    arrays = {"qv": qv, "qr": qr, "t": t, "p": pressure,
              "refl": refl}
    shape = tuple(refl.shape)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError(f"Kessler reflectivity needs a nonempty 3-D field, got {shape}")
    if shape[0] > 256:
        raise ValueError("Kessler reflectivity supports at most 256 levels")
    if int(np.prod(shape, dtype=np.int64)) > 0x7FFFFFFF:
        raise ValueError("Kessler reflectivity size exceeds the int32 ABI")
    if any(not isinstance(value, MetalArray)
           or value.dtype != np.dtype(np.float32)
           or value.shape != shape
           or not value.flags.c_contiguous for value in arrays.values()):
        raise TypeError("Kessler reflectivity fields must be matching contiguous FP32 MetalArrays")
    context = refl.buffer.context
    if any(value.buffer.context is not context for value in arrays.values()):
        raise ValueError("Kessler reflectivity fields must share one Metal context")
    nz, ny, nx = shape
    return context.submit_kernel(
        "refl10cm_kessler_cell", arrays,
        {"nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=nz * ny * nx, threads_per_group=256)


def submit_refl_temperature_product(theta, pii, temperature):
    """Materialize WRF's FP32 ``theta * exner`` temperature scratch."""
    arrays = {"theta": theta, "pii": pii, "temperature": temperature}
    shape = tuple(temperature.shape)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError(f"temperature product needs a nonempty 3-D field, got {shape}")
    if any(not isinstance(value, MetalArray)
           or value.dtype != np.dtype(np.float32)
           or value.shape != shape
           or not value.flags.c_contiguous for value in arrays.values()):
        raise TypeError("temperature product fields must be matching contiguous FP32 MetalArrays")
    context = temperature.buffer.context
    if any(value.buffer.context is not context for value in arrays.values()):
        raise ValueError("temperature product fields must share one Metal context")
    nz, ny, nx = shape
    return context.submit_kernel(
        "refl_temperature_product", arrays,
        {"nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=nz * ny * nx, threads_per_group=256)


def submit_refl10cm_thompson(*args, **kwargs):
    raise NotImplementedError("Metal Thompson reflectivity remains unsupported")


def submit_refl10cm_wsm6(*args, **kwargs):
    raise NotImplementedError("Metal WSM6 reflectivity remains unsupported")
