"""Metal boundaries for Thompson cold-start initialization.

``cold_start_alt`` performs IEEE RN binary64-to-binary32 conversion using
integer shifts. ``cold_start_surface`` classifies the aerosol branch through
the shared integer-bit binary64 arithmetic. The number and temperature
exports remain fail-closed until their pinned libm and closure-receipt
requirements have faithful Metal support.
"""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray, MetalArrayModule


def submit_cold_start_alt(input_f64: MetalArray, output_f32: MetalArray,
                          n: int | None = None):
    """Round the first ``n`` binary64 storage values to float32, RN-even.

    The shader never executes FP64 arithmetic. It reads a float32 view over
    the binary64 bytes and reconstructs each input from two uint32 words.
    """
    if (not isinstance(input_f64, MetalArray)
            or input_f64.dtype != np.dtype(np.float64)
            or not input_f64.flags.c_contiguous):
        raise TypeError("cold_start_alt input must be contiguous float64 storage")
    if (not isinstance(output_f32, MetalArray)
            or output_f32.dtype != np.dtype(np.float32)
            or not output_f32.flags.c_contiguous):
        raise TypeError("cold_start_alt output must be contiguous float32 storage")
    if input_f64.buffer.context is not output_f32.buffer.context:
        raise ValueError("cold_start_alt arrays must belong to one Metal context")
    n = input_f64.size if n is None else int(n)
    if n <= 0:
        return None
    if n > 0x7FFFFFFF // 2:
        raise ValueError("cold_start_alt count exceeds int32 word-count capacity")
    if input_f64.size < n or output_f32.size < n:
        raise ValueError("cold_start_alt buffers must contain at least n values")
    input_lo = input_f64.byte_offset
    input_hi = input_lo + n * np.dtype(np.float64).itemsize
    output_lo = output_f32.byte_offset
    output_hi = output_lo + n * np.dtype(np.float32).itemsize
    if (input_f64.buffer.handle == output_f32.buffer.handle
            and input_lo < output_hi and output_lo < input_hi):
        raise ValueError("cold_start_alt does not support overlapping buffers")

    words = input_f64.view((2 * n,), dtype=np.float32,
                           byte_offset=input_f64.byte_offset)
    output = output_f32.view((n,), byte_offset=output_f32.byte_offset)
    return input_f64.buffer.context.submit_kernel(
        "cold_start_alt", {"input_words": words, "output": output},
        {"n": n, "word_count": 2 * n}, exact_policy=False,
        grid_size=n)


def submit_cold_start_surface(alt: MetalArray, aerosol: MetalArray,
                              surface: MetalArray, n: int):
    """Classify aerosol surface branch on Metal with the CUDA operation order.

    ``surface`` is int32 storage carrying the CUDA byte codes (0/1/2). This
    avoids a byte resource while preserving the values consumed by the host
    caller. The kernel computes ``cs_mul(aerosol, cs_div(1, alt))`` through
    integer-bit RN binary64 helpers and explicitly preserves float32
    subnormals.
    """
    if n < 0 or n > 0x7fffffff:
        raise ValueError("cold_start_surface count exceeds int32 capacity")
    arrays = {"alt": alt, "aerosol": aerosol, "surface": surface}
    for name in ("alt", "aerosol"):
        value = arrays[name]
        if (not isinstance(value, MetalArray)
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous or value.size < n):
            raise TypeError(f"cold_start_surface {name} must be contiguous FP32 storage with n values")
    if (not isinstance(surface, MetalArray)
            or surface.dtype != np.dtype(np.int32)
            or not surface.flags.c_contiguous or surface.size < n):
        raise TypeError("cold_start_surface output must be contiguous int32 storage with n values")
    if any(value.buffer.context is not alt.buffer.context for value in arrays.values()):
        raise ValueError("cold_start_surface arrays must belong to one Metal context")
    if n == 0:
        return None
    return alt.buffer.context.submit_kernel(
        "cold_start_surface", arrays, {"n": n}, exact_policy=False,
        grid_size=n)


def submit_cold_start_temperature(theta: MetalArray, pressure: MetalArray,
                                  indices: MetalArray, temperature: MetalArray,
                                  n: int, thermo_constants: MetalArray):
    """Evaluate WRF's f64 temperature formula at gathered seed indices."""
    if n < 0 or n > 0x7fffffff:
        raise ValueError("cold_start_temperature count exceeds int32 capacity")
    arrays = {"theta": theta, "pressure": pressure, "indices": indices,
              "thermo_constants": thermo_constants, "temperature": temperature}
    for name in ("theta", "pressure", "thermo_constants"):
        value = arrays[name]
        if (not isinstance(value, MetalArray)
                or value.dtype != np.dtype(np.float64)
                or not value.flags.c_contiguous):
            raise TypeError(f"cold_start_temperature {name} must be contiguous float64 storage")
    if theta.shape != pressure.shape or theta.ndim != 1:
        raise ValueError("cold_start_temperature theta/pressure must be equal-length vectors")
    if thermo_constants.shape != (2,):
        raise ValueError("cold_start_temperature needs binary64 [P0, RCP] constants")
    if (not isinstance(indices, MetalArray)
            or indices.dtype != np.dtype(np.int64)
            or not indices.flags.c_contiguous or indices.size < n):
        raise TypeError("cold_start_temperature indices must be contiguous int64 storage with n values")
    if (not isinstance(temperature, MetalArray)
            or temperature.dtype != np.dtype(np.float32)
            or not temperature.flags.c_contiguous or temperature.size < n):
        raise TypeError("cold_start_temperature output must be contiguous FP32 storage with n values")
    if any(value.buffer.context is not theta.buffer.context for value in arrays.values()):
        raise ValueError("cold_start_temperature arrays must belong to one Metal context")
    if n == 0:
        return None
    return theta.buffer.context.submit_kernel(
        "cold_start_temperature", arrays,
        {"n": n, "n_total": theta.size}, exact_policy=False,
        grid_size=n)


def submit_cold_start_seed_compact(mass: MetalArray, number: MetalArray,
                                   base: int = 0):
    """Compact exact WRF seed predicates into a device int64 index array.

    Only the scalar count is read back; the selected indices stay on Metal
    and can feed the gathered temperature kernel directly.
    """
    if (not isinstance(mass, MetalArray) or not isinstance(number, MetalArray)
            or mass.dtype != np.dtype(np.float32)
            or number.dtype != np.dtype(np.float32)
            or mass.shape != number.shape or mass.ndim != 1
            or not mass.flags.c_contiguous or not number.flags.c_contiguous):
        raise TypeError("cold_start_seed_compact requires equal-length contiguous FP32 vectors")
    if mass.buffer.context is not number.buffer.context:
        raise ValueError("cold_start_seed_compact arrays must belong to one Metal context")
    if base < 0 or base + mass.size > 0x7fffffff:
        raise ValueError("cold_start_seed_compact global indices exceed int32 kernel capacity")
    module = MetalArrayModule(mass.buffer.context)
    indices = module.zeros((mass.size,), np.int64)
    count = module.zeros((1,), np.int32)
    if mass.size:
        mass.buffer.context.submit_kernel(
            "cold_start_seed_compact",
            {"mass": mass, "number": number, "indices": indices,
             "count": count},
            {"n": mass.size, "base": base}, exact_policy=False,
            grid_size=mass.size)
    actual = int(count.to_numpy()[0])
    return indices.view((actual,)), actual


def submit_cold_start_temperature_scatter(indices: MetalArray,
                                          temperature: MetalArray,
                                          full_temperature: MetalArray,
                                          n_total: int):
    """Scatter sparse seed temperatures into a device lookup field."""
    if (not isinstance(indices, MetalArray)
            or indices.dtype != np.dtype(np.int64)
            or not indices.flags.c_contiguous
            or not isinstance(temperature, MetalArray)
            or temperature.dtype != np.dtype(np.float32)
            or not temperature.flags.c_contiguous
            or not isinstance(full_temperature, MetalArray)
            or full_temperature.dtype != np.dtype(np.float32)
            or not full_temperature.flags.c_contiguous
            or indices.size != temperature.size
            or full_temperature.size != n_total):
        raise TypeError("cold_start_temperature_scatter received incompatible contiguous arrays")
    if any(value.buffer.context is not full_temperature.buffer.context
           for value in (indices, temperature)):
        raise ValueError("cold_start_temperature_scatter arrays must belong to one Metal context")
    if indices.size == 0:
        return None
    return full_temperature.buffer.context.submit_kernel(
        "cold_start_temperature_scatter",
        {"indices": indices, "temperature": temperature,
         "full_temperature": full_temperature},
        {"n": indices.size, "n_total": n_total},
        exact_policy=False, grid_size=indices.size)


def submit_cold_start_numbers(
    mass: MetalArray, number: MetalArray, alt: MetalArray,
    temperature: MetalArray, aerosol: MetalArray, xland: MetalArray,
    retab: MetalArray, gratio: MetalArray, constants_f64: MetalArray,
    final_number: MetalArray, seed_volume: MetalArray, n: int, species: int,
):
    """Submit a faithful soft64/soft-pow cold-start number kernel."""
    if n < 0 or n > 0x7fffffff:
        raise ValueError("cold_start_numbers count exceeds int32 capacity")
    if species not in (0, 1, 2):
        raise ValueError("cold_start_numbers species must be rain=0, ice=1, cloud=2")
    arrays = {"mass": mass, "number": number, "alt": alt,
              "temperature": temperature, "aerosol": aerosol, "xland": xland,
              "retab": retab, "gratio": gratio, "constants": constants_f64,
              "final_number": final_number, "seed_volume": seed_volume,
              "seeded": None, "offender": None}
    vector_names = ("mass", "number", "alt", "temperature", "aerosol", "xland",
                    "final_number", "seed_volume")
    for name in vector_names:
        value = arrays[name]
        if (not isinstance(value, MetalArray)
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous or value.size < n):
            raise TypeError(f"cold_start_numbers {name} must be contiguous FP32 storage with n values")
    if (not isinstance(retab, MetalArray) or retab.dtype != np.dtype(np.float32)
            or retab.shape != (95,) or not retab.flags.c_contiguous):
        raise TypeError("cold_start_numbers retab must be a contiguous 95-entry FP32 table")
    if (not isinstance(gratio, MetalArray) or gratio.dtype != np.dtype(np.float32)
            or gratio.shape != (15,) or not gratio.flags.c_contiguous):
        raise TypeError("cold_start_numbers gratio must be a contiguous 15-entry FP32 table")
    if (not isinstance(constants_f64, MetalArray)
            or constants_f64.dtype != np.dtype(np.float64)
            or constants_f64.shape != (21,) or not constants_f64.flags.c_contiguous):
        raise TypeError("cold_start_numbers constants must be a contiguous 21-entry float64 vector")
    # Diagnostic/validation outputs are caller-owned to make their readback
    # policy explicit and keep this kernel free of host indexing or compaction.
    module = MetalArrayModule(mass.buffer.context)
    arrays["seeded"] = module.zeros((n,), np.int32)
    arrays["offender"] = module.zeros((n,), np.int32)
    if any(value.buffer.context is not mass.buffer.context for value in arrays.values()):
        raise ValueError("cold_start_numbers arrays must belong to one Metal context")
    for name in vector_names[:6]:
        if arrays[name].size < n:
            raise ValueError(f"cold_start_numbers {name} is smaller than n")
    if n == 0:
        return arrays["seeded"], arrays["offender"], None
    submission = mass.buffer.context.submit_kernel(
        "cold_start_numbers", arrays,
        {"n": n, "species": species, "retab_n": retab.size,
         "gratio_n": gratio.size},
        exact_policy=False, grid_size=n)
    return arrays["seeded"], arrays["offender"], submission
