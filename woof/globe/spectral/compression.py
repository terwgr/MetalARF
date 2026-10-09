"""Progressive quantized spherical-harmonic compression for global fields.

Strictly positive scalar fields may be represented in log space, following the
same anti-ringing rule used by the regional Level-2 operators.  Wind is stored
as quantized vorticity/divergence coefficients rather than two scalar U/V
fits, preserving the vector transform's geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .pins import ACCEPTED_PINS_HASHES, PINS_HASH
from .transform import SphericalHarmonicTransform
from .vector import VorticityDivergenceOperator

SCALAR_SCHEMA = "gpuwm.global-spectral-compressed-scalar/v1"
WIND_SCHEMA = "gpuwm.global-spectral-compressed-wind/v1"
_QUANT_MAX = 32767.0


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _array_hash(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(arr.dtype.str.encode())
    digest.update(str(arr.shape).encode())
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def _weighted_mean(field: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(field.mean(axis=-1) * weights, axis=-1)


def _weighted_mean_square(field: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return _weighted_mean(field * field, weights)


def _relative_l2(error: np.ndarray, reference: np.ndarray, weights) -> np.ndarray:
    numerator = np.sqrt(np.maximum(0.0, _weighted_mean_square(error, weights)))
    denominator = np.sqrt(
        np.maximum(1.0e-300, _weighted_mean_square(reference, weights))
    )
    return numerator / denominator


def _metric_summary(values: np.ndarray) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "shape": list(array.shape),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "values": array.tolist(),
    }


@dataclass(frozen=True)
class QuantizedTriangle:
    q_real: np.ndarray
    q_imag: np.ndarray
    scales: np.ndarray
    truncation: int

    def validate(self) -> None:
        expected = (self.truncation + 1, self.truncation + 1)
        if self.q_real.shape[-2:] != expected:
            raise ValueError("quantized triangle has the wrong spectral shape")
        if self.q_imag.shape != self.q_real.shape:
            raise ValueError("quantized real/imaginary payload shapes differ")
        if self.q_real.dtype != np.int16 or self.q_imag.dtype != np.int16:
            raise ValueError("quantized coefficient payload must be int16")
        if self.scales.shape != self.q_real.shape[:-2] + (self.truncation + 1,):
            raise ValueError("quantized degree scales have the wrong shape")
        if not np.isfinite(self.scales).all() or np.any(self.scales < 0.0):
            raise ValueError("quantized degree scales must be finite and nonnegative")
        for n in range(self.truncation + 1):
            if np.any(self.q_real[..., n, n + 1 :] != 0) or np.any(
                self.q_imag[..., n, n + 1 :] != 0
            ):
                raise ValueError("quantized payload is nonzero outside m<=n")
            if np.any(self.q_imag[..., n, 0] != 0):
                raise ValueError("m=0 coefficients must be real")

    @property
    def payload_bytes(self) -> int:
        return int(self.q_real.nbytes + self.q_imag.nbytes + self.scales.nbytes)


@dataclass(frozen=True)
class CompressedScalarField:
    triangle: QuantizedTriangle
    header: dict[str, object]

    def validate(self) -> None:
        self.triangle.validate()
        _validate_header(self.header, SCALAR_SCHEMA, self.triangle.truncation)
        claimed = self.header.get("payload_sha256")
        base = dict(self.header)
        base.pop("payload_sha256", None)
        if claimed != _payload_hash(base, {"coefficients": self.triangle}):
            raise ValueError("compressed scalar payload hash mismatch")


@dataclass(frozen=True)
class CompressedWindField:
    vorticity: QuantizedTriangle
    divergence: QuantizedTriangle
    header: dict[str, object]

    def validate(self) -> None:
        self.vorticity.validate()
        self.divergence.validate()
        if self.vorticity.q_real.shape != self.divergence.q_real.shape:
            raise ValueError("compressed vorticity/divergence shapes differ")
        _validate_header(self.header, WIND_SCHEMA, self.vorticity.truncation)
        claimed = self.header.get("payload_sha256")
        base = dict(self.header)
        base.pop("payload_sha256", None)
        if claimed != _payload_hash(
            base,
            {"vorticity": self.vorticity, "divergence": self.divergence},
        ):
            raise ValueError("compressed wind payload hash mismatch")


def _validate_header(header: dict, schema: str, truncation: int) -> None:
    if header.get("schema") != schema:
        raise ValueError(f"compressed schema mismatch: {header.get('schema')!r}")
    if header.get("pins_hash") not in ACCEPTED_PINS_HASHES:
        raise ValueError("compressed arithmetic pins do not match this build")
    if int(header.get("truncation", -1)) != truncation:
        raise ValueError("compressed header and payload truncation disagree")


def quantize_coefficients(coefficients) -> QuantizedTriangle:
    host = np.asarray(coefficients, dtype=np.complex128)
    if host.ndim < 2 or host.shape[-1] != host.shape[-2]:
        raise ValueError("coefficients must end in a square triangular array")
    truncation = host.shape[-1] - 1
    q_real = np.zeros(host.shape, dtype=np.int16)
    q_imag = np.zeros(host.shape, dtype=np.int16)
    scales = np.zeros(host.shape[:-2] + (truncation + 1,), dtype=np.float64)
    for n in range(truncation + 1):
        row = host[..., n, : n + 1]
        maximum = np.maximum(
            np.max(np.abs(row.real), axis=-1),
            np.max(np.abs(row.imag), axis=-1),
        )
        # A real field's m=0 harmonic is real.  Zeroing a nonzero imaginary
        # part here would be silent: the discarded value has already entered
        # the degree scale above, so every other coefficient at this degree
        # is coarsened by it and can quantize to zero.
        if np.any(row.imag[..., 0] != 0.0):
            raise ValueError(
                f"degree {n} carries a nonzero m=0 imaginary part; these are "
                "not the coefficients of a real field and quantizing them "
                "would discard that value while inflating the degree scale"
            )
        scale = maximum / _QUANT_MAX
        scales[..., n] = scale
        safe = np.where(scale > 0.0, scale, 1.0)
        q_real[..., n, : n + 1] = np.rint(
            row.real / safe[..., None]
        ).clip(-32767, 32767).astype(np.int16)
        q_imag[..., n, : n + 1] = np.rint(
            row.imag / safe[..., None]
        ).clip(-32767, 32767).astype(np.int16)
    result = QuantizedTriangle(q_real, q_imag, scales, truncation)
    result.validate()
    return result


def dequantize_coefficients(
    triangle: QuantizedTriangle,
    *,
    keep_degree: int | None = None,
) -> np.ndarray:
    triangle.validate()
    keep = triangle.truncation if keep_degree is None else int(keep_degree)
    if not 0 <= keep <= triangle.truncation:
        raise ValueError("keep_degree must lie in 0..stored truncation")
    output = np.zeros(triangle.q_real.shape, dtype=np.complex128)
    for n in range(keep + 1):
        scale = triangle.scales[..., n, None]
        output[..., n, : n + 1] = scale * (
            triangle.q_real[..., n, : n + 1].astype(np.float64)
            + 1j * triangle.q_imag[..., n, : n + 1].astype(np.float64)
        )
    return output


def _triangle_hashes(triangle: QuantizedTriangle) -> dict[str, str]:
    return {
        "q_real_sha256": _array_hash(triangle.q_real),
        "q_imag_sha256": _array_hash(triangle.q_imag),
        "scales_sha256": _array_hash(triangle.scales),
    }


def _payload_hash(header_without_hash: dict, triangles: dict[str, QuantizedTriangle]) -> str:
    body = {
        "header": header_without_hash,
        "triangles": {name: _triangle_hashes(value) for name, value in triangles.items()},
    }
    return hashlib.sha256(_canonical(body)).hexdigest()


def _truncate(coeff: np.ndarray, keep_degree: int) -> np.ndarray:
    result = np.array(coeff, copy=True)
    result[..., keep_degree + 1 :, :] = 0.0
    return result


def _compression_metrics(source: np.ndarray, decoded: np.ndarray, weights) -> dict:
    error = decoded - source
    relative = _relative_l2(error, source, weights)
    scale = np.maximum(1.0e-300, np.max(np.abs(source), axis=(-2, -1)))
    relative_linf = np.max(np.abs(error), axis=(-2, -1)) / scale
    return {
        "relative_l2": _metric_summary(relative),
        "relative_linf": _metric_summary(relative_linf),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "source_minimum": float(np.min(source)),
        "source_maximum": float(np.max(source)),
        "decoded_minimum": float(np.min(decoded)),
        "decoded_maximum": float(np.max(decoded)),
    }


def compress_scalar(
    transform: SphericalHarmonicTransform,
    field,
    *,
    space: str = "linear",
    floor: float = 1.0e-20,
    preserve_mean: bool = True,
    keep_degree: int | None = None,
    source_identity: dict[str, object] | None = None,
) -> CompressedScalarField:
    source = transform.backend.to_numpy(field)
    transform._validate_grid(source)
    source = np.asarray(source)
    if not np.isfinite(source).all():
        raise ValueError("scalar source contains non-finite values")
    mode = str(space).strip().lower()
    if mode not in {"linear", "log"}:
        raise ValueError("space must be 'linear' or 'log'")
    floor = float(floor)
    if mode == "log":
        if not math.isfinite(floor) or floor <= 0.0:
            raise ValueError("log-space floor must be finite and positive")
        if np.any(source < 0.0):
            raise ValueError("log-space compression refuses negative source values")
        analysis_field = np.log(np.maximum(source.astype(np.float64), floor))
        floored = int(np.count_nonzero(source < floor))
    else:
        analysis_field = source.astype(np.float64, copy=False)
        floored = 0
    keep = transform.truncation if keep_degree is None else int(keep_degree)
    if not 0 <= keep <= transform.truncation:
        raise ValueError("keep_degree must lie in 0..transform truncation")
    coefficients = transform.backend.to_numpy(transform.forward(analysis_field))
    coefficients = _truncate(coefficients, keep)
    triangle = quantize_coefficients(coefficients)
    source_mean = _weighted_mean(source.astype(np.float64), transform.grid.quadrature_weights)
    provisional_header: dict[str, object] = {
        "schema": SCALAR_SCHEMA,
        "pins_hash": PINS_HASH,
        "geometry": transform.geometry_identity,
        "geometry_hash": transform.geometry_hash,
        "truncation": transform.truncation,
        "keep_degree": keep,
        "source_shape": list(source.shape),
        "source_dtype": str(source.dtype),
        "source_sha256": _array_hash(source),
        "space": mode,
        "floor": floor if mode == "log" else None,
        "floored_value_count": floored,
        "preserve_mean": bool(preserve_mean),
        "source_mean": np.asarray(source_mean).tolist(),
        "source_identity": dict(source_identity or {}),
        "quantization": {
            "dtype": "int16",
            "maximum_integer_magnitude": int(_QUANT_MAX),
            "scale": "independent maximum real-or-imaginary magnitude per degree",
        },
    }
    result = CompressedScalarField(triangle, provisional_header)
    decoded = decode_scalar(transform, result, validate_payload=False)
    provisional_header["metrics"] = _compression_metrics(
        source.astype(np.float64), decoded, transform.grid.quadrature_weights
    )
    raw_bytes = int(source.nbytes)
    provisional_header["raw_bytes"] = raw_bytes
    provisional_header["coefficient_payload_bytes"] = triangle.payload_bytes
    provisional_header["payload_to_raw_ratio"] = (
        triangle.payload_bytes / raw_bytes if raw_bytes else 0.0
    )
    provisional_header["payload_sha256"] = _payload_hash(
        provisional_header, {"coefficients": triangle}
    )
    result = CompressedScalarField(triangle, provisional_header)
    result.validate()
    return result


def decode_scalar(
    transform: SphericalHarmonicTransform,
    compressed: CompressedScalarField,
    *,
    keep_degree: int | None = None,
    dtype=np.float64,
    validate_payload: bool = True,
) -> np.ndarray:
    if validate_payload:
        compressed.validate()
    if compressed.header.get("geometry_hash") != transform.geometry_hash:
        raise ValueError("compressed scalar geometry does not match transform")
    stored_keep = int(compressed.header["keep_degree"])
    keep = stored_keep if keep_degree is None else int(keep_degree)
    if keep > stored_keep:
        raise ValueError(
            f"requested keep_degree={keep} exceeds stored degree {stored_keep}"
        )
    coefficients = dequantize_coefficients(
        compressed.triangle, keep_degree=keep
    )
    grid = transform.backend.to_numpy(
        transform.inverse(
            transform.backend.asarray(
                coefficients, dtype=transform.backend.complex_dtype
            )
        )
    ).astype(np.float64)
    if compressed.header["space"] == "log":
        grid = np.exp(grid)
    if bool(compressed.header.get("preserve_mean", False)):
        target = np.atleast_1d(
            np.asarray(compressed.header["source_mean"], dtype=np.float64)
        )
        current = np.atleast_1d(
            np.asarray(_weighted_mean(grid, transform.grid.quadrature_weights))
        )
        if compressed.header["space"] == "log":
            # Log space decodes through exp(), so the field is a positive
            # scale and its mean is restored multiplicatively, which keeps
            # positivity.  A level whose decoded mean underflowed to zero
            # carries no scale to correct; multiplying it by target/0 would
            # write inf or nan into an artifact that then hashes clean.
            degenerate = (current <= 0.0) & (target != 0.0)
            if np.any(degenerate):
                raise FloatingPointError(
                    "cannot restore a nonzero mean from a nonpositive decoded "
                    f"mean at leading index {np.flatnonzero(degenerate).tolist()}"
                )
            safe = np.where(current > 0.0, current, 1.0)
            ratio = np.where(current > 0.0, target / safe, 1.0)
            grid *= ratio.reshape(grid.shape[:-2] + (1, 1))
        else:
            # Linear space: the area mean is the n=0 coefficient, an additive
            # offset.  Restoring it multiplicatively divides by a mean that is
            # rounding noise for any zero-mean field -- vorticity, divergence,
            # any anomaly -- and scales the whole field by an arbitrary ratio.
            grid += (target - current).reshape(grid.shape[:-2] + (1, 1))
    return grid.astype(dtype, copy=False)


def compress_wind(
    transform: SphericalHarmonicTransform,
    u,
    v,
    *,
    keep_degree: int | None = None,
    source_identity: dict[str, object] | None = None,
) -> CompressedWindField:
    un = transform.backend.to_numpy(u)
    vn = transform.backend.to_numpy(v)
    transform._validate_grid(un)
    transform._validate_grid(vn)
    if un.shape != vn.shape:
        raise ValueError("wind component shapes differ")
    if not np.isfinite(un).all() or not np.isfinite(vn).all():
        raise ValueError("wind source contains non-finite values")
    keep = transform.truncation if keep_degree is None else int(keep_degree)
    if not 1 <= keep <= transform.truncation:
        raise ValueError("wind keep_degree must lie in 1..transform truncation")
    vector = VorticityDivergenceOperator(transform)
    zeta, divergence = vector.vordiv_from_wind(
        transform.backend.asarray(un, dtype=transform.backend.float_dtype),
        transform.backend.asarray(vn, dtype=transform.backend.float_dtype),
    )
    zeta = _truncate(transform.backend.to_numpy(zeta), keep)
    divergence = _truncate(transform.backend.to_numpy(divergence), keep)
    zq = quantize_coefficients(zeta)
    dq = quantize_coefficients(divergence)
    header: dict[str, object] = {
        "schema": WIND_SCHEMA,
        "pins_hash": PINS_HASH,
        "geometry": transform.geometry_identity,
        "geometry_hash": transform.geometry_hash,
        "truncation": transform.truncation,
        "keep_degree": keep,
        "source_shape": list(un.shape),
        "source_dtype_u": str(un.dtype),
        "source_dtype_v": str(vn.dtype),
        "source_u_sha256": _array_hash(un),
        "source_v_sha256": _array_hash(vn),
        "source_identity": dict(source_identity or {}),
        "representation": "vorticity-divergence-vector-spherical-harmonics",
        "quantization": {
            "dtype": "int16",
            "maximum_integer_magnitude": int(_QUANT_MAX),
            "scale": "independent maximum real-or-imaginary magnitude per degree and carrier",
        },
    }
    provisional = CompressedWindField(zq, dq, header)
    decoded_u, decoded_v = decode_wind(
        transform, provisional, validate_payload=False
    )
    speed_error = np.sqrt((decoded_u - un) ** 2 + (decoded_v - vn) ** 2)
    speed_reference = np.sqrt(un * un + vn * vn)
    header["metrics"] = {
        "u": _compression_metrics(un.astype(np.float64), decoded_u, transform.grid.quadrature_weights),
        "v": _compression_metrics(vn.astype(np.float64), decoded_v, transform.grid.quadrature_weights),
        "wind_vector_relative_l2": _metric_summary(
            np.sqrt(_weighted_mean(speed_error**2, transform.grid.quadrature_weights))
            / np.sqrt(
                np.maximum(
                    1.0e-300,
                    _weighted_mean(speed_reference**2, transform.grid.quadrature_weights),
                )
            )
        ),
        "maximum_vector_error_m_s": float(np.max(speed_error)),
    }
    raw_bytes = int(un.nbytes + vn.nbytes)
    payload_bytes = zq.payload_bytes + dq.payload_bytes
    header["raw_bytes"] = raw_bytes
    header["coefficient_payload_bytes"] = payload_bytes
    header["payload_to_raw_ratio"] = payload_bytes / raw_bytes if raw_bytes else 0.0
    header["payload_sha256"] = _payload_hash(
        header, {"vorticity": zq, "divergence": dq}
    )
    result = CompressedWindField(zq, dq, header)
    result.validate()
    return result


def decode_wind(
    transform: SphericalHarmonicTransform,
    compressed: CompressedWindField,
    *,
    keep_degree: int | None = None,
    dtype=np.float64,
    validate_payload: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if validate_payload:
        compressed.validate()
    if compressed.header.get("geometry_hash") != transform.geometry_hash:
        raise ValueError("compressed wind geometry does not match transform")
    stored_keep = int(compressed.header["keep_degree"])
    keep = stored_keep if keep_degree is None else int(keep_degree)
    if keep > stored_keep:
        raise ValueError(
            f"requested keep_degree={keep} exceeds stored degree {stored_keep}"
        )
    zeta = dequantize_coefficients(compressed.vorticity, keep_degree=keep)
    divergence = dequantize_coefficients(
        compressed.divergence, keep_degree=keep
    )
    vector = VorticityDivergenceOperator(transform)
    u, v = vector.wind_from_vordiv(
        transform.backend.asarray(zeta, dtype=transform.backend.complex_dtype),
        transform.backend.asarray(divergence, dtype=transform.backend.complex_dtype),
    )
    return (
        transform.backend.to_numpy(u).astype(dtype, copy=False),
        transform.backend.to_numpy(v).astype(dtype, copy=False),
    )


def write_compressed_scalar(
    path: str | Path, compressed: CompressedScalarField
) -> Path:
    compressed.validate()
    return _write_archive(
        path,
        compressed.header,
        {
            "q_real": compressed.triangle.q_real,
            "q_imag": compressed.triangle.q_imag,
            "scales": compressed.triangle.scales,
        },
    )


def read_compressed_scalar(path: str | Path) -> CompressedScalarField:
    header, arrays = _read_archive(
        path, {"q_real", "q_imag", "scales"}, SCALAR_SCHEMA
    )
    triangle = QuantizedTriangle(
        np.asarray(arrays["q_real"], dtype=np.int16),
        np.asarray(arrays["q_imag"], dtype=np.int16),
        np.asarray(arrays["scales"], dtype=np.float64),
        int(header["truncation"]),
    )
    result = CompressedScalarField(triangle, header)
    result.validate()
    return result


def write_compressed_wind(path: str | Path, compressed: CompressedWindField) -> Path:
    compressed.validate()
    arrays = {}
    for prefix, triangle in (
        ("zeta", compressed.vorticity),
        ("div", compressed.divergence),
    ):
        arrays[f"{prefix}_q_real"] = triangle.q_real
        arrays[f"{prefix}_q_imag"] = triangle.q_imag
        arrays[f"{prefix}_scales"] = triangle.scales
    return _write_archive(path, compressed.header, arrays)


def read_compressed_wind(path: str | Path) -> CompressedWindField:
    required = {
        "zeta_q_real",
        "zeta_q_imag",
        "zeta_scales",
        "div_q_real",
        "div_q_imag",
        "div_scales",
    }
    header, arrays = _read_archive(path, required, WIND_SCHEMA)
    t = int(header["truncation"])
    zeta = QuantizedTriangle(
        np.asarray(arrays["zeta_q_real"], dtype=np.int16),
        np.asarray(arrays["zeta_q_imag"], dtype=np.int16),
        np.asarray(arrays["zeta_scales"], dtype=np.float64),
        t,
    )
    divergence = QuantizedTriangle(
        np.asarray(arrays["div_q_real"], dtype=np.int16),
        np.asarray(arrays["div_q_imag"], dtype=np.int16),
        np.asarray(arrays["div_scales"], dtype=np.float64),
        t,
    )
    result = CompressedWindField(zeta, divergence, header)
    result.validate()
    return result


def _write_archive(path: str | Path, header: dict, arrays: dict[str, np.ndarray]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.partial-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            __header__=np.asarray(json.dumps(header, sort_keys=True, allow_nan=False)),
            **arrays,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return target


def _read_archive(path: str | Path, required: set[str], schema: str):
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if "__header__" not in archive:
            raise ValueError(f"compressed archive {source} has no header")
        members = set(archive.files) - {"__header__"}
        if members != required:
            raise ValueError(
                f"compressed archive members {sorted(members)} != {sorted(required)}"
            )
        header = json.loads(str(archive["__header__"].item()))
        arrays = {name: np.array(archive[name], copy=True) for name in required}
    if header.get("schema") != schema:
        raise ValueError(f"compressed archive schema mismatch: {header.get('schema')!r}")
    return header, arrays


__all__ = [
    "CompressedScalarField",
    "CompressedWindField",
    "QuantizedTriangle",
    "compress_scalar",
    "compress_wind",
    "decode_scalar",
    "decode_wind",
    "dequantize_coefficients",
    "quantize_coefficients",
    "read_compressed_scalar",
    "read_compressed_wind",
    "write_compressed_scalar",
    "write_compressed_wind",
]
