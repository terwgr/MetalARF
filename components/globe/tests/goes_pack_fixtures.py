"""Build ``GPWMGOES`` packs in memory, for tests that need one.

An independent transcription of the framing in
``tools/rustwx/crates/rw-goes/src/pack.rs`` -- deliberately written from
the Rust writer's own field list rather than from
:mod:`woof.obs.goes_pack`, so that a reader test is a test of two
independent readings of one contract and not of a module agreeing with
itself.

WHAT IS NOT CLOSED HERE.  No test in this tree hands a pack this module
wrote to the Rust ``rw_goes`` reader, so the framing is judged by two
independent Python readings of one contract and by nothing else.  A cross
check against the Rust decoder is open work, named here rather than implied
by a reference to a test that does not exist.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MAGIC = b"GPWMGOES"
VERSION = 1
HEADER_BYTES = 64

CWP_SCHEMA = "gpuwm-obs.goes-cwp.v1"
CWP_SCHEMA_V2 = "gpuwm-obs.goes-cwp.v2"
CLOUDTOP_SCHEMA = "gpuwm-obs.goes-cloudtop.v1"
CLOUDTOP_SCHEMA_V2 = "gpuwm-obs.goes-cloudtop.v2"

#: GOES-19 CONUS, as the real granules carry it.
PROJECTION = {
    "perspective_point_height_m": 35786023.0,
    "semi_major_axis_m": 6378137.0,
    "semi_minor_axis_m": 6356752.31414,
    "longitude_of_projection_origin_deg": -75.0,
    "sweep_angle_axis": "x",
}

COEFFICIENTS = {
    "formula": "CWP [g m-2] = (2/3) * rho(phase) * tau * r_e",
    "liquid_density_g_cm3": 1.0,
    "ice_density_g_cm3": 0.917,
    "ice_coefficient_provisional": True,
    "mixed_phase_takes_ice_branch_provisional": True,
    "clear_sky_emits_zero": True,
}

#: The bridge corrected this string on 2026-08-06: it used to name
#: ``gpuwm-obs.goes-cwp.v1`` literally, which shipped a wrong schema name
#: inside v2 cloud-top packs.  It now names ``pairs_with_schema`` instead,
#: carrying no version.  Nothing in this reader parses it -- it is prose
#: for humans -- but the fixture tracks the real string so a test never
#: asserts against a sentence the bridge has stopped writing.
NO_REGRID = (
    "none: planes are on the granules' own fixed grid, bit-identical; any "
    "join to the pairs_with_schema sibling's grid is the consumer's "
    "explicit choice")


def source(product: str, *, condemn_mask: int | None = 88,
           total: int = 100, finite: int = 90,
           dqf_plane: str | None = None) -> dict:
    """One ``SourceEntry`` row.

    ``condemn_mask`` defaults to 88 (snow/sea-ice 8 | twilight 16 |
    glint 64), which is what the live packs actually carry -- verified
    against g19_conus_20260804_1801.  Bits 256/512 are outside it by
    design, which is why the operator inflates them rather than gating.

    ``dqf_plane`` names this product's per-pixel DQF plane (v2 only).
    """

    return {
        "product": product,
        "filename": f"OR_ABI-L2-{product}C-M6_G19_s20262161801170.nc",
        "bytes": 1234567,
        "sha256": hashlib.sha256(product.encode()).hexdigest(),
        "dqf_rule": "bitfield" if condemn_mask is not None else "enumerated",
        **({"condemn_mask": condemn_mask} if condemn_mask is not None else {}),
        **({"dqf_plane": dqf_plane} if dqf_plane is not None else {}),
        "dqf": {"total": total, "primary_missing": 3, "dqf_missing": 2,
                "dqf_bad": 4, "masked": total - finite, "finite": finite},
    }


def derive_cwp(cod, cps, phase, *, coefficients=None) -> np.ndarray:
    """The bridge's own CWP derivation, in the bridge's operand order.

    ``rw_sat::cwp::cloud_water_path_g_m2``: clear sky is a genuine 0.0,
    unknown phase and missing or negative inputs are NaN, and the product
    is formed as ``((2/3 * rho) * tau) * r_e`` in float32.
    """

    coefficients = COEFFICIENTS if coefficients is None else coefficients
    cod = np.asarray(cod, dtype=np.float32)
    cps = np.asarray(cps, dtype=np.float32)
    phase = np.asarray(phase, dtype=np.float32)
    out = np.full(cod.shape, np.nan, dtype=np.float32)
    two_thirds = np.float32(2.0) / np.float32(3.0)
    usable = (np.isfinite(phase) & (phase >= 0.0) & (phase <= 5.0)
              & (np.floor(phase) == phase))
    codes = np.where(usable, phase, -1.0).astype(np.int64)
    inputs_ok = (np.isfinite(cod) & np.isfinite(cps)
                 & (cod >= np.float32(0.0)) & (cps >= np.float32(0.0)))
    out[codes == 0] = np.float32(0.0)
    for code, key in ((1, "liquid_density_g_cm3"),
                      (2, "liquid_density_g_cm3"),
                      (3, "ice_density_g_cm3"),
                      (4, "ice_density_g_cm3")):
        rho = np.float32(coefficients[key])
        where = (codes == code) & inputs_ok
        if np.any(where):
            out[where] = ((two_thirds * rho) * cod[where]) * cps[where]
    return out


def _encode(meta: dict, payload: bytes) -> bytes:
    meta_json = json.dumps(meta).encode("utf-8")
    header = bytearray(HEADER_BYTES)
    header[0:8] = MAGIC
    header[8:12] = np.uint32(VERSION).tobytes()
    header[12:16] = np.uint32(len(meta_json)).tobytes()
    header[16:24] = np.uint64(len(payload)).tobytes()
    return bytes(header) + meta_json + payload


def _pack_planes(named):
    """``(payload, planes, plane_order, arrays)`` from ordered planes."""

    payload = bytearray()
    planes: dict[str, str] = {}
    order: list[str] = []
    arrays: dict[str, dict] = {}
    for index, (name, values) in enumerate(named):
        values = np.ascontiguousarray(values, dtype="<f4")
        key = f"a{index:05d}"
        offset = len(payload)
        payload.extend(values.tobytes(order="C"))
        planes[name] = key
        order.append(name)
        arrays[key] = {"dtype": "<f4", "shape": list(values.shape),
                       "offset": offset, "bytes": values.nbytes}
    return bytes(payload), planes, order, arrays


BT_SCHEMA_V1 = "gpuwm-obs.goes-bt.v1"


def write_bt_pack(path, *, bt, rad, lat, lon, bcm=None, cmip_bt=None, band=13,
                  satellite="G19", sector="F", schema=BT_SCHEMA_V1,
                  planck=None, scan_start="2026-09-01T18:00:20.300Z",
                  scan_end="2026-09-01T18:09:52.300Z", x_scan_rad=None, y_scan_rad=None,
                  provenance=None):
    """A ``gpuwm-obs.goes-bt.v1`` pack the way ``rw_goes bt`` writes one:
    bt, rad, lat, lon, then bcm and cmip_bt when given, then one ``_dqf``
    plane per source, with the Planck row and the L1b DQF policy in the
    metadata.  The default scan angles put the planes on the ABI 2 km
    lattice (``(index + 1/2) * 56 urad``) starting at the north-west of
    the full disk."""
    bt = np.asarray(bt, dtype="<f4")
    ny, nx = bt.shape
    pitch = 56.0e-6
    if x_scan_rad is None:
        x_scan_rad = [(-2712 + i + 0.5) * pitch for i in range(nx)]
    if y_scan_rad is None:
        y_scan_rad = [(2711 - j + 0.5) * pitch for j in range(ny)]
    named = [("bt", bt), ("rad", rad), ("lat", lat), ("lon", lon)]
    sources = [{
        "product": f"RAD{band:02d}", "filename": f"OR_ABI-L1b-RadF-M6C{band:02d}_{satellite}_s0_e0_c0.nc",
        "bytes": 1, "sha256": "0" * 64, "dqf_rule": "enumerated", "dqf": {
            "total": int(bt.size), "primary_missing": 0, "dqf_missing": 0, "dqf_bad": 0,
            "masked": 0, "finite": int(np.isfinite(bt).sum())}, "dqf_plane": "rad_dqf",
    }]
    dqf_planes = [("rad_dqf", np.zeros_like(bt))]
    if bcm is not None:
        named.append(("bcm", bcm))
        sources.append({**sources[0], "product": "ACM", "dqf_plane": "acm_dqf",
                        "filename": f"OR_ABI-L2-ACMF-M6_{satellite}_s0_e0_c0.nc"})
        dqf_planes.append(("acm_dqf", np.zeros_like(bt)))
    if cmip_bt is not None:
        named.append(("cmip_bt", cmip_bt))
        sources.append({**sources[0], "product": f"CMIP{band:02d}", "dqf_plane": f"cmip{band:02d}_dqf",
                        "filename": f"OR_ABI-L2-CMIPF-M6C{band:02d}_{satellite}_s0_e0_c0.nc"})
        dqf_planes.append((f"cmip{band:02d}_dqf", np.zeros_like(bt)))
    named.extend(dqf_planes)
    payload, planes, order, arrays = _pack_planes(named)
    meta = {
        "schema": schema, "status": "READY", "satellite": satellite, "sector": sector, "band": band,
        "scan_start": scan_start, "scan_end": scan_end, "sources": sources,
        "projection": {"perspective_point_height_m": 35786023.0, "semi_major_axis_m": 6378137.0,
                       "semi_minor_axis_m": 6356752.31414, "longitude_of_projection_origin_deg": -75.0,
                       "sweep_angle_axis": "x"},
        "nx": nx, "ny": ny, "x_scan_rad": list(x_scan_rad), "y_scan_rad": list(y_scan_rad),
        "planes": planes, "plane_order": order, "arrays": arrays, "payload_bytes": len(payload),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "planck": planck or {"fk1": 10860.4, "fk2": 1395.19, "bc1": 0.07481, "bc2": 0.99975},
        "brightness_temperature_formula": "T[K] = (planck_fk2 / ln(planck_fk1 / Rad + 1) - planck_bc1) / planck_bc2",
        "dqf_policy": "L1b DQF enumerated: 0 good keeps the pixel",
        "counts": {"total": int(bt.size), "rad_missing": 0, "rad_nonpositive": 0, "dqf_missing": 0,
                   "dqf_bad": 0, "finite": int(np.isfinite(bt).sum())},
    }
    if provenance is not None:
        # the bookkeeping row rw_goes bt writes since 2026-09-06 (measurement, publication, receipt, identity)
        meta["provenance"] = dict(provenance)
    Path(path).write_bytes(_encode(meta, payload))
    return Path(path)


def write_cwp_pack(path, *, cod, cps, phase, lat, lon, cwp=None,
                   satellite="G19", sector="C",
                   scan_start="2026-08-04T18:01:17.0Z",
                   scan_end="2026-08-04T18:06:17.0Z",
                   x_scan_rad=None, y_scan_rad=None,
                   projection=None, coefficients=None, status="READY",
                   schema=CWP_SCHEMA, window=None, sources=None,
                   cloud_top_height_m=None, dqf_planes=None) -> Path:
    """Write one CWP pack, v1 or v2.

    ``dqf_planes`` is ``{product: (ny, nx) array}``; supplying it appends
    the per-pixel DQF planes AFTER lat/lon (v2 layout, indices 0-5
    unmoved) and names each in its source row's ``dqf_plane``.
    """

    cod = np.asarray(cod, dtype=np.float32)
    ny, nx = cod.shape
    coefficients = COEFFICIENTS if coefficients is None else coefficients
    if cwp is None:
        cwp = derive_cwp(cod, cps, phase, coefficients=coefficients)
    named = [("cwp", cwp), ("phase", phase), ("cod", cod), ("cps", cps)]
    if cloud_top_height_m is not None:
        named.append(("cloud_top_height_m", cloud_top_height_m))
    named += [("lat", lat), ("lon", lon)]
    if dqf_planes:
        for product in ("COD", "CPS", "ACTP"):
            if product in dqf_planes:
                named.append((f"{product.lower()}_dqf",
                              dqf_planes[product]))
    payload, planes, order, arrays = _pack_planes(named)
    if sources is None:
        def _plane_name(product):
            return (f"{product.lower()}_dqf"
                    if dqf_planes and product in dqf_planes else None)
        sources = [source("COD", dqf_plane=_plane_name("COD")),
                   source("CPS", dqf_plane=_plane_name("CPS")),
                   source("ACTP", condemn_mask=None,
                          dqf_plane=_plane_name("ACTP"))]
    meta = {
        "schema": schema,
        "status": status,
        "satellite": satellite,
        "sector": sector,
        "scan_start": scan_start,
        "scan_end": scan_end,
        "sources": sources,
        "projection": dict(PROJECTION if projection is None else projection),
        "nx": int(nx),
        "ny": int(ny),
        "x_scan_rad": (list(np.linspace(-0.05, 0.05, nx)) if x_scan_rad is None
                       else list(map(float, x_scan_rad))),
        "y_scan_rad": (list(np.linspace(0.10, 0.06, ny)) if y_scan_rad is None
                       else list(map(float, y_scan_rad))),
        "planes": planes,
        "plane_order": order,
        "arrays": arrays,
        "payload_bytes": len(payload),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "cwp_counts": {"clear_zero": 1, "liquid": 1, "supercooled": 0,
                       "mixed": 0, "ice": 1, "unknown": 0,
                       "phase_missing": 0, "input_missing": 0, "finite": 3},
        "coefficients": dict(coefficients),
    }
    if window is not None:
        meta["window"] = list(window)
    path = Path(path)
    path.write_bytes(_encode(meta, payload))
    return path


def write_cloudtop_pack(path, *, cloud_top_height_m, lat, lon,
                        cloud_top_pressure_hpa=None,
                        satellite="G19", sector="C",
                        scan_start="2026-08-04T18:01:17.0Z",
                        scan_end="2026-08-04T18:06:17.0Z",
                        x_scan_rad=None, y_scan_rad=None,
                        projection=None, status="READY",
                        schema=CLOUDTOP_SCHEMA, window=None,
                        sibling=None, dqf_planes=None) -> Path:
    """Write one cloud-top pack, v1 or v2.

    ``dqf_planes`` is ``{product: array}`` for ACHA/CTP; supplying it
    appends the per-pixel DQF planes after lat/lon and names each in its
    source row, which a v2 pack must carry.
    """

    heights = np.asarray(cloud_top_height_m, dtype=np.float32)
    ny, nx = heights.shape
    products = ["ACHA"] + (["CTP"] if cloud_top_pressure_hpa is not None
                           else [])
    if schema.endswith(".v2") and dqf_planes is None:
        dqf_planes = {name: np.zeros((ny, nx), np.float32)
                      for name in products}
    named = [("cloud_top_height_m", heights)]
    if cloud_top_pressure_hpa is not None:
        named.append(("cloud_top_pressure_hpa", cloud_top_pressure_hpa))
    named += [("lat", lat), ("lon", lon)]
    if dqf_planes:
        for product in products:
            if product in dqf_planes:
                named.append((f"{product.lower()}_dqf",
                              dqf_planes[product]))
    payload, planes, order, arrays = _pack_planes(named)
    meta = {
        "schema": schema,
        "status": status,
        "satellite": satellite,
        "sector": sector,
        "scan_start": scan_start,
        "scan_end": scan_end,
        "sources": [
            source(name, condemn_mask=None,
                   dqf_plane=(f"{name.lower()}_dqf"
                              if dqf_planes and name in dqf_planes else None))
            for name in products],
        "projection": dict(PROJECTION if projection is None else projection),
        "nx": int(nx),
        "ny": int(ny),
        "x_scan_rad": (list(np.linspace(-0.05, 0.05, nx)) if x_scan_rad is None
                       else list(map(float, x_scan_rad))),
        "y_scan_rad": (list(np.linspace(0.10, 0.06, ny)) if y_scan_rad is None
                       else list(map(float, y_scan_rad))),
        "planes": planes,
        "plane_order": order,
        "arrays": arrays,
        "payload_bytes": len(payload),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "pairs_with_schema": (CWP_SCHEMA_V2 if schema.endswith(".v2")
                              else CWP_SCHEMA),
        "regrid": NO_REGRID,
    }
    if window is not None:
        meta["window"] = list(window)
    if sibling is not None:
        meta["sibling"] = dict(sibling)
    path = Path(path)
    path.write_bytes(_encode(meta, payload))
    return path


def sibling_block(cwp_path) -> dict:
    """The ``sibling`` block ``rw_goes cloud-top --pairs-with`` writes."""

    from woof.globe.obs_pack import read_goes_pack

    pack = read_goes_pack(cwp_path)
    return {
        "schema": pack.schema,
        "filename": Path(cwp_path).name,
        "content_sha256": pack.meta["content_sha256"],
        "nx": int(pack.meta["nx"]),
        "ny": int(pack.meta["ny"]),
    }
