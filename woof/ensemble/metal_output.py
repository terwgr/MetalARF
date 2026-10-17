"""Host-side ensemble products for Metal ensemble member forecasts."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def _host(value):
    to_numpy = getattr(value, "to_numpy", None)
    if callable(to_numpy):
        value = to_numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32))



#: The fields a Metal ensemble publishes statistics of.
METAL_PRODUCT_FIELDS = ("wind10", "temperature2", "rain_total")

class MetalEnsembleCollector:
    """Capture ordinary history callbacks and publish complete-roster products.

    Prognostic stepping and surface diagnostics stay on Metal. The public
    products are reduced on the host after member output callbacks have
    copied their small surface planes from MetalArray storage.
    """

    def __init__(self, root, members, thresholds=None):
        from woof.ensemble.batch_products import DEFAULT_THRESHOLDS
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.members = int(members)
        overrides = dict(thresholds or {})
        # The CUDA products' table: each field's default thresholds in its
        # stored units, replaced field by field by the request.
        self.thresholds = {
            field: (units, tuple(float(np.float32(x))
                                 for x in overrides.get(field, values)))
            for field, (units, values) in DEFAULT_THRESHOLDS.items()}
        self.frames = {}
        self.member_history_count = 0
        self.products = None

    def submit(self, *, state, streamed, metadata, refl_field, valid_time,
               grid_id, episode, member_id):
        if streamed is not None:
            supplied = streamed.history_fields()
            if getattr(supplied, "deferred", False):
                supplied = supplied.materialize()
            fields = dict(supplied)
        else:
            physics = getattr(state, "physics", None)
            output = getattr(physics, "output_fields", None)
            fields = dict(output()) if callable(output) else {}
        fields = {str(name).upper(): value for name, value in fields.items()}
        lat = np.asarray(metadata["XLAT"], dtype=np.float32)
        lon = np.asarray(metadata["XLONG"], dtype=np.float32)
        if lat.ndim == 3 and lat.shape[0] == 1:
            lat, lon = lat[0], lon[0]
        if lat.ndim != 2 or lon.shape != lat.shape:
            raise ValueError("Metal ensemble products require matching 2-D coordinates")
        key = (int(grid_id), int(episode), str(valid_time))
        frame = self.frames.setdefault(key, {"latitude": lat, "longitude": lon,
                                              "members": {}})
        if not np.array_equal(frame["latitude"], lat) or not np.array_equal(frame["longitude"], lon):
            raise ValueError("Metal ensemble member output grids differ at the same frame")
        member = int(member_id)
        if member in frame["members"]:
            raise ValueError(f"Metal ensemble member {member} wrote duplicate frame {valid_time}")
        surface = {}
        for name in ("U10", "V10", "T2", "RAINNC", "RAINC", "RAINSH"):
            if name in fields:
                value = _host(fields[name])
                if value.shape != lat.shape:
                    raise ValueError(f"Metal ensemble {name} has shape {value.shape}, expected {lat.shape}")
                if not np.isfinite(value).all():
                    raise ValueError(f"Metal ensemble {name} contains non-finite values")
                surface[name] = value
        if "U10" in surface and "V10" in surface:
            surface["wind10"] = np.hypot(surface["U10"], surface["V10"])
        if "T2" in surface:
            surface["temperature2"] = surface["T2"]
        rain_names = ("RAINNC", "RAINC", "RAINSH")
        present = [name for name in rain_names if name in surface]
        if present:
            cumulative = sum((surface[name] for name in present), np.zeros(lat.shape, np.float32))
            rain_key = (int(grid_id), int(episode), member)
            initial = getattr(self, "_initial_rain", None)
            if initial is None:
                self._initial_rain = {}
                initial = self._initial_rain
            if rain_key not in initial:
                initial[rain_key] = cumulative.copy()
            surface["rain_total"] = np.maximum(cumulative - initial[rain_key], 0.0)
        if not any(name in surface for name in METAL_PRODUCT_FIELDS):
            raise ValueError("Metal ensemble history has none of wind10, temperature2 or rain_total")
        frame["members"][member] = {name: value for name, value in surface.items()
                                     if name in METAL_PRODUCT_FIELDS}
        self.member_history_count += 1

    def finish_run(self):
        manifests = []
        out = self.root / "d01" / "ensemble" / "products"
        out.mkdir(parents=True, exist_ok=True)
        from woof.io.classic_product import ClassicProduct
        for (grid, episode, valid), frame in sorted(self.frames.items(), key=lambda item: item[0][2]):
            members = frame["members"]
            if set(members) != set(range(self.members)):
                raise ValueError(f"Metal ensemble frame {valid} has {len(members)}/{self.members} members")
            fields = sorted(set.intersection(*(set(member) for member in members.values())))
            if not fields:
                raise ValueError(f"Metal ensemble frame {valid} has no common surface products")
            token = str(valid).replace(":", "-").replace("/", "-").replace("\\", "-")
            path = out / f"ensemble_{token}.nc"
            if path.exists():
                raise FileExistsError(f"Metal ensemble product already exists: {path}")
            pending = path.with_suffix(".nc.pending")
            with ClassicProduct(pending) as writer:
                writer.createDimension("south_north", frame["latitude"].shape[0])
                writer.createDimension("west_east", frame["latitude"].shape[1])
                writer.setncattr("ensemble_backend", "metal")
                writer.setncattr("valid_time", str(valid))
                writer.setncattr("member_count", self.members)
                for name in fields:
                    stack = np.stack([members[index][name] for index in range(self.members)])
                    mean = np.mean(stack, axis=0, dtype=np.float64).astype(np.float32)
                    spread = (np.std(stack, axis=0, ddof=1, dtype=np.float64).astype(np.float32)
                              if self.members > 1 else np.zeros_like(mean))
                    if not np.isfinite(mean).all() or not np.isfinite(spread).all():
                        raise ValueError(f"Metal ensemble {name} mean/spread is non-finite")
                    writer.createVariable(f"{name}_mean", "float32", ("south_north", "west_east"))[:] = mean
                    writer.createVariable(f"{name}_spread", "float32", ("south_north", "west_east"))[:] = spread
                    writer.createVariable(f"{name}_min", "float32", ("south_north", "west_east"))[:] = stack.min(axis=0)
                    writer.createVariable(f"{name}_max", "float32", ("south_north", "west_east"))[:] = stack.max(axis=0)
                    units, thresholds = self.thresholds[name]
                    if thresholds:
                        # P(member >= threshold) over the complete roster,
                        # divided in binary64 and rounded once, as on CUDA.
                        axis = f"{name}_threshold"
                        writer.createDimension(axis, len(thresholds))
                        levels = writer.createVariable(axis, "float32", (axis,))
                        levels[:] = np.asarray(thresholds, np.float32)
                        levels.units = units
                        count = (stack[None] >= np.asarray(thresholds, np.float32)[:, None, None, None]).sum(axis=1)
                        writer.createVariable(f"{name}_probability", "float32",
                                              (axis, "south_north", "west_east"))[:] = (
                            count.astype(np.float64) / np.float64(self.members)).astype(np.float32)
                writer.createVariable("XLAT", "float32", ("south_north", "west_east"))[:] = frame["latitude"]
                writer.createVariable("XLONG", "float32", ("south_north", "west_east"))[:] = frame["longitude"]
            pending.replace(path)
            manifests.append({"valid_time": str(valid), "grid_id": grid, "episode": episode,
                              "path": str(path.relative_to(self.root)), "fields": fields,
                              "thresholds": {name: list(self.thresholds[name][1]) for name in fields},
                              "member_count": self.members, "finite": True})
        self.products = {"backend": "metal", "member_history_count": self.member_history_count,
                         "frames": manifests,
                         "statistics": ["mean", "spread", "min", "max", "probability"]}
        (self.root / "ensemble-products.json").write_text(
            json.dumps(self.products, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return self.products

    def require_complete(self):
        if self.products is None:
            raise RuntimeError("Metal ensemble products were not finalized")
