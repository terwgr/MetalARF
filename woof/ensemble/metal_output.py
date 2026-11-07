"""The CUDA ensemble product engine, fed by Metal member forecasts.

The engine's reductions already run on the host (the Rust CPU reducer reads
the diagnostic spool), so a Metal ensemble reuses the whole of it: the
spool, the hourly replay, the product files and the maps. Only the member
diagnostics differ. CUDA derives them in kernels; here each kernel is
transcribed to NumPy from its CUDA source, one IEEE operation per NumPy
operation and no fused multiply-add. Transcendental functions are the
host libm's.
"""
from __future__ import annotations

import hashlib
from operator import index

import numpy as np

from woof.ensemble.batch_product_output import HeadlineDiagnosticCollector
from woof.ensemble.ice_accretion import FIELDS as ICE_FIELDS, SavedFrameIce

_MISSING = np.float32(np.nan)
_SURFACE = ("U10", "V10", "T2", "Q2", "PSFC", "RAINNC", "RAINC", "RAINSH",
            "REFD_MAX", "GUST", "WIND_GUST", "UH")


def _host(value, *, levels=None):
    """A float32 host copy of a MetalArray or ndarray, optionally its lowest levels."""
    if levels is not None:
        value = value[:levels]
    to_numpy = getattr(value, "to_numpy", None)
    if callable(to_numpy):
        value = to_numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32))


def headline_wind(u, v):
    """``ensemble_headline_diagnostics`` wind: sqrt(u*u + v*v) in float32."""
    return np.sqrt(u * u + v * v)


def headline_cumulative(rainnc, rainc=None, rainsh=None):
    """The summed precipitation counters, added in the kernel's order."""
    total = rainnc + (rainc if rainc is not None else np.float32(0))
    return total + (rainsh if rainsh is not None else np.float32(0))


def headline_rain_difference(current, earlier):
    """``ensemble_headline_rain_difference``: fmaxf(0, current - earlier)."""
    return np.fmax(np.float32(0), current - earlier)


def headline_dewpoint_humidity(t2, q2, psfc):
    """The 2 m dewpoint (K) and relative humidity (%) of ``ensemble_headline_diagnostics``."""
    with np.errstate(all="ignore"):
        moisture, pressure = q2.astype(np.float64), psfc.astype(np.float64)
        vapor = (moisture * pressure) / (0.622 + moisture)
        logarithm = np.log(np.fmax(vapor, 1.0) / 611.2)
        td = (243.5 * logarithm) / (17.67 - logarithm)
        dewpoint = np.where(np.isfinite(q2) & np.isfinite(psfc) & (q2 > 0) & (psfc > 0),
                            (td + 273.15).astype(np.float32), _MISSING)
        temperature = t2.astype(np.float64) - 273.15
        saturation = 611.2 * np.exp((17.67 * temperature) / (temperature + 243.5))
        rh = (100.0 * vapor) / saturation
        clipped = np.fmin(np.fmax(rh, 0.0), 100.0).astype(np.float32)
        humidity = np.where(np.isfinite(t2) & np.isfinite(q2) & np.isfinite(psfc) & (t2 > 0)
                            & ~np.isnan(rh), clipped, _MISSING)
    return dewpoint.astype(np.float32), humidity.astype(np.float32)


def headline_composite(volume):
    """``ensemble_headline_composite``: the NaN-skipping column maximum."""
    result = np.full(volume.shape[1:], np.nan, np.float32)
    for level in volume:
        result = np.where(np.isfinite(level),
                          np.where(np.isnan(result), level, np.fmax(result, level)), result)
    return result.astype(np.float32)


def compound_event(planes, conditions):
    """``ensemble_compound_field``: 1 when every term holds, 0 otherwise, NaN where a term is missing."""
    relations = {"ge": np.greater_equal, "gt": np.greater, "le": np.less_equal, "lt": np.less}
    event = valid = None
    for condition in conditions:
        x = planes[condition.field]
        term = relations[condition.comparison](x, np.float32(condition.threshold))
        event = term if event is None else event & term
        valid = np.isfinite(x) if valid is None else valid & np.isfinite(x)
    return np.where(valid, np.where(event, np.float32(1), np.float32(0)), _MISSING).astype(np.float32)


_RH_COEFFICIENTS = (.99999683, -.0090826951, .000078736169, -.00000061117958,
                    4.3884187e-9, -2.9883885e-11, 2.1874425e-13, -1.7892321e-15,
                    1.1112018e-17, -3.0994571e-20)


def _visibility_rh(p, t, r):
    x = t - 273.15
    pol = np.zeros_like(x)
    for coefficient in reversed(_RH_COEFFICIENTS):
        pol = pol * x + coefficient
    es = 6.1078 / np.power(pol, 8.0)
    return np.fmin(100.0, np.fmax(0.0, 100.0 * (((p / 100.0) * r) / (.62197 + r)) / es))


def visibility_column(p, pb, theta, base, qv, species, *, mode):
    """``visibility_column`` (woof.ensemble.visibility) over the lowest (up to three) levels.

    ``species`` is (qc, qr, qi, qs, qg) with None for an absent carrier;
    ``base`` broadcasts against ``theta`` (a level profile arrives as (nz, 1, 1)).
    """
    def good(x):
        return np.isfinite(x) & (np.abs(x) < 1.e30)
    with np.errstate(all="ignore"):
        pr = p.astype(np.float64) + (pb.astype(np.float64) if mode == 0 else 0.0)
        th = theta.astype(np.float64) + (300.0 if mode in (0, 2) else base.astype(np.float64))
        t = th * np.power(pr / 100000.0, 0.2857142857)
        r = qv.astype(np.float64)
        ok = np.all(good(pr) & good(t) & good(r) & (pr > 0.0) & (t >= 150.0) & (t <= 350.0)
                    & (r >= 0.0), axis=0)
        maxrh = np.zeros(pr.shape[1:])
        for k in range(min(2, pr.shape[0])):
            maxrh = np.fmax(maxrh, _visibility_rh(pr[k], t[k], r[k]))
        mq = []
        for plane in species:
            if plane is None:
                mq.append(np.zeros(pr.shape[1:]))
                continue
            q = plane.astype(np.float64)
            ok = ok & np.all(good(q), axis=0)
            mq.append(np.max(np.fmax(0.0, q), axis=0))
        p0, t0, r0 = pr[0], t[0], r[0]
        tv = (t0 * (1.0 + r0 / .62197)) / (1.0 + r0)
        rho = p0 / (287.0 * tv)
        volume = (1.0 + r0) / rho + (mq[0] + mq[1]) / 1000.0 + (mq[2] + mq[3] + mq[4]) / 917.0
        c = [1000.0 * m / volume for m in mq]
        beta = (144.7 * np.power(c[0], .88) + 2.24 * np.power(c[1], .75) + 327.8 * c[2]
                + 10.36 * np.power(c[3], .7776) + 8.0 * np.power(c[4], .75) + 1.e-10)
        haze = 90000.0 * np.exp(-2.5 * np.fmin(.8, np.fmax(0.0, maxrh / 100.0 - .15)))
        v = np.fmin(90000.0, np.fmin((1000.0 * 3.912023005428146) / beta, haze))
        return np.where(ok, v.astype(np.float32), _MISSING).astype(np.float32)


def saved_frame_ice(rain, previous_rain, previous, t2, skin, q2, psfc, u10, v10, qr, qs, qg,
                    *, hours, have_skin):
    """``saved_frame_ice`` (core/kernels/ice_accretion.cu): liquid, horizontal, radial, wet bulb, rate."""
    with np.errstate(all="ignore"):
        t2d, psfcd = t2.astype(np.float64), psfc.astype(np.float64)
        t = t2d - 273.15
        pressure = psfcd / 100.0
        q, r, s, g = (x.astype(np.float64) for x in (q2, qr, qs, qg))
        delta = rain.astype(np.float64) - previous_rain.astype(np.float64)
        bad = (~np.isfinite(t) | (t2 <= 0) | ~np.isfinite(pressure) | (pressure <= 0)
               | ~np.isfinite(q) | (q < 0) | ~np.isfinite(r) | ~np.isfinite(s) | ~np.isfinite(g)
               | ~np.isfinite(delta) | (delta < 0) | ~np.isfinite(u10) | ~np.isfinite(v10)
               | ~np.isfinite(previous[0]) | ~np.isfinite(previous[1]) | ~np.isfinite(previous[2]))
        if have_skin:
            bad |= ~np.isfinite(skin) | (skin <= 0)
        vapor = pressure * q / (0.622 + q)
        lo, hi = np.full(t.shape, -100.0), t.copy()
        for _ in range(40):
            tw = 0.5 * (lo + hi)
            es = 6.112 * np.exp(17.67 * tw / (tw + 243.5))
            e = es - 0.00066 * (1.0 + 0.00115 * tw) * pressure * (t - tw)
            hi = np.where(e > vapor, tw, hi)
            lo = np.where(e > vapor, lo, tw)
        tw = 0.5 * (lo + hi)
        r = np.fmax(r, 0.0) * 5.0
        s = np.fmax(s, 0.0)
        g = np.fmax(g, 0.0) * 3.0
        total = r + s + g
        flux = psfcd / (287.04 * t2d) * total * 3600.0
        cold = t2 < np.float32(273.15)
        if have_skin:
            cold = cold | (skin < np.float32(273.15))
        freezing = (total > 0) & (flux >= 0.1) & ((s + g) / total <= 0.3) & cold
        active = freezing & (delta > 0) & (hours > 0)
        dz = np.where(active, delta, 0.0)
        chill = active & (tw < 1.0)
        p = (delta / hours) / 25.4 if hours > 0 else np.zeros_like(delta)
        v = np.hypot(u10.astype(np.float64), v10.astype(np.float64)) / (1852.0 / 3600.0)
        w = np.fmax(tw, -7.0)
        ip = 0.1395 * np.power(p, -0.541)
        it = ((-0.0071 * w - 0.1039) * w - 0.3904) * w + 0.5545
        iv = (0.0014 * v + 0.0027) * v + 0.7574
        ilr = np.where(w > -0.35, 0.70 * ip + 0.29 * it + 0.01 * iv,
                       np.where(v > 12.0, 0.73 * ip + 0.01 * it + 0.26 * iv,
                                0.79 * ip + 0.20 * it + 0.01 * iv))
        flat = np.where(chill, delta * np.fmax(ilr, 0.0), 0.0)
        rate = (dz / hours).astype(np.float32) if hours > 0 else np.zeros(t.shape, np.float32)
        outputs = [(previous[0].astype(np.float64) + dz).astype(np.float32),
                   (previous[1].astype(np.float64) + flat).astype(np.float32),
                   (previous[2].astype(np.float64) + flat * 0.394).astype(np.float32),
                   tw.astype(np.float32), rate]
    return [np.where(bad, _MISSING, out).astype(np.float32) for out in outputs]


class HostSavedFrameIce(SavedFrameIce):
    """Saved-frame ice accumulation with the kernel run on the host."""

    def __init__(self):
        super().__init__(np)

    def submit(self, key, tick, fields, rain):
        self.last_interval = None
        if not all(name in fields for name in ("T2", "Q2", "PSFC", "U10", "V10", "QRAIN", "QSNOW")):
            self.invalidate(key)
            return {}
        have_skin = "TSK" in fields
        prior = self.history.get(key)
        if prior is not None and tick <= prior["tick"]:
            raise ValueError("saved-frame ice clock must advance; duplicate output would double accretion")
        if prior is None:
            previous_rain = rain.copy()
            previous = [np.full(rain.shape, 0 if tick == 0 else np.nan, np.float32) for _ in ICE_FIELDS]
            hours = 0.0
        else:
            hours = (tick - prior["tick"]) / 3_600_000_000
            self.intervals.add(hours)
            self.last_interval = hours
            previous_rain = prior["rain"]
            previous = [prior[name] for name in ICE_FIELDS]
        graupel = fields.get("QGRAUP", np.zeros(rain.shape, np.float32))
        liquid, horizontal, radial, _, rate = saved_frame_ice(
            rain, previous_rain, previous, fields["T2"], fields.get("TSK", fields["T2"]),
            fields["Q2"], fields["PSFC"], fields["U10"], fields["V10"], fields["QRAIN"],
            fields["QSNOW"], graupel, hours=hours, have_skin=have_skin)
        self.history[key] = {"tick": tick, "rain": rain.copy(),
                             **dict(zip(ICE_FIELDS, (liquid, horizontal, radial)))}
        return {"freezing_rain_total": liquid, "ice_accretion_horizontal": horizontal,
                "ice_accretion": radial, "freezing_rain_rate": rate}


class MetalEnsembleCollector(HeadlineDiagnosticCollector):
    """The headline collector with member diagnostics derived on the host.

    Member output arrives as MetalArray planes; each frame is copied to the
    host once, diagnosed there and handed to the shared spool, whose
    complete-roster hours reduce and render exactly as on CUDA.
    """

    def __init__(self, root, *, members, renderer, start_time, **options):
        from woof import rustwx
        rustwx.require_ensemble_diagnostic_reducer(renderer)
        super().__init__(root, members=members, renderer=renderer, start_time=start_time,
                         array_module=np, **options)
        self.ice = HostSavedFrameIce()

    def _budget(self):
        return 0

    def _visibility(self, state, fields, streamed, shape):
        """Visibility from a member's three lowest levels, or None when its carriers are absent."""
        def lowest(value):
            return _host(value, levels=3) if value.ndim > 1 else _host(value)
        if streamed:
            names = ("P", "PB", "T", "QVAPOR", "QCLOUD")
            if any(name not in fields for name in names):
                return None
            p, pb, theta, vapor, cloud = (lowest(fields[name]) for name in names)
            base, mode = theta, 0
            species = [cloud] + [None if fields.get(name) is None else lowest(fields[name])
                                 for name in ("QRAIN", "QICE", "QSNOW", "QGRAUP")]
        else:
            carriers = [getattr(state, name, None) for name in ("p", "thp", "thb", "qv", "qc")]
            if any(a is None for a in carriers):
                return None
            p, theta, base, vapor, cloud = (lowest(a) for a in carriers)
            pb, mode = p, 2 if getattr(state, "_WRF_EXACT", False) else 1
            species = [cloud] + [None if getattr(state, name, None) is None
                                 else lowest(getattr(state, name))
                                 for name in ("qr", "qi", "qs", "qg")]
        if p.ndim != 3 or p.shape[1:] != shape:
            raise ValueError("visibility needs mass-level pressure on the output grid")
        if base.ndim == 1:
            base = base[:p.shape[0], None, None]
        return visibility_column(p, pb, theta, base, vapor, species, mode=mode)

    def _submit(self, *, state, streamed, metadata, refl_field, valid_time,
                grid_id, episode, member_id):
        frame = self._frame(state=state, streamed=streamed, metadata=metadata,
            refl_field=refl_field, valid_time=valid_time, grid_id=grid_id,
            episode=episode, member_id=member_id)
        if frame is None:
            return None
        fields, latitude, longitude, valid, geometry, rain_key, tick, spool = frame
        shape = latitude.shape
        surface = {}
        for name in _SURFACE:
            if name in fields:
                surface[name] = _host(fields[name])
                if surface[name].shape != shape:
                    raise ValueError(f"{name} has a different surface grid than ordinary output metadata")
        history = self.rain_history.setdefault(rain_key, {})
        output_key = (rain_key, tick)
        if output_key in self.output_ticks:
            raise ValueError("ensemble member output clock is duplicated")
        prior_outputs = [old_tick for old_key, old_tick in self.output_ticks if old_key == rain_key]
        if prior_outputs and tick <= max(prior_outputs):
            raise ValueError("ensemble member output times must advance in ordinary clock order")
        diagnosed = {}
        if any(request.field == "visibility" for request in self.requests):
            visibility = self._visibility(state, fields, streamed is not None, shape)
            if visibility is not None:
                diagnosed["visibility"] = visibility
        if "U10" in surface and "V10" in surface:
            diagnosed["wind10"] = headline_wind(surface["U10"], surface["V10"])
        if "T2" in surface:
            diagnosed["temperature2"] = surface["T2"]
            diagnosed["temperature2_below_freezing"] = surface["T2"]
        if all(name in surface for name in ("T2", "Q2", "PSFC")):
            diagnosed["dewpoint2"], diagnosed["humidity2"] = headline_dewpoint_humidity(
                surface["T2"], surface["Q2"], surface["PSFC"])
        has_rain = "RAINNC" in surface
        cumulative = (headline_cumulative(surface["RAINNC"], surface.get("RAINC"), surface.get("RAINSH"))
                      if has_rain else history.get(tick))
        if has_rain:
            history = self._record_rain(rain_key, tick, cumulative)
        if cumulative is not None and tick in history:
            # A counter sampler's declared zeros serve a member without a surface driver.
            if self.rain_initial_ticks[rain_key] == 0:
                diagnosed["rain_total"] = headline_rain_difference(cumulative, history["initial"])
            for name, hours in (("qpf_1h", 1), ("qpf_3h", 3), ("qpf_6h", 6)):
                earlier = tick - hours * 3_600_000_000
                if earlier >= 0 and earlier in history:
                    diagnosed[name] = headline_rain_difference(cumulative, history[earlier])
        if has_rain:
            ice_fields = {name: surface[name] for name in ("T2", "Q2", "PSFC", "U10", "V10") if name in surface}
            for name in ("QRAIN", "QSNOW", "QGRAUP", "TSK"):
                if name in fields:
                    value = fields[name]
                    ice_fields[name] = _host(value[0] if value.ndim == 3 else value)
            diagnosed.update(self.ice.submit(rain_key, tick, ice_fields, cumulative))
        else:
            self.ice.invalidate(rain_key)
        if self.member_archive is not None:
            archive = {name: surface[name] for name in ("T2", "U10", "V10") if name in surface}
            if "rain_total" in diagnosed:
                archive["RAIN_TOTAL"] = diagnosed["rain_total"]
            self.member_archive.submit(archive, member_id=member_id, valid_time=valid_time,
                grid_id=grid_id, episode=episode, latitude=latitude, longitude=longitude)
        if "REFD_MAX" in surface:
            diagnosed["refl"] = surface["REFD_MAX"]
        elif "REFL_10CM" in fields:
            volume = _host(fields["REFL_10CM"])
            if volume.ndim == 2 and volume.shape == shape:
                diagnosed["refl"] = volume
            elif volume.ndim == 3 and volume.shape[1:] == shape:
                diagnosed["refl"] = headline_composite(volume)
            else:
                raise ValueError("reflectivity has a different grid than ordinary output metadata")
        for name in ("GUST", "WIND_GUST"):
            if name in surface:
                diagnosed["gust"] = surface[name]
                break
        if "UH" in surface:
            diagnosed["uh"] = surface["UH"]
        for name, conditions in self.events.items():
            if {condition.field for condition in conditions} <= set(diagnosed):
                diagnosed[name] = compound_event(diagnosed, conditions)
        return self._commit(spool, valid, diagnosed, member_id=member_id, grid_id=grid_id,
            episode=episode, geometry=geometry, output_key=output_key, device=0)

    def capture_rain_counters(self, fields, *, valid_time, grid_id, episode,
                              member_id, latitude, longitude, absent_zero_fields=()):
        """Record a precipitation counter endpoint without publishing a frame."""
        with self._lock:
            fields = {str(name).upper(): value for name, value in fields.items()}
            absent_zero_fields = {str(name).upper() for name in absent_zero_fields}
            if not absent_zero_fields <= {"RAINNC", "RAINC", "RAINSH"}:
                raise ValueError("zero-counter declarations must name original precipitation providers")
            if fields.get("RAINNC") is None and "RAINNC" not in absent_zero_fields:
                return False
            latitude, longitude = np.asarray(latitude, np.float32), np.asarray(longitude, np.float32)
            if latitude.ndim == 3 and latitude.shape[0] == 1:
                latitude, longitude = latitude[0], longitude[0]
            if latitude.ndim != 2 or longitude.shape != latitude.shape:
                raise ValueError("rain counter coordinates need matching two-dimensional grids")
            counters = {}
            for name in ("RAINNC", "RAINC", "RAINSH"):
                if fields.get(name) is not None:
                    counters[name] = _host(fields[name])
                    if counters[name].shape != latitude.shape:
                        raise ValueError(f"{name} counter grid differs from its coordinates")
                elif name in absent_zero_fields:
                    counters[name] = np.zeros(latitude.shape, np.float32)
            geometry = hashlib.sha256(latitude.tobytes() + longitude.tobytes()).hexdigest()
            key = self._placement_key(index(grid_id), index(episode), index(member_id), geometry, advance=True)
            self._record_rain(key, self._tick(valid_time), headline_cumulative(
                counters["RAINNC"], counters.get("RAINC"), counters.get("RAINSH")))
            return True
