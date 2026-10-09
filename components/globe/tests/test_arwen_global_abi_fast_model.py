"""The ABI fast forward model's arithmetic and its table contract, without
the reference, without a granule and without the card.

Proven here: the band-corrected Planck relation round-trips; the emission
march reads the skin through a transparent sky, the layer temperature
through an opaque isothermal one, and a gray surface's emissivity; the
feature vocabulary follows its definitions (thickness, vapor path, slant
path above the layer middle, the neighbour rows); a planted linear table
is recovered by the trainer to rounding and the two-term trainer
recovers a planted wet-plus-dry law on synthetic columns (both
directions: the fitted table reproduces the planted optical depths on
held-out columns and a table fitted to a different law does not); the
emissivity table composes fractions; the columns stream carries the
coordinate and the Rust operator, when its binary is present, agrees with
the Python oracle to 1e-6 K and refuses another coordinate by name.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest

from woof.globe import abi_fast_model as fm
from woof.globe import abi_reference as ref


def _planck():
    return {"fk1": 10860.400390625, "fk2": 1395.18994140625, "bc1": 0.07480999827384949, "bc2": 0.999750018119812}


def _columns(n=60, nlay=12, seed=3):
    """Synthetic columns on a fixed hybrid coordinate: warm moist to cold dry."""
    rng = np.random.default_rng(seed)
    a_half = np.linspace(100.0, 0.0, nlay + 1) ** 1.0
    a_half = np.concatenate([[100.0], np.zeros(nlay)])                  # top 1 hPa, then sigma
    b_half = np.linspace(0.0, 1.0, nlay + 1) ** 1.6
    b_half[0] = 0.0
    ps = rng.uniform(70000.0, 102000.0, n)
    p_half = a_half[None, :] + b_half[None, :] * ps[:, None]
    p_full = np.sqrt(p_half[:, :-1] * p_half[:, 1:])
    t_sfc = rng.uniform(260.0, 310.0, n)
    temperature = t_sfc[:, None] - 6.5e-3 * 7000.0 * np.log(ps[:, None] / p_full)
    temperature = np.maximum(temperature, 195.0)
    q_sfc = rng.uniform(0.5, 18.0, n)
    q = q_sfc[:, None] * (p_full / ps[:, None]) ** 3.0
    zen = rng.uniform(5.0, 65.0, n)
    return {
        "temperature_k": temperature, "q_gkg": q, "p_half_hpa": p_half / 100.0, "p_full_hpa": p_full / 100.0,
        "zenith": zen, "skin_k": t_sfc + rng.normal(0.0, 1.0, n), "land_fraction": (rng.uniform(0, 1, n) > 0.5).astype(float),
        "landuse_category": rng.integers(1, 21, n).astype(np.int32), "sea_ice_fraction": np.zeros(n), "snowh_m": np.zeros(n),
        "lat": rng.uniform(-60, 60, n), "lon": np.round(rng.uniform(-140, -10, n) * 4) / 4, "psfc_hpa": ps / 100.0,
        "o3_ppmv": np.stack([ref.ozone_ppmv(p_full[i] / 100.0) for i in range(n)]),
        "wind10_ms": rng.uniform(0, 12, n), "climatology": np.ones(n, dtype=np.int32), "a_half_pa": a_half, "b_half": b_half,
        "emissivity_channels": [13], "emissivity_user": np.full((n, 1), 0.99), "checks": {},
    }


def _emissivity():
    return {"water": 0.981, "land_by_igbp_class": [0.96 + 0.001 * k for k in range(20)], "snow": 0.985, "ice": 0.98,
            "snow_depth_threshold_m": 0.01}


def test_planck_round_trips_and_the_march_reads_skin_or_layer():
    p = _planck()
    t = np.array([200.0, 250.0, 300.0, 330.0])
    assert np.allclose(fm.planck_temperature(p, fm.planck_radiance(p, t)), t, atol=1e-9)
    temp = np.full((1, 6), 250.0)
    transparent = np.zeros((1, 6))
    bt = fm.planck_temperature(p, fm.emission_radiance(p, transparent, temp, np.array([301.5]), np.array([1.0])))
    assert abs(bt[0] - 301.5) < 1e-9
    opaque = np.full((1, 6), 5.0)
    bt = fm.planck_temperature(p, fm.emission_radiance(p, opaque, temp, np.array([301.5]), np.array([1.0])))
    assert abs(bt[0] - 250.0) < 1e-6
    gray = fm.planck_temperature(p, fm.emission_radiance(p, transparent, temp, np.array([300.0]), np.array([0.9])))
    assert abs(gray[0] - fm.planck_temperature(p, 0.9 * fm.planck_radiance(p, 300.0))) < 1e-9


def test_features_follow_their_definitions():
    t = np.array([[220.0, 250.0, 280.0]])
    q = np.array([[0.01, 1.0, 10.0]])
    p_half = np.array([[100.0, 300.0, 600.0, 1000.0]])
    p_full = np.array([[173.2, 424.3, 774.6]])
    f = fm.features(t, q, p_half, p_full, np.array([60.0]))
    assert set(f) == set(fm.FEATURE_NAMES)
    assert f["dp"][0, 0] == 200.0 and f["u"][0, 1] == 300.0 and f["Tn"][0, 1] == 0.0
    assert abs(f["lsec"][0, 0] - np.log(2.0)) < 1e-9
    assert abs(f["lUs"][0, 1] - np.log(304.0)) < 1e-9          # (2 + 150) above the middle of layer 1, times sec 60
    assert f["lu_up"][0, 0] == f["lu"][0, 0] and f["lu_dn"][0, 2] == f["lu"][0, 2] and f["lu_up"][0, 2] == f["lu"][0, 1]
    assert f["Tn_dn"][0, 0] == f["Tn"][0, 1]


def test_emissivity_table_composes_fractions():
    e = _emissivity()
    cols = {"land_fraction": np.array([0.0, 1.0, 1.0, 0.0, 0.5]), "landuse_category": np.array([17, 10, 10, 17, 10]),
            "sea_ice_fraction": np.array([0.0, 0.0, 0.0, 1.0, 0.0]), "snowh_m": np.array([0.0, 0.0, 0.5, 0.0, 0.0])}
    out = fm.table_emissivity(e, cols)
    assert np.allclose(out, [0.981, 0.969, 0.985, 0.98, 0.5 * 0.981 + 0.5 * 0.969])


def test_linear_trainer_recovers_a_planted_window_law_on_held_out_columns():
    cols = _columns()
    feat = fm.features(cols["temperature_k"], cols["q_gkg"], cols["p_half_hpa"], cols["p_full_hpa"], cols["zenith"])
    # a planted law in the trainer's own vocabulary: dry + continuum (self and foreign)
    od = 2.0e-4 * feat["dp"] * (1.0 + 0.1 * feat["Tn"]) + 1.5e-3 * feat["u"] + 4.0e-4 * feat["u"] * feat["e"]
    train = (np.round(cols["lon"] * 4).astype(int) % 2 == 0)
    table = fm.train_band(13, cols, od, train=train, form="linear", planck=_planck())
    table["emissivity"] = _emissivity()
    pred = fm.layer_optical_depth(table, feat)
    test = ~train
    rel = np.abs(pred[test] - od[test]) / od[test]
    # inside the training envelope the planted law is in the trainer's span (rounding); a held-out
    # column outside it is held at the envelope's edge, so its error is bounded by the clip margin
    assert np.median(rel) < 1e-8 and np.max(rel) < 5e-2, (float(np.median(rel)), float(rel.max()))
    report = fm.validate_band(table, cols, od, test=test, emissivity=np.full(cols["lat"].size, 0.98))
    assert report["test_rms_k"] < 5e-3 and report["test_columns"] == int(test.sum())


def test_two_term_trainer_recovers_a_planted_law_and_a_wrong_law_does_not_fit():
    cols = _columns(n=400, seed=5)
    feat = fm.features(cols["temperature_k"], cols["q_gkg"], cols["p_half_hpa"], cols["p_full_hpa"], cols["zenith"])
    wet = np.exp(-2.0 + 0.9 * feat["lu"] + 0.15 * feat["lnp"] - 0.3 * feat["Tn"] + 0.05 * feat["lUs"])
    dry = 3.0e-4 * feat["dp"] * (1.0 + 0.2 * feat["Tn"])
    od = wet + dry
    train = (np.round(cols["lon"] * 4).astype(int) % 2 == 0)
    table = fm.train_band(8, cols, od, train=train, form="two_term", planck=_planck())
    pred = fm.layer_optical_depth(table, feat)
    rel = np.abs(pred - od) / od
    assert np.median(rel[~train]) < 2e-3, float(np.median(rel[~train]))
    # the other direction: the same table applied to columns drawn from a different law reads them wrong
    other = od * (1.0 + 0.5 * np.tanh(feat["Tn"]))
    assert np.median(np.abs(pred - other) / other) > 5e-2


def test_oracle_jacobians_have_the_signs_of_the_physics():
    cols = _columns(n=20, seed=9)
    cols["skin_k"] = cols["temperature_k"][:, -1] + 3.0          # a skin warmer than the air above it
    feat = fm.features(cols["temperature_k"], cols["q_gkg"], cols["p_half_hpa"], cols["p_full_hpa"], cols["zenith"])
    od = 2.0e-4 * feat["u"] + 2.0e-5 * feat["dp"]                                    # a semi-transparent window
    train = np.ones(cols["lat"].size, dtype=bool)
    table = fm.train_band(13, cols, od, train=train, form="linear", planck=_planck())
    table["emissivity"] = dict(_emissivity(), water=1.0, land_by_igbp_class=[1.0] * 20, snow=1.0, ice=1.0)   # a black surface
    full = {"schema": fm.FAST_MODEL_SCHEMA, "bands": {"13": table}}
    jac = fm.jacobians(full, 13, cols)
    assert np.all(jac["jac_tskin"] > 0.0) and np.all(jac["jac_tskin"] <= 1.0)      # a window band sees the skin
    assert jac["jac_tskin"].mean() > 0.3
    assert np.all(jac["jac_t"].sum(axis=1) >= 0.0)
    # a uniform relative moistening of a column under a skin warmer than its air reads colder
    assert np.all(np.sum(jac["jac_q"] * cols["q_gkg"], axis=1) < 0.0)


def _write_table(path: Path, cols, od, planck):
    train = np.ones(cols["lat"].size, dtype=bool)
    band = fm.train_band(13, cols, od, train=train, form="linear", planck=planck)
    band["emissivity"] = _emissivity()
    table = {"schema": fm.FAST_MODEL_SCHEMA, "satellite": "G19", "sensor": "abi",
             "vertical": fm.coordinate_identity(cols["a_half_pa"], cols["b_half"]), "bands": {"13": band}}
    fm.write_table(path, table)
    return table


def _rw_goes():
    env = os.environ.get("WOOF_RW_GOES")
    candidates = [env] if env else []
    repo = Path(__file__).resolve().parents[1]
    candidates += [str(repo / "tools" / "rustwx" / "target" / "release" / name) for name in ("rw_goes", "rw_goes.exe")]
    found = shutil.which("rw_goes")
    if found:
        candidates.append(found)
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    return None


def test_columns_stream_round_trips_through_the_sidecar(tmp_path):
    cols = _columns(n=7, seed=1)
    path = ref.write_columns(tmp_path / "c.bin", cols, provenance={"test": True})
    raw = path.read_bytes()
    magic, version, n, nlay, nuser = struct.unpack_from("<5i", raw, 0)
    assert (magic, version, n, nlay, nuser) == (ref.COLUMNS_MAGIC, 2, 7, 12, 1)
    a = np.frombuffer(raw, dtype="<f8", count=13, offset=20)
    assert np.array_equal(a, cols["a_half_pa"])
    side = json.loads(Path(f"{path}.json").read_text())
    assert side["schema"] == "gpuwm-da.abi-columns.v2" and side["layers"] == 12 and len(side["coordinate_sha256"]) == 64


@pytest.mark.skipif(_rw_goes() is None, reason="rw_goes is not built here; the Rust operator's agreement is proven where it is")
def test_rust_forward_agrees_with_the_oracle_and_refuses_another_coordinate(tmp_path):
    cols = _columns(n=25, seed=11)
    feat = fm.features(cols["temperature_k"], cols["q_gkg"], cols["p_half_hpa"], cols["p_full_hpa"], cols["zenith"])
    od = 3.0e-4 * feat["dp"] * (1.0 + 0.1 * feat["Tn"]) + 1.2e-3 * feat["u"] + 5.0e-4 * feat["u"] * feat["e"]
    table = _write_table(tmp_path / "table.json", cols, od, _planck())
    columns_path = ref.write_columns(tmp_path / "columns.bin", cols)
    exe = _rw_goes()
    out = tmp_path / "fwd.bin"
    done = subprocess.run([str(exe), "forward", "--columns", str(columns_path), "--table", str(tmp_path / "table.json"),
                           "--out", str(out), "--emis-mode", "0", "--threads", "2"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    record = json.loads(done.stdout)
    assert record["schema"] == "gpuwm-da.abi-forward.v1" and record["bands"] == [13]
    rust = ref.read_crtm_output(out)
    oracle = fm.jacobians(table, 13, cols)
    assert np.max(np.abs(rust["bt"][13] - oracle["bt"])) < 1e-6
    assert np.max(np.abs(rust["jac_tskin"][13] - oracle["jac_tskin"])) < 1e-6
    assert np.max(np.abs(rust["jac_t"][13] - oracle["jac_t"])) < 1e-5
    assert np.max(np.abs(rust["emissivity"][13] - oracle["emissivity"])) < 1e-12
    # emissivity mode 1 reads the columns' own value
    done = subprocess.run([str(exe), "forward", "--columns", str(columns_path), "--table", str(tmp_path / "table.json"),
                           "--out", str(out), "--emis-mode", "1"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert np.allclose(ref.read_crtm_output(out)["emissivity"][13], 0.99)
    # another coordinate is refused by name
    other = dict(cols)
    other["b_half"] = cols["b_half"] ** 1.1
    other_path = ref.write_columns(tmp_path / "other.bin", other)
    done = subprocess.run([str(exe), "forward", "--columns", str(other_path), "--table", str(tmp_path / "table.json"),
                           "--out", str(out)], capture_output=True, text=True)
    assert done.returncode != 0 and "vertical coordinate mismatch" in done.stderr
