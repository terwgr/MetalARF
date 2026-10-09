"""Reading a brightness-temperature pack on a published engine.

`rw_goes bt` is a door this package publishes and the engine's bundle does
not, and it writes a `gpuwm-obs.goes-bt.v1` pack in the engine's own
`GPWMGOES` container.  The engine's reader knows the container and not the
family: a published `KNOWN_SCHEMAS` lists the two cloud families and stops,
so every BT pack was refused by name before a plane was read and the whole
ABI radiance leg stopped at its first file.

What this file holds:

* THE READ IS THE ENGINE'S.  The widening is one table entry for the length
  of one call.  Everything a pack can be wrong about is still refused, in
  the engine's own words, and `expected_schema` still fails closed on the
  family -- the two GOES families share a container on purpose, and a pack
  of the wrong one decodes perfectly and answers a different question.
* THE TABLE IS PUT BACK.  Including after a refused pack.  A module
  attribute left widened would make the next read in the same process accept
  a family nobody decided to accept.
* IT RETIRES ITSELF.  An engine whose own table lists the family is used
  directly, and the probe reads `KNOWN_SCHEMAS` rather than the constant,
  because an engine that defined the name and forgot the table row would
  still refuse every pack.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from woof.globe import obs_pack  # noqa: E402
from goes_pack_fixtures import write_bt_pack, write_cwp_pack  # noqa: E402


def _planes(ny=3, nx=4):
    bt = np.full((ny, nx), 280.0, dtype=np.float32)
    lat = np.linspace(30.0, 31.0, ny * nx, dtype=np.float32).reshape(ny, nx)
    lon = np.linspace(-100.0, -99.0, ny * nx, dtype=np.float32).reshape(ny, nx)
    return {"bt": bt, "rad": bt * 0.3, "lat": lat, "lon": lon}


def test_the_family_name_is_the_engines_when_the_engine_has_it() -> None:
    """The carried constant is a stand-in, never a competing value."""

    import woof.obs.goes_pack as engine

    carried = obs_pack.BT_SCHEMA_V1
    if hasattr(engine, "BT_SCHEMA_V1"):
        assert carried == engine.BT_SCHEMA_V1
        assert obs_pack.bt_schemas() == tuple(engine.BT_SCHEMAS)
    else:
        assert carried == "gpuwm-obs.goes-bt.v1"
        assert obs_pack.bt_schemas() == (carried,)


def test_the_probe_reads_the_table_the_reader_actually_checks(
        monkeypatch) -> None:
    """A constant without a table row still refuses every pack."""

    import woof.obs.goes_pack as engine

    monkeypatch.setattr(engine, "BT_SCHEMAS", (obs_pack.BT_SCHEMA_V1,),
                        raising=False)
    monkeypatch.setattr(engine, "KNOWN_SCHEMAS", ("gpuwm-obs.goes-cwp.v1",))
    assert not obs_pack.engine_knows_bt_schema()

    monkeypatch.setattr(engine, "KNOWN_SCHEMAS",
                        ("gpuwm-obs.goes-cwp.v1", obs_pack.BT_SCHEMA_V1))
    assert obs_pack.engine_knows_bt_schema()


def test_a_bt_pack_opens_through_the_package(tmp_path) -> None:
    """The whole point, on whatever engine is installed."""

    path = write_bt_pack(tmp_path / "bt.goespack", **_planes())
    pack = obs_pack.read_goes_pack(path, expected_schema=obs_pack.BT_SCHEMA_V1)
    assert pack.schema == obs_pack.BT_SCHEMA_V1
    assert pack.shape == (3, 4)
    for plane in ("bt", "rad", "lat", "lon"):
        assert plane in pack.planes, plane
    assert pack.meta["status"] == "READY"


def test_the_wrong_family_is_still_refused(tmp_path) -> None:
    """The two families share a container; failing closed is the point."""

    path = write_bt_pack(tmp_path / "bt.goespack", **_planes())
    with pytest.raises(obs_pack.GoesPackError):
        obs_pack.read_goes_pack(path, expected_schema="gpuwm-obs.goes-cwp.v2")


def test_an_unknown_family_is_still_refused(tmp_path) -> None:
    """The widening is one entry, not an opened door."""

    path = write_bt_pack(tmp_path / "odd.goespack",
                         schema="gpuwm-obs.goes-invented.v9", **_planes())
    with pytest.raises(obs_pack.GoesPackError) as caught:
        obs_pack.read_goes_pack(path)
    assert "gpuwm-obs.goes-invented.v9" in str(caught.value)


def test_a_corrupt_pack_earns_the_engines_own_refusal(tmp_path) -> None:
    path = write_bt_pack(tmp_path / "bt.goespack", **_planes())
    data = bytearray(Path(path).read_bytes())
    data[:8] = b"NOTAPACK"
    Path(path).write_bytes(bytes(data))
    with pytest.raises(obs_pack.GoesPackError) as caught:
        obs_pack.read_goes_pack(path)
    assert "magic" in str(caught.value)


def test_the_engines_table_is_restored_after_a_read(tmp_path) -> None:
    import woof.obs.goes_pack as engine

    before = engine.KNOWN_SCHEMAS
    path = write_bt_pack(tmp_path / "bt.goespack", **_planes())
    obs_pack.read_goes_pack(path)
    assert engine.KNOWN_SCHEMAS == before


def test_the_engines_table_is_restored_after_a_refusal(tmp_path) -> None:
    """A refused pack must not leave the next read widened."""

    import woof.obs.goes_pack as engine

    before = engine.KNOWN_SCHEMAS
    path = write_bt_pack(tmp_path / "bt.goespack", **_planes())
    with pytest.raises(obs_pack.GoesPackError):
        obs_pack.read_goes_pack(path, expected_schema="gpuwm-obs.goes-cwp.v1")
    assert engine.KNOWN_SCHEMAS == before


def test_a_read_inside_a_read_finishes(tmp_path) -> None:
    """A nested pack read must not hang on the table lock.

    The widening swaps a process-global attribute and holds a lock across
    the read.  With a plain lock a caller who opened a second pack from
    inside the first would block on itself, and a deadlock is the one
    failure that says nothing at all: no refusal, no traceback, a suite
    that stops.  Two things keep it from happening and this asserts on
    both: the inner context sees the table already widened and never
    reaches the lock, and the lock is reentrant if it ever does.
    """

    import woof.obs.goes_pack as engine

    before = engine.KNOWN_SCHEMAS
    outer = write_bt_pack(tmp_path / "outer.goespack", **_planes())
    inner = write_bt_pack(tmp_path / "inner.goespack", **_planes())

    with obs_pack._bt_family_known() as widened:
        assert widened is True
        assert obs_pack.read_goes_pack(inner).schema == obs_pack.BT_SCHEMA_V1
        # Acquiring again in this thread is what a future narrower
        # widening would do; a plain lock deadlocks here.
        with obs_pack._TABLE_LOCK:
            pass
    assert engine.KNOWN_SCHEMAS == before
    assert obs_pack.read_goes_pack(outer).schema == obs_pack.BT_SCHEMA_V1
    assert engine.KNOWN_SCHEMAS == before


def test_the_error_class_is_the_engines_own() -> None:
    """A second class with the same name is a handler that never fires."""

    import woof.obs.goes_pack as engine

    assert obs_pack.GoesPackError is engine.GoesPackError


def test_a_cloud_pack_is_unaffected(tmp_path) -> None:
    """The families the engine already knows read exactly as before."""

    shape = (3, 4)
    ones = np.ones(shape, dtype=np.float32)
    path = write_cwp_pack(
        tmp_path / "cwp.goespack",
        cod=ones * 10.0, cps=ones * 12.0, phase=ones,
        lat=np.linspace(30.0, 31.0, 12, dtype=np.float32).reshape(shape),
        lon=np.linspace(-100.0, -99.0, 12, dtype=np.float32).reshape(shape),
    )
    pack = obs_pack.read_goes_pack(path)
    assert pack.family == "cwp"
