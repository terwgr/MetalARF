"""The DA scorecard: O-B and O-A as distributions per stream, variable,
region and cycle, with the four assessments of design amendment G.

The instrument is calibrated before any analysis number reads through it:
every synthetic family of :func:`woof.globe.da_scorecard.calibrate`
is planted in both directions (a stream the analysis moved closer reads
so with every row improved, a stream it moved away from stays
engineering-complete and reads so, an unmoved stream reads unmoved, the
conflicting-report family of the amendment reads moved closer with half
its rows worsened, the Desroziers estimate recovers planted observation
errors of 1 and 2, the regions are boxes the planted rows land in
exactly, the withheld rows are read beside the assimilated ones, a stream
that never reached the operators is the one named incomplete, and the
per-cycle merge counts what each cycle said).
"""
from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np
import pytest

from woof.globe.da_scorecard import (
    DESROZIERS_ASSUMPTIONS,
    REGIONS,
    RULE,
    VERDICT_REGION,
    Departures,
    Region,
    calibrate,
    calibration_holds,
    main,
    merge_cycles,
    render_table,
    scorecard,
)
from woof.globe.obs_table import ObsRow


def test_every_calibration_family_holds_both_directions():
    result = calibrate()
    assert calibration_holds(result["readings"]) == []
    readings = result["readings"]
    assert readings["halved_departures"]["rms_ratio"] == pytest.approx(0.5)
    assert readings["doubled_departures"]["verdict"] == "complete"
    assert readings["doubled_departures"]["below"] is False
    assert readings["conflicting_reports"]["o_a_rms"] == pytest.approx(math.sqrt(((5.0 / 3.0) ** 2 + (7.0 / 3.0) ** 2) / 2.0))
    for name, reading in readings["desroziers"]["readings"].items():
        assert abs(reading["estimate"] - reading["planted"]) <= 0.1 * reading["planted"], name


def test_the_calibration_door_prints_readings_and_exits_clean(capsys):
    assert main(["calibrate"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["failures"] == []
    assert set(payload["readings"]) >= {"halved_departures", "conflicting_reports", "desroziers", "regions", "merge"}


def test_the_rule_is_stated_and_the_verdict_region_is_global():
    assert "engineering validity" in RULE and "need not move closer to every stream" in RULE
    assert "-2/3" in RULE
    assert "optimal gain" in DESROZIERS_ASSUMPTIONS
    assert VERDICT_REGION == "global"
    names = [region.name for region in REGIONS]
    assert names[0] == "global" and "conus" in names and "tropics" in names


def test_a_box_across_the_antimeridian_contains_both_sides():
    pacific = Region("pacific", -10.0, 10.0, 160.0, -160.0, "170E to 170W")
    inside = pacific.contains([0.0, 0.0, 0.0, 0.0], [170.0, -170.0, 190.0, 0.0])
    assert inside.tolist() == [True, True, True, False]
    conus = REGIONS[-1]
    assert conus.name == "conus"
    assert conus.contains([40.0, 40.0], [-100.0, 260.0]).tolist() == [True, True]
    assert conus.contains([40.0], [10.0]).tolist() == [False]


def _rows(values, source="s", variable="temperature_k", level=None, lat=40.0, error=1.0):
    when = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)
    return [
        ObsRow(source=source, station_id=f"K{k}", latitude_deg=lat, longitude_deg=-100.0 + k,
               elevation_m=0.0, level_pa=level, valid_time=when, variable=variable,
               value=float(v), error=error)
        for k, v in enumerate(values)
    ]


def test_from_rows_reads_departures_errors_spread_and_offsets_and_drops_operator_nans():
    rows = _rows([10.0, 12.0, 14.0], error=1.5)
    dep = Departures.from_rows(
        rows, [9.0, math.nan, 13.0], [9.5, 12.0, 13.9], withheld=False,
        spread_h=[0.3, 0.4, 0.5], time_offset_s=[-600.0, 0.0, 300.0],
        o_minus_a_label="linearised",
    )
    assert dep.count == 2
    assert dep.o_minus_b.tolist() == pytest.approx([1.0, 1.0])
    assert dep.o_minus_a.tolist() == pytest.approx([0.5, 0.1])
    assert dep.error.tolist() == [1.5, 1.5]
    assert dep.spread_h.tolist() == pytest.approx([0.3, 0.5])
    assert dep.time_offset_s.tolist() == [-600.0, 300.0]
    assert dep.o_minus_a_label == "linearised"
    assert np.isnan(dep.level_pa).all()
    with pytest.raises(ValueError, match="one operator value per row"):
        Departures.from_rows(rows, [1.0, 2.0], [1.0, 2.0, 3.0], withheld=False)


def test_a_stream_the_analysis_did_not_move_stays_complete_and_reads_unmoved():
    """The dewpoint rows with the moisture update off: read, evaluated,
    unchanged.  Under amendment G that is a statistical reading (unmoved),
    not an engineering failure: the card is complete and says which
    variable the analysis did not move."""
    taken = Departures.from_rows(_rows([1.0] * 5), [0.0] * 5, [0.5] * 5, withheld=False)
    untouched = Departures.from_rows(
        _rows([1.0] * 5, variable="dewpoint_k"), [0.0] * 5, [0.0] * 5, withheld=False,
    )
    card = scorecard(Departures.concatenate([taken, untouched]), label="t")
    stream = card["streams"]["s"]
    temperature = stream["variables"]["temperature_k"]["assessments"]
    dewpoint = stream["variables"]["dewpoint_k"]["assessments"]
    assert temperature["engineering"]["verdict"] == "pass"
    assert temperature["statistical_consistency"]["o_a_rms_below_o_b_rms"] is True
    assert dewpoint["engineering"]["verdict"] == "pass"
    assert dewpoint["statistical_consistency"]["moved"] is False
    assert card["verdict"] == "complete" and card["incomplete_streams"] == []
    assert card["assessments"]["statistical_consistency"]["unmoved"] == ["s/dewpoint_k"]
    assert stream["o_a_not_below_o_b_variables"] == ["dewpoint_k"]
    table = render_table(card)
    assert "engineering COMPLETE" in table.splitlines()[0]
    assert any("dewpoint_k" in line and line.rstrip().endswith("pass") for line in table.splitlines())
    # The distributions ride with every cell.
    cell = stream["variables"]["temperature_k"]["regions"]["global"]
    assert cell["o_minus_b"]["quantiles"]["p50"] == pytest.approx(1.0)
    assert cell["moved_closer_fraction"] == 1.0
    assert cell["consistency"]["assigned_sigma_o"] == pytest.approx(1.0)


def test_the_consistency_reading_uses_the_ensemble_spread_when_it_is_there():
    rng = np.random.default_rng(3)
    n = 2000
    sigma_o, spread = 1.0, 0.8
    d_ob = rng.normal(0.0, math.sqrt(sigma_o ** 2 + spread ** 2), n)
    d_oa = d_ob * sigma_o ** 2 / (sigma_o ** 2 + spread ** 2)
    rows = [
        ObsRow(source="e", station_id=f"S{k}", latitude_deg=10.0, longitude_deg=float(k % 360) - 180.0,
               elevation_m=0.0, level_pa=None, valid_time=dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc),
               variable="temperature_k", value=0.0, error=sigma_o)
        for k in range(n)
    ]
    dep = Departures.from_rows(rows, -d_ob, -d_oa, withheld=False, spread_h=np.full(n, spread))
    cell = scorecard(dep)["streams"]["e"]["variables"]["temperature_k"]["regions"]["global"]
    consistency = cell["consistency"]
    assert consistency["innovation_variance_ratio"] == pytest.approx(1.0, abs=0.1)
    assert consistency["desroziers_ratio"] == pytest.approx(1.0, abs=0.1)
    assert consistency["plausible"] is True


def test_merge_cycles_tracks_every_cycle_in_order():
    good = scorecard(Departures.from_rows(_rows([1.0] * 3), [0.0] * 3, [0.5] * 3, withheld=False))
    bad = scorecard(Departures.from_rows(_rows([1.0] * 3), [0.0] * 3, [2.0] * 3, withheld=False))
    merged = merge_cycles([("a", good), ("b", bad)])
    entry = merged["streams"]["s"]["temperature_k"]
    assert entry["cycles"] == 2 and entry["complete"] == 2 and entry["o_a_below_o_b"] == 1
    assert entry["o_a_not_below_o_b_cycles"] == ["b"] and entry["incomplete_cycles"] == []
    assert entry["o_minus_b_rms"] == pytest.approx([1.0, 1.0])
    assert entry["o_minus_a_rms"] == pytest.approx([0.5, 1.0])
    assert merged["complete_cycles"] == 2 and merged["incomplete_cycles"] == []
