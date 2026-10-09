"""What a stream door returns when it cannot give a table, and which doors
this distribution publishes a command for.

Both are door questions rather than science questions.  A caller that
branches on the exit code -- a workspace, a cycle script, a CI job -- acts on
3 by staging a bundle and on 1 by reading the reason, so a stream whose
archive is simply empty must never answer 3.  And a binary this package pins,
stages and verifies is not shipped until some command runs it.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from woof.globe import cli, obs_streams
from woof.globe.obs_streams import STREAMS


def _run(argv, capsys):
    parser = cli.build_parser()
    args = parser.parse_args(argv)
    code = args.func(args)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_an_empty_archive_window_is_a_refusal_not_a_missing_door(monkeypatch, capsys, tmp_path):
    """Exit 3 says "stage a bundle".  The bundle is staged: the door ran,
    passed its pin and reported an empty collection."""

    def fake_door(name, arguments):
        assert name == "rw_gnssro" and arguments[0] == "list"
        return {"status": "EMPTY", "latest_day_in_bucket": "2025/07/29/"}

    monkeypatch.setattr(obs_streams, "_run_door", fake_door)
    code, out, err = _run([
        "obs", "fetch", "--stream", "gnss-ro",
        "--start", "2026-09-05T00:00:00Z", "--end", "2026-09-07T00:00:00Z",
        "--out", str(tmp_path)], capsys)

    assert code == 1, "an empty window answered with the missing-door code"
    record = json.loads(out)
    assert record["status"] == "EMPTY"
    assert record["latest_day_in_bucket"] == "2025/07/29/"
    assert err.splitlines()[0].startswith("stream gnss-ro holds no files")
    assert "2025/07/29/" in err, "the remedy does not name the newest day it holds"


def test_an_account_gate_is_a_refusal_not_an_argparse_error(capsys, tmp_path):
    """Exit 2 is argparse's, and the command line was correct."""

    gated = [s.name for s in STREAMS.values() if s.account_gated]
    assert gated, "no account-gated stream to check"
    code, out, err = _run([
        "obs", "fetch", "--stream", gated[0],
        "--start", "2026-09-05T00:00:00Z", "--end", "2026-09-07T00:00:00Z",
        "--out", str(tmp_path)], capsys)

    assert code == 1
    assert json.loads(out)["status"] == "ACCOUNT_GATED"
    assert "account" in err.splitlines()[0]


def test_a_stream_with_a_subscriber_publishes_a_command_for_it():
    """The reachability rule, applied to a door that writes no table."""

    subscribing = [s for s in STREAMS.values() if s.subscribes]
    assert subscribing, "no stream declares a subscriber"
    parser = cli.build_parser()
    for spec in subscribing:
        args = parser.parse_args([
            "obs", "subscribe", "--stream", spec.name, "--out", ".", "--seconds", "30"])
        assert args.func is obs_streams._cmd_subscribe
        assert args.seconds == 30


def test_fetch_refuses_a_subscriber_only_stream_and_names_the_command(
        monkeypatch, capsys, tmp_path):
    """A refusal that names no reachable command is where the subscriber was.

    NO SHIPPED STREAM IS IN THIS STATE TODAY, and that is a change rather than
    an omission: this test was written when the information-system door
    subscribed and had no decoder, and its table decoder has since landed, so
    `fetch` on it now gives a table.  The refusal it guards is still reachable
    the moment a stream arrives with a subscriber and no decoder, which is how
    every one of them starts, so the case is constructed rather than deleted.
    Deleting it would retire the guard along with the one stream that happened
    to need it.
    """

    subscriber = next(s for s in STREAMS.values() if s.subscribes)
    spec = dataclasses.replace(subscriber, decoder_built=False)
    monkeypatch.setitem(STREAMS, spec.name, spec)

    parser = cli.build_parser()
    args = parser.parse_args([
        "obs", "fetch", "--stream", spec.name,
        "--start", "2026-09-07T00:00:00Z", "--end", "2026-09-07T01:00:00Z",
        "--out", str(tmp_path)])
    code = args.func(args)
    captured = capsys.readouterr()

    assert code == 1, "a stream with no decoder answered with the missing-door code"
    assert json.loads(captured.out)["status"] == "NO_DECODER"
    assert f"obs subscribe --stream {spec.name}" in captured.err


def test_subscribe_refuses_an_archive_stream_by_name():
    """`--stream` on subscribe offers only the doors that subscribe, and the
    handler refuses one that does not even when reached directly."""

    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["obs", "subscribe", "--stream", "igra2", "--out", "."])

    args = parser.parse_args([
        "obs", "subscribe", "--stream",
        next(s.name for s in STREAMS.values() if s.subscribes), "--out", "."])
    args.stream = "igra2"
    with pytest.raises(SystemExit) as raised:
        args.func(args)
    assert "has no subscriber" in str(raised.value)
def test_the_subscribe_summary_carries_the_doors_own_keys(monkeypatch, capsys, tmp_path):
    """A counter the door does not write is named as missing, never printed as
    a null: four invented names once reported an archive of 68 messages from
    five centres as four nulls, at exit 0."""

    door_record = {
        "schema": "gpuwm-obs.wis2-subscribe.v1", "status": "READY",
        "broker": "example:8883", "listened_s": 30.0,
        "coverage": {"messages": 68, "payloads_downloaded": 68,
                     "messages_by_centre": {"a": 41, "b": 27}},
        "latency_behind_real_time_s": 3978,
    }

    def fake_door(name, arguments):
        assert arguments[0] == "subscribe"
        return door_record

    monkeypatch.setattr(obs_streams, "_run_door", fake_door)
    stream = next(s.name for s in STREAMS.values() if s.subscribes)
    code, out, _ = _run(["obs", "subscribe", "--stream", stream,
                         "--out", str(tmp_path), "--seconds", "30"], capsys)

    assert code == 0
    summary = json.loads(out)
    assert summary["coverage"]["messages"] == 68
    assert summary["coverage"]["messages_by_centre"] == {"a": 41, "b": 27}
    assert None not in summary.values(), "a counter the door never wrote came back null"
    assert "latency_basis" in summary["not_reported_by_the_door"]
