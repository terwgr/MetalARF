"""IGRA2 soundings to the door's level table (``igra2-levels-csv``).

Reads the IGRA2 v2 station data files (``*-data-beg2026.txt.zip`` as the
archive serves them, or the unzipped text), keeps the soundings at the
nominal times asked for, and writes one CSV row per station, nominal time
and pressure level in the ``igra2-levels-csv`` layout the observation table
declares (``woof.globe.obs_table``): station, valid (the nominal
hour), lon, lat, gph_m (the level's geopotential height), pressure_hpa,
temp_c, dewpoint_c (temperature minus the archive's dewpoint depression),
wdir_deg, wspd_m_s.  ``--levels-hpa`` selects the pressure levels (default
the mandatory levels 1000 to 300 hPa); ``--variables`` selects which
columns are filled (``all`` or a comma list of temperature, dewpoint,
wind), the others left blank so the door finds them underivable and offers
nothing for them.  A level whose value is the archive's missing sentinel
(-9999, -8888) is blank.  A JSON record beside the CSV counts the files,
soundings, levels and rows and carries the SHA-256 of the CSV.

Fixed columns of an IGRA2 v2 data record (1-based): header ``#`` ID 2-12,
year 14-17, month 19-20, day 22-23, hour 25-26, release time 28-31, level
count 33-36, latitude 56-62 and longitude 64-71 in 1e-4 degrees; level
lines LVLTYP1 1, LVLTYP2 2, ETIME 4-8, PRESS 10-15 (Pa), GPH 17-21 (m),
TEMP 23-27 (0.1 C), RH 29-33 (0.1 percent), DPDP 35-39 (0.1 C), WDIR 41-45
(degrees), WSPD 47-51 (0.1 m/s).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

MISSING = {-9999, -8888}
MANDATORY_HPA = (1000.0, 925.0, 850.0, 700.0, 500.0, 400.0, 300.0)
HEADER = ("station", "valid", "lon", "lat", "gph_m", "pressure_hpa",
          "temp_c", "dewpoint_c", "wdir_deg", "wspd_m_s")
VARIABLES = ("temperature", "dewpoint", "wind")


def _int(text: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return None if value in MISSING else value


def parse_igra2(text: str, nominal: set[str]) -> list[dict]:
    """Soundings at the wanted nominal instants (ISO ``...Z``) as dicts
    with station, nominal, lat, lon, release and levels ``{pressure_pa:
    {gph, temp_c, dpdp_c, wdir, wspd}}``."""
    soundings: list[dict] = []
    current = None
    for line in text.splitlines():
        if line.startswith("#"):
            current = None
            station = line[1:12].strip()
            try:
                year, month, day, hour = (int(line[13:17]), int(line[18:20]),
                                          int(line[21:23]), int(line[24:26]))
            except ValueError:
                continue
            if hour == 99:
                continue
            stamp = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:00:00Z"
            if stamp not in nominal:
                continue
            lat = _int(line[55:62])
            lon = _int(line[63:71])
            if lat is None or lon is None:
                continue
            current = {
                "station": station, "nominal": stamp,
                "latitude": lat / 1.0e4, "longitude": lon / 1.0e4,
                "release": line[27:31].strip(), "levels": {},
            }
            soundings.append(current)
            continue
        if current is None or len(line) < 51:
            continue
        press = _int(line[9:15])
        if press is None or press <= 0:
            continue
        gph = _int(line[16:21])
        temp = _int(line[22:27])
        dpdp = _int(line[34:39])
        wdir = _int(line[40:45])
        wspd = _int(line[46:51])
        current["levels"][press] = {
            "gph_m": gph,
            "temp_c": None if temp is None else temp / 10.0,
            "dpdp_c": None if dpdp is None else dpdp / 10.0,
            "wdir_deg": wdir,
            "wspd_m_s": None if wspd is None else wspd / 10.0,
        }
    return soundings


def rows_for(sounding: dict, levels_hpa, variables) -> list[dict]:
    rows = []
    want_t = "temperature" in variables
    want_td = "dewpoint" in variables
    want_w = "wind" in variables
    for hpa in levels_hpa:
        level = sounding["levels"].get(int(round(hpa * 100.0)))
        if level is None or level["gph_m"] is None:
            continue
        temp = level["temp_c"]
        dewpoint = (None if temp is None or level["dpdp_c"] is None
                    else temp - level["dpdp_c"])
        wdir, wspd = level["wdir_deg"], level["wspd_m_s"]
        fields = {
            "temp_c": temp if want_t else None,
            "dewpoint_c": dewpoint if want_td else None,
            "wdir_deg": wdir if want_w else None,
            "wspd_m_s": wspd if want_w else None,
        }
        if all(v is None for v in fields.values()):
            continue
        rows.append({
            "station": sounding["station"],
            "valid": sounding["nominal"].replace("T", " ")[:16],
            "lon": f"{sounding['longitude']:.4f}",
            "lat": f"{sounding['latitude']:.4f}",
            "gph_m": f"{level['gph_m']:.1f}",
            "pressure_hpa": f"{hpa:g}",
            **{k: ("" if v is None else f"{v:.2f}") for k, v in fields.items()},
        })
    return rows


def read_source(path: Path) -> str:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if not n.endswith("/")]
            return "\n".join(archive.read(n).decode("ascii", "replace") for n in names)
    return path.read_text(encoding="ascii", errors="replace")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--zips", required=True, help="directory of IGRA2 station files (zip or txt)")
    parser.add_argument("--nominal", action="append", required=True,
                        help="nominal instant to keep, ISO-8601 ...Z (repeatable)")
    parser.add_argument("--levels-hpa", default=",".join(f"{v:g}" for v in MANDATORY_HPA))
    parser.add_argument("--variables", default="all",
                        help="all, or a comma list of temperature, dewpoint, wind")
    parser.add_argument("--out", required=True, help="CSV path; a .json record is written beside it")
    args = parser.parse_args(argv)

    nominal = set()
    for stamp in args.nominal:
        moment = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        nominal.add(moment.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    levels = tuple(float(v) for v in args.levels_hpa.split(",") if v.strip())
    variables = (set(VARIABLES) if args.variables == "all"
                 else {v.strip() for v in args.variables.split(",") if v.strip()})
    unknown = variables - set(VARIABLES)
    if unknown:
        raise SystemExit(f"unknown variables {sorted(unknown)}; the choices are {VARIABLES}")

    root = Path(args.zips)
    files = sorted(p for p in root.iterdir() if p.suffix in {".zip", ".txt"})
    if not files:
        raise SystemExit(f"no IGRA2 files under {root}")
    rows: list[dict] = []
    soundings = 0
    levels_seen = 0
    by_nominal: dict[str, int] = {}
    for path in files:
        for sounding in parse_igra2(read_source(path), nominal):
            soundings += 1
            levels_seen += len(sounding["levels"])
            new = rows_for(sounding, levels, variables)
            if new:
                by_nominal[sounding["nominal"]] = by_nominal.get(sounding["nominal"], 0) + 1
            rows.extend(new)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=HEADER, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    text = buffer.getvalue()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    record = {
        "source": "IGRA2 v2 station data files",
        "files": len(files), "soundings_at_nominal": soundings,
        "levels_read": levels_seen, "rows": len(rows),
        "soundings_with_rows_by_nominal": by_nominal,
        "nominal": sorted(nominal), "levels_hpa": list(levels),
        "variables": sorted(variables), "csv": str(out),
        "csv_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
