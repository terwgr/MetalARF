# WOOF global observation streams: interface notes and measurements

The observation front doors of the global data-assimilation system
(2026-09-06). This page records every interface decision the observation
streams carry, the measurements behind the numbers it states, and what is
not built, so the ensemble, the doors and the satellite operators build
against committed code.

## 1. The contract in one paragraph

Every observation stream is fetched and decoded by a Rust front door
(`tools/rustwx/crates/rw-obs`: `rw_igra2`, `rw_amv`, `rw_ndbc`,
`rw_gnssro`, `rw_asos` with `networks`, `fetch --networks`, `table` and
`awc`, and `rw_wis2`) into ONE neutral table, `gpuwm-obs.table.v2`, whose
first ten columns are the v1 table every reader of the tree already takes
and whose five bookkeeping columns say what a number is, which synoptic
hour it is filed under, when its source published it, when this system
first held it and which source object it came from.
`woof.globe.obs_table.load_obs` reads v2 and v1 files without
conversion, so `woof global assimilate --obs`, the cycle door and the
ensemble filter's `batches_from_rows` take them unchanged.
`woof.globe.obs_streams` is the roster of streams with their doors,
errors, measured latencies and latency classes, the hourly cut under an
information cutoff, and the manifests; `woof obs streams` is its door.
`woof.globe.obs_operators` carries the operators the v1 door does
not: refractivity at a tangent height (as a column function and as the
member-batched operator the ensemble filter's `PointObs.operator` contract
takes) and the wind read at an assigned pressure.

## 2. Decisions

1. **One table, versioned, read by everyone.** `gpuwm-obs.table.v2`
   (`obs_table.TABLE_HEADER`): `source, station_id, latitude_deg,
   longitude_deg, elevation_m, level_pa, valid_time, variable, value,
   error, measurement, nominal_time, published_time, received_time,
   revision`.  The Rust writer (`rw-obs/src/table.rs`) and the Python
   reader (`obs_table.decode_neutral_csv`) carry the same header text and
   the same vocabulary, bounds and per-stream errors, transcribed once.  A
   v1 file (ten columns) reads with the bookkeeping empty.  Adding a column
   is a new version, never a silent widening.
2. **The measurement column is the acceptance contract's definition.**
   `obs_table.MEASUREMENT_TABLE`: `station_pressure_from_altimeter` (METAR,
   the ISA inversion of the altimeter setting at the station elevation,
   exact by the setting's definition), `station_pressure` (a sounding's
   surface level), `sea_level_pressure` (NDBC PRES, anchored at 0 m and
   never mistaken for a station pressure), `screen_temperature_2m`,
   `screen_dewpoint_2m`, `platform_temperature` and `platform_dewpoint`
   (buoy sensors at about 4 m), `anemometer_wind_10m`,
   `anemometer_wind_5m_reduced_to_10m` (buoys, neutral log law, factor
   1.064), `sonde_level`, `amv_assigned_pressure`,
   `ro_refractivity_tangent_point`.  The IEM `mslp` column (the METAR
   SLP group, reduced by the station's own method) is not written: the
   station pressure is the direct measurement and the analysis compares at
   the station height.
3. **Observations at their own times.**  A row's `valid_time` is the
   instant it was measured and nothing moves it to an analysis hour.  A
   radiosonde level's time is the header's release time (RELTIME, read on
   the nominal date and moved a day when it lands more than twelve hours
   from the nominal) plus the level's elapsed time (ETIME, MMMSS), with the
   nominal hour in `nominal_time`; a level without an elapsed time keeps
   the nominal hour and is counted (`levels_at_nominal_time`).  On the
   case window 118,632 levels carry their own time and 93,067 keep the
   nominal hour (the archive has no ETIME for them); the levels with a time
   sit 20 to 90 minutes before the nominal hour.  The archive carries no
   per-level position, so every level sits at the station (stated).  A
   motion vector is at its image mid-point; a buoy report at its own
   minute; a METAR at its observation minute.
4. **Causal bookkeeping.**  `published_time` is what the source states
   (an archive file's `Last-Modified`, a granule's creation stamp from its
   filename, the AWC cache's `Last-Modified`), `received_time` is the fetch
   record's `fetched_at` for the object the row came from, `revision` is
   the first twelve hex digits of the source object's SHA-256 (with `:COR`
   appended for a corrected METAR in the AWC cache).  Rows decoded from
   files whose arrival was not recorded (the case window's bulk prefetch
   of 2026-09-06 00:33 to 00:38 UTC) carry an empty `received_time` and the
   reader counts them `latency_unverified`; the design's "a historical
   case with unknown arrival times is labelled latency unverified" is that
   count in the hours manifest.
5. **The information cutoff is a selection on `received_time`.**
   `obs_streams.apply_information_cutoff(rows, cutoff)` keeps a row whose
   receipt is at or before the cutoff, drops and counts one received later
   (`after_cutoff`), keeps and counts one without a receipt time.
   `woof obs streams hours --cutoff analysis` applies each hour's own
   instant.  Measured on the case: with `--cutoff analysis` every motion
   vector row (received 2026-09-06 01:01 from the cache walk) is dropped
   as after-cutoff and every prefetched METAR, sonde and buoy row is kept
   as unverified, which is the correct statement about a case fetched
   five days late; the hour set of record for the case therefore has no
   cutoff and carries the unverified counts (
   `hours2/full/hours.json`).
6. **The external analysis is an anchor source, not a stream.**
   `obs_streams.ANCHOR_SOURCES` (`gdas`, `ifs-open-data`) record the
   object, the mapping row that reads it, and its measured availability
   latency (`AnchorSource.record` writes `gpuwm-obs.anchor.v1` with the
   source cycle, publication and receipt times and the latency); no
   pseudo-observation rows are produced anywhere.  The analysis lanes
   implement the weak low-pass constraint and record its increment
   separately; it constrains selected scales only.  Measured 2026-09-06
   (object LastModified minus cycle time): GDAS f000 6 h 50 min to 7 h 33
   min behind (six cycles 2026-08-31 18Z to 09-02 00Z: 7:21, 7:08, 6:50,
   6:54, 7:23, 6:59; 2026-09-05 12Z 6:53, 18Z 7:33); IFS open data step 0
   7 h 34 min behind at 00Z and 12Z (2026-08-31, 09-01, 09-05).
7. **Latency classes follow the measurement.**  `fast` (feeds the hourly
   analysis): IEM METAR (a live fetch of ten networks answered reports up
   to 137 s old; the earlier probe 443 s), the AWC cache (rewritten every
   minute, 15 s old when read), NDBC (file Last-Modified minus the latest
   report: median 1,242 s, maximum 2,135 s over 61 live platforms), GOES
   motion vectors (bucket LastModified minus scan end: 619 to 1,861 s,
   median 921 s over 365 granules), WIS2 (ten minutes on
   globalbroker.meteo.fr, 2026-09-06 02:03 to 02:13 UTC: 2,920
   notifications from 29 centres, 292 a minute, synop 2,470, temp 400,
   ship 50; publication delay, pubtime minus the report's datetime, median
   388 s and minimum 160 s with a tail of resends up to 26 h; transport
   delay, receipt minus pubtime, median 158 s and maximum 558 s; 2,918
   payloads of 1.8 MB archived, 2,891 with the publisher's sha512 verified,
   23 failing the digest (11 of them links a centre labelled canonical that
   point at OSCAR station pages, text/html, not data; 12 BUFR payloads from
   fr-meteofrance, ru-roshydromet, ar-smn, my-metmalaysia and th-tmd), 4
   without a digest; ma-marocmeteo's 105 pubtimes run ahead of the
   receiving clock and are recorded as negative transport delays; 360 of
   us-noaa-nws's TEMP notifications are resends more than two hours after
   the report's time).  `replay` (the delayed replay at its measured lag): IGRA2, whose
   year-to-date files rebuild once a day near 21:36 UTC and were 77,802 to
   120,998 s (21.6 to 33.6 h) behind the latest sounding they held.
   `retrospective`: the AWS `gnss-ro-data` archive, whose COSMIC-2 UCAR
   collection ends at 2025-07-29 with no 2026 object in any collection.
   `gated`: MADIS aircraft and the UCAR CDAAC near-real-time occultation
   feed, both needing accounts, reported and never fetched.
8. **Quality control is counted, never silent, and lives in the doors.**
   Gross limits (the vocabulary's bounds), sentinels, a dewpoint above its
   temperature, a variable-direction wind (direction 0 with speed), a limb
   past 68 degrees, a DQF other than 0, a pressure outside 100 to 1050 hPa,
   a repeated pressure inside a sounding, a duplicate station and instant,
   a superrefraction layer: each is a counter in the door's record.
   Thinning: the AMV door keeps one vector per 0.5 degree box and 50 hPa
   layer per satellite and hour (the smallest local zenith angle wins);
   `obs_streams.thin_to_grid` is a pre-thinning to a stated grid (one row
   per source, variable, cell and 0.1 ln p layer, nearest the instant; a
   buoy under a METAR station never competes with it) and the ensemble
   filter thins again to its own cell at analysis time.  The background
   check against `sqrt(error^2 + spread^2)` is the filter's (the ensemble
   lane's quality-control order) because it needs the members.
9. **The operators the filter cannot evaluate itself ship with their
   calibration.**  `obs_operators.RefractivityOperator(transform,
   vertical, terrain)` is a `PointObs.operator(members, batch) -> (R, n)`:
   per member the surface pressure, potential temperature and vapour
   columns are synthesised at the batch's distinct points (the tree's
   `sample_scalar`), the full-level pressures follow the hybrid coordinate,
   heights the hypsometric integration with the model's gravity and gas
   constant, and inside the layer holding a target `ln p` is linear in
   height (the layer-mean virtual temperature the integration assumed) with
   temperature and vapour linear in `ln p`, so the dry term `77.6 p/T` is
   captured exactly in `p`.  A member missing `log_surface_pressure`,
   `theta` or `qv` is refused by the field's name; a batch of another
   variable is refused by name; a target outside the column is NaN, never
   extrapolated.  The AMV wind read is the aloft wind operator the ensemble
   package already evaluates (linear in `ln p` to the assigned pressure);
   its calibration is recorded here beside the refractivity's.  The
   bending-angle operator the design prefers as the numerical reference is
   not built (section 5).
10. **The v1 door refuses what it cannot evaluate.**  `refractivity_n`
    joined the vocabulary; `assimilate.VARIABLES_WITHOUT_OPERATOR` names it
    and `_table_quality_control` refuses its rows by name
    (`REJECTION_BREAKAGE["no_operator"]`) before a NaN innovation could
    reach the spreading sums; `DEFAULT_BACKGROUND_ERRORS` covers the
    vocabulary as its constructor demands.
11. **WIS2 is subscribed, archived and measured; not decoded.**  `rw_wis2`
    frames MQTT 3.1.1 by hand over rustls (no MQTT crate is in the vendor
    closure), connects to a global broker with the public `everyone`
    credentials, subscribes to
    `cache/a/wis2/+/data/core/weather/surface-based-observations/#`,
    archives every notification message on receipt and downloads each
    payload once with its SHA-256 and the publisher's integrity digest
    checked, and records coverage per centre and topic with the
    publication delay (pubtime minus the report's datetime) and the
    transport delay (receipt minus pubtime).  The BUFR payloads are
    archived and counted as bytes; the reader into the neutral table is
    the named open item (section 5).  The door is reached with
    `woof global obs subscribe --stream wis2 --out obs --seconds 600`,
    which is a separate command from `obs fetch` because it produces an
    archive and a coverage report rather than a table; `obs fetch
    --stream wis2` refuses and names it.
12. **The Aviation Weather Center cache is the complementary METAR
    source.**  `rw_asos awc` fetches the gzip CSV (the last hour,
    worldwide), keeps the raw object with its digest beside the table,
    converts through the same arithmetic as the IEM route (from Celsius),
    and marks a corrected report's revision `COR`.
13. **A radiosonde's surface level is anchored at the station, or
    dropped.**  The surface level writes rows without a pressure, so
    their `elevation_m` IS the anchor the surface operators evaluate
    at.  The archive gives 72 percent of the case's surface levels no
    geopotential height (1,221 of 1,690), and the first cut of the
    tables stamped those rows with the ISA height of the surface
    pressure, tens to hundreds of metres from the station (Dar-El-Beida
    at 25 m read -6.2 m; Albuquerque at 1,619 m would read the ISA
    height of its 840 hPa), and dropped their station pressure row.
    `rw_igra2 fetch` downloads the IGRA2 station list (id, position,
    elevation; 2,932 stations, 2,701 with an elevation; 261 kB) beside
    the files with its digest and Last-Modified, and `table` reads it
    (beside the files, or `--station-list`): a height-less surface
    level is anchored at the list's elevation and its station pressure
    row is written; a level with neither is dropped and counted
    (`surface_levels_dropped_without_anchor`), never stamped.  The
    case re-decoded: 469 surface levels anchored by the archive's
    height, 1,095 by the list, 126 dropped (stations the list carries
    with elevation -998.8, among them AYM00089664, BDM00078016 and
    eight CHM stations), 1,564 station pressure rows where there were
    469.  Aloft levels keep the ISA stamp as a label only (their
    anchor is `level_pa`; 76,933 in the case).

## 3. What the other lanes read today

- The hourly tables of the case, `hours2/full/<YYYY-MM-DDTHH>.csv`
  (31 hours, 2026-08-31 18Z to 2026-09-02 00Z, 1.4 GB, with `hours.json`)
  and the same thinned to the T127 Gaussian grid (192 x 384) under
  `hours2/t127/` (563 MB).  Per window: 32,000 to 37,000 METAR
  rows from about 4,300 to 7,000 stations, 2,400 to 2,900 buoy rows,
  75,000 to 280,000 motion-vector rows, and 140,000 to 185,000 sounding
  rows in the 00Z and 12Z hours (the 18Z hour is a half window because
  the tables begin at 18:00).  The first hour a lane needs is one file;
  `load_obs` reads it.
- The stream tables, `tables2/`: `metar.csv` (1,010,500
  rows, 212,355 reports from 4,606 stations, 93.7 MB), `igra2.csv`
  (698,842 rows, 1,814 soundings, 107.1 MB; re-decoded with the station
  list, decision 13) and `igra2-mandatory.csv` (102,241 rows),
  `ndbc.csv` (77,367 rows, 24,297 hourly reports from 829 platforms),
  `amv/<hour>.csv` (31 tables, 6,466,820 rows, 3,233,410 vectors kept of
  11,182,455 in 353 granules, 20 empty products), `gnssro-2025-sample.csv`
  (443 rows from three 2025-07-29 occultations, the decoder proven on the
  only data the bucket holds).  Every table has its `.json` record with
  counters, digests and latency.
- The live probes, `live/`: IGRA2 four stations
  (`igra2/table.csv`, 50,740 rows with published and received times),
  NDBC 70 platforms (`ndbc/table.csv`, 9,959 rows; 8 platforms answered
  404 and are named in `fetch.json`), METAR ten networks (`metar/`), the
  266-network list (`networks.json`), the WIS2 subscription (`wis2/run1/`),
  the AWC cache (`awc/`), and the anchor probe (`anchor/probe.tsv`).
- The Python interface: `obs_table.ObsRow` (with `measurement`,
  `nominal_time`, `published_time`, `received_time`, `revision`,
  defaulting empty), `obs_streams.STREAMS`, `ANCHOR_SOURCES`,
  `apply_information_cutoff`, `cut_hours`, `thin_to_grid`,
  `obs_operators.RefractivityOperator`.

## 4. Measurements

Fetch of the case window (2026-08-31 17Z to 09-02 01Z, 2026-09-06
00:33 to 00:38 UTC; every file with SHA-256 in the per-stream
`SHA256SUMS`):

| stream | objects | bytes | wall | per hour of the window |
|---|---|---|---|---|
| IGRA2 year-to-date | 807 zips + the station list (261 kB) | 631.3 MB | 103 s (8 parallel) | 1,814 soundings in 31 h; a sounding is about 385 rows |
| IEM METAR, 266 networks | 25 requests of 10 networks | 18.6 MB | 90 s, paced 2 s | about 7,000 reports and 32,600 rows an hour, 3.0 MB of table |
| GOES-18 and GOES-19 DMWF | 377 granules (353 with vectors) | 326.4 MB | 330 s | 12 granules, 10.5 MB, 360,000 vectors, 104,000 kept an hour |
| NDBC realtime2 | 943 files + station table | 526.4 MB (45-day files) | 46 s (6 parallel) | 800 hourly reports, 2,500 rows an hour |
| GDAS f000 (anchor) | 6 cycles | 2.82 GB | 248 s | 470 MB per 6 h cycle |
| GNSS-RO | 0 in the window | 0 | | bucket ends 2025-07-29 |

Decode (the v2 doors, one core, `decode2.log`): IGRA2 807 zips to
698,842 rows in 6 s; NDBC 943 files to 77,367 rows in 2 s; METAR
224,464 CSV rows to 1,010,500 table rows in 1 s; AMV 353 granules to
6,466,820 rows in 16 s; the hourly cut of 8.25 million rows in 78 s
(Python).

Latency, measured 2026-09-06 01:41 to 02:13 UTC: section 2, decision 7;
the AWC cache read at 02:07 held 4,925 reports (13 corrected) whose latest
was 134 s old, the cache rewritten 69 s after its latest report.
Operator calibration (`python -m woof.globe.obs_operators
calibrate`): dry ISA and moist tropical columns on 40 levels against
400-level truths, read-back median 0.038 and 0.051 percent, maximum 0.42
and 0.54 percent in the one layer straddling a lapse-rate kink; the 1 K
fixed-height response within a median 0.0005 and 0.0009 N/T of the
truth's (maximum 0.10 and 0.07 at the kink layers), meeting the analytic
`-N_dry/T - 2 N_wet/T` at the lowest target within 5.5 and 8.0 percent;
the member operator equal to the column function to 3e-13 N; an
unchanged column moves exactly nothing; the wind read exact on a
ln p-linear profile and within 0.48 m/s on a 60 m/s jet at 25 hPa spacing.

## 5. Not built, with reasons

- **A BUFR decoder into the neutral table.**  WIS2 payloads are BUFR
  (SYNOP 307080, TEMP 309052 and the marine templates); a reader needs the
  table B and D descriptors and the compressed-data path, several days of
  Rust.  The subscriber archives every payload with its digest so nothing
  is lost meanwhile, and the record says `decoded_into_table: false`.
- **A bending-angle operator.**  The design prefers it as the numerical
  reference over retrieved refractivity; it needs the ray integral through
  the model refractivity with an established implementation to compare
  against.  Refractivity at the tangent point ships calibrated; the bucket
  holds no 2026 occultation to exercise either on the case.
- **A TEMP (FM-35) text decoder** for the NWS upper-air feed, which is
  three minutes behind the nominal hour; IGRA2 covers the same soundings a
  day later and is the sounding stream of record.
- **Himawari motion vectors**: not reached (no public bucket with the
  product was located in the window of this work); GOES-18 and GOES-19
  cover the Pacific from 137 W.
