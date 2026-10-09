#!/bin/bash
# The upper-air scorecard on one finished Arwen Global arm:
#   arwen_global_score_upper_air.sh <tree> <arm> [label]
# Reads <tree>/runs/<arm> (checkpoints + receipt) and writes
# <tree>/score/upper-air-<arm>.json, .png, -h24.png, .log and the
# representation floor upper-air-floor-<arm>.json.  Pairs every checkpoint
# with every reference product valid at its time: analyses (f000 files)
# and the reference model's forecast at that lead, each labelled in the
# rows.  Runs on the CPU only (numpy transform, Rust GRIB decode).
# Environment (all optional):
#   RUN_DIR, SCORE_DIR   default <tree>/runs/<arm>, <tree>/score
#   PY                   the interpreter that has woof global installed
#                        (default python3)
#   INSTRUMENT_TREE      a source checkout to score FROM instead of the
#                        installed package (its src/ goes on PYTHONPATH);
#                        an interpreter that cannot import the scorecard
#                        names the refusal
#   REFS                 root of reference products: <REFS>/gdas/*/gdas.*.f000
#                        and <REFS>/gfs/*/gfs.*.f0?? (REQUIRED)
#   INITIAL_ANALYSIS     the run's initial analysis GRIB (surface geopotential;
#                        SHA-256 checked against the receipt); default
#                        <tree>/cases/baseline-2026090100/gdas.t00z.pgrb2.0p25.f000
#   CACHE_DIR            decoded fields cache (default <SCORE_DIR>/upper-air-cache;
#                        the reference part is shared across arms of one grid)
#   WORKERS              parallel decode/synthesis processes (default 6)
#   OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, RAYON_NUM_THREADS
#                        per-worker thread caps (default 2, 2, 2, 3): six
#                        workers of uncapped numpy and Rust threads put a
#                        24-core host at load 130 and starved every tenant
set -u
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2} RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-3}
TREE=${1:?tree}; ARM=${2:?arm}; LABEL=${3:-$(basename "$TREE")-$ARM}
TREE=$(cd "$TREE" && pwd) || exit 2
export GPUWM_NO_LOCAL_GPU=1
# THE BREAKAGE THIS NAMES: this driver runs `python -m
# woof.globe.upper_air_scorecard`, so what has to be true is that the
# interpreter can IMPORT it.  The check used to be for a source file at a
# path this distribution does not have, which refused every installed
# copy of the package it ships in.  A checkout is still scoreable: point
# INSTRUMENT_TREE at it and its src/ goes ahead of site-packages.
if [ -n "${INSTRUMENT_TREE:-}" ]; then
  INSTRUMENT_TREE=$(cd "$INSTRUMENT_TREE" && pwd) || exit 2
  export ARWEN_TREE=$INSTRUMENT_TREE
  export PYTHONPATH=$INSTRUMENT_TREE/src${PYTHONPATH:+:$PYTHONPATH}
fi
export TMPDIR=${TMPDIR:-$TREE/tmp} WOOF_COMPOSE_SCRATCH=${WOOF_COMPOSE_SCRATCH:-$TREE/tmp}
mkdir -p "$TMPDIR"
PY=${PY:-python3}
if ! "$PY" -c "import woof.globe.upper_air_scorecard" >/dev/null 2>&1; then
  echo "SCORE-UPPER-AIR-REFUSED $LABEL: $PY cannot import woof.globe.upper_air_scorecard (install woof global into it, or set INSTRUMENT_TREE to a checkout)"
  exit 2
fi
RUN=${RUN_DIR:-$TREE/runs/$ARM}; SCORE=${SCORE_DIR:-$TREE/score}
if [ -z "${REFS:-}" ]; then
  echo "SCORE-UPPER-AIR-REFUSED $LABEL: REFS is unset (set it to the root holding gdas/ and gfs/ reference products)"
  exit 2
fi
INITIAL=${INITIAL_ANALYSIS:-$TREE/cases/baseline-2026090100/gdas.t00z.pgrb2.0p25.f000}
CACHE=${CACHE_DIR:-$SCORE/upper-air-cache}
WORKERS=${WORKERS:-6}
mkdir -p "$SCORE" "$CACHE"
if [ ! -f "$RUN/arwen-global-receipt.json" ]; then
  echo "SCORE-UPPER-AIR-REFUSED $LABEL: $RUN/arwen-global-receipt.json missing (the run has not finished)"; exit 2
fi
if [ ! -f "$INITIAL" ]; then
  echo "SCORE-UPPER-AIR-REFUSED $LABEL: initial analysis $INITIAL missing (set INITIAL_ANALYSIS)"; exit 2
fi
# The run's own initial analysis is a reference too: the hour-0 row
# against it is the representation floor read from the artifact.
REFLIST=$(ls "$INITIAL" "$REFS"/gdas/*/gdas.*.pgrb2.0p25.f000 "$REFS"/gfs/*/gfs.*.pgrb2.0p25.f0?? 2>/dev/null | sort -u)
if [ -z "$REFLIST" ]; then
  echo "SCORE-UPPER-AIR-REFUSED $LABEL: no reference products under $REFS"; exit 2
fi
echo "SCORE-UPPER-AIR-START $LABEL at $(date -u +%FT%TZ) tree=$TREE run=$RUN instrument=${INSTRUMENT_TREE:-installed} refs=$(echo "$REFLIST" | wc -l) files"
S=$(date +%s)
# 1. the scorecard at every checkpoint hour, the growth-curve chart and the hour-24 table
nice -n 10 "$PY" -m woof.globe.upper_air_scorecard --run-dir "$RUN" --label "$LABEL"   --reference $REFLIST --initial-analysis "$INITIAL" --cache-dir "$CACHE" --workers "$WORKERS"   --out "$SCORE/upper-air-$ARM.json" --chart "$SCORE/upper-air-$ARM.png" --title "$LABEL upper-air scorecard"   --table "$SCORE/upper-air-$ARM-h24.png" --table-hour 24 --table-kind analysis   > "$SCORE/upper-air-$ARM.log" 2>&1
RC=$?
echo "SCORE-UPPER-AIR $LABEL rc=$RC"
# 2. the representation floor of the initial analysis on this run's grid and levels
#    (vertical round trip, spectral truncation, the cold start read back as a checkpoint)
nice -n 10 "$PY" -m woof.globe.upper_air_scorecard --run-dir "$RUN" --floor "$INITIAL"   --out "$SCORE/upper-air-floor-$ARM.json" > "$SCORE/upper-air-floor-$ARM.log" 2>&1
echo "SCORE-UPPER-AIR-FLOOR $LABEL rc=$?"
echo "SCORE-UPPER-AIR-DONE $LABEL wall=$(( $(date +%s) - S ))s at $(date -u +%FT%TZ)"
exit $RC
