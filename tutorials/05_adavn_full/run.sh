#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA_DIR=${ADAVN_DATA_DIR:-"$ROOT_DIR/build/adavn_full"}
EXECUTABLE=${ADAVN_MEAN_EXECUTABLE:-"$ROOT_DIR/build/adavn_mean_flow"}
MODE=${1:-transient}
case "$MODE" in prepare|mean|transient|graph) ;; *) echo "Usage: $0 [prepare|mean|transient|graph]" >&2; exit 2;; esac
if [[ "$MODE" == transient || "$MODE" == graph ]]; then
  RUN_NAME=$MODE
  if [[ "$MODE" == transient ]]; then RUN_NAME=transient_10cycles; fi
  if [[ -e "$DATA_DIR/$RUN_NAME" ]]; then
    echo "Output directory already exists: $DATA_DIR/$RUN_NAME. Set ADAVN_DATA_DIR to a fresh directory." >&2
    exit 1
  fi
fi
python3 "$ROOT_DIR/tools/adan/download_sources.py" --data-dir "$ROOT_DIR/build/adan"
python3 "$ROOT_DIR/tools/adan/prepare_full.py" --data-dir "$ROOT_DIR/build/adan" --output-dir "$DATA_DIR"
python3 - "$ROOT_DIR" "$DATA_DIR" <<'PY'
import sys
from pathlib import Path
root, data = map(lambda s: Path(s).resolve(), sys.argv[1:])
for name in ("mean", "assembly"):
    template = root / "tutorials/05_adavn_full" / (name+".prm.in")
    (data/(name+".prm")).write_text(template.read_text().replace("@DATA_DIR@", str(data)))
template = (root/"tutorials/05_adavn_full/hdg.prm.in").read_text()
for name, mode, final in (("transient", "transient", "8"), ("graph", "steady network", "0")):
    text = template.replace("@DATA_DIR@", str(data)).replace("@RUN_MODE@", mode)
    output_name = "transient_10cycles" if name == "transient" else name
    text = text.replace("@FINAL_TIME@", final).replace("@OUTPUT_NAME@", output_name)
    text = text.replace("@INITIAL_SOURCE@", "steady network")
    inflow = "96.36e-6*(1-0.8*cos(2*3.141592653589793*t/0.8))" if name == "transient" else "96.36e-6"
    text = text.replace("@INFLOW_FUNCTION@", inflow)
    (data/(name+".prm")).write_text(text)
PY
if [[ "$MODE" == prepare ]]; then exit 0; fi
if [[ "$MODE" == transient || "$MODE" == graph ]]; then
  BLOOD_FLOW=${ADAVN_BLOOD_FLOW_EXECUTABLE:-"$ROOT_DIR/build/blood_flow"}
  "$BLOOD_FLOW" "$DATA_DIR/$MODE.prm" | tee "$DATA_DIR/$MODE.log"
  exit 0
fi
if [[ ! -x "$EXECUTABLE" ]]; then
  echo "Build target adavn_mean_flow and set ADAVN_MEAN_EXECUTABLE to its absolute path." >&2
  exit 1
fi
"$EXECUTABLE" "$DATA_DIR/mean.prm" | tee "$DATA_DIR/mean.log"
python3 "$ROOT_DIR/tools/adan/analyze_full.py" --data-dir "$DATA_DIR"
