#!/usr/bin/env bash
# Validate and, when explicitly configured, run the ADAN56 tutorial input.
set -euo pipefail

TUTORIAL_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$TUTORIAL_DIR/../.." && pwd)
NETWORK="$TUTORIAL_DIR/network.vtk"
PARAMETER_INPUT=${1:-"$TUTORIAL_DIR/parameters.prm.in"}
EXECUTABLE=${METRIC_FLOW_X_EXECUTABLE:-"$ROOT_DIR/build/metric_flow_x"}

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to validate network.vtk; no solver run was performed." >&2
  exit 2
fi
if [[ ! -f "$NETWORK" ]]; then
  echo "ERROR: missing tutorial network: $NETWORK" >&2
  exit 2
fi
if ! python3 "$ROOT_DIR/tools/validate_vtk_network.py" "$NETWORK" >/dev/null; then
  echo "ERROR: network validation failed; refusing to run the tutorial." >&2
  exit 2
fi
if [[ ! -f "$PARAMETER_INPUT" ]]; then
  echo "ERROR: parameter file does not exist: $PARAMETER_INPUT" >&2
  exit 2
fi
if [[ ! -x "$EXECUTABLE" ]]; then
  echo "ERROR: configured metric_flow_x executable not found or not executable: $EXECUTABLE" >&2
  echo "       Configure and build the project, or set METRIC_FLOW_X_EXECUTABLE." >&2
  echo "       No solver run was performed; this script does not claim runtime success." >&2
  exit 3
fi

RESOLVED_PARAMETER="$PARAMETER_INPUT"
TEMP_PARAMETER=""
if grep -q '@SOURCE_DIR@' "$PARAMETER_INPUT"; then
  TEMP_PARAMETER=$(mktemp "${TMPDIR:-/tmp}/adan56-parameters.XXXXXX.prm")
  trap 'rm -f "$TEMP_PARAMETER"' EXIT
  sed "s|@SOURCE_DIR@|$ROOT_DIR|g" "$PARAMETER_INPUT" > "$TEMP_PARAMETER"
  RESOLVED_PARAMETER="$TEMP_PARAMETER"
fi

if [[ "$RESOLVED_PARAMETER" == *.in ]]; then
  echo "ERROR: unresolved parameter template (expected @SOURCE_DIR@ expansion): $RESOLVED_PARAMETER" >&2
  exit 2
fi

mkdir -p "$ROOT_DIR/output/adan56"
cd "$ROOT_DIR"
echo "Running ADAN56 tutorial with $RESOLVED_PARAMETER"
exec "$EXECUTABLE" "$RESOLVED_PARAMETER"
