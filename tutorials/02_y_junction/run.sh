#!/usr/bin/env bash
# Run the source-backed Y-junction tutorial after validating all local inputs.
set -euo pipefail

TUTORIAL_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$TUTORIAL_DIR/../.." && pwd)
NETWORK="$TUTORIAL_DIR/network.vtk"
PARAMETER_INPUT=${1:-"$TUTORIAL_DIR/parameters.prm.in"}
EXECUTABLE=${METRIC_FLOW_X_EXECUTABLE:-"$ROOT_DIR/build/metric_flow_x"}

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to validate network.vtk; refusing to claim a run." >&2
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
  echo "       No solver run was performed; this script will not claim success." >&2
  exit 3
fi

# CMake expands @SOURCE_DIR@ in parameters/ templates.  This local tutorial
# template is also useful directly, so resolve the token into a temporary file.
RESOLVED_PARAMETER="$PARAMETER_INPUT"
TEMP_PARAMETER=""
if grep -q '@SOURCE_DIR@' "$PARAMETER_INPUT"; then
  TEMP_PARAMETER=$(mktemp "${TMPDIR:-/tmp}/y-junction-parameters.XXXXXX.prm")
  trap 'rm -f "$TEMP_PARAMETER"' EXIT
  sed "s|@SOURCE_DIR@|$ROOT_DIR|g" "$PARAMETER_INPUT" > "$TEMP_PARAMETER"
  RESOLVED_PARAMETER="$TEMP_PARAMETER"
fi

if [[ "$RESOLVED_PARAMETER" == *.in ]]; then
  echo "ERROR: unresolved parameter template (expected @SOURCE_DIR@ expansion): $RESOLVED_PARAMETER" >&2
  exit 2
fi

# The configured template uses a repository-relative output path.  Make that
# path explicit and run from the repository root regardless of caller cwd.
mkdir -p "$ROOT_DIR/output/y_junction"
cd "$ROOT_DIR"
echo "Running Y-junction tutorial with $RESOLVED_PARAMETER"
exec "$EXECUTABLE" "$RESOLVED_PARAMETER"
