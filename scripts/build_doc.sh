#!/bin/sh

# Build the Doxygen XML consumed by Breathe/Exhale, then render the Sphinx site.
# Run this script from any directory; all paths are resolved from the repository.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
VENV_DIR="${ROOT_DIR}/env"
REQUIREMENTS_FILE="${ROOT_DIR}/doc/requirements.txt"
BUILD_DIR="${ROOT_DIR}/build/docs"
DOXYGEN_DIR="${BUILD_DIR}/doxygen"
SITE_DIR="${BUILD_DIR}/site"
DOCS_SOURCE_DIR="${ROOT_DIR}/doc"
API_STUB_DIR="${DOCS_SOURCE_DIR}/api"
DOXYFILE="${ROOT_DIR}/doc/Doxyfile"
TMP_DOXYFILE="${BUILD_DIR}/Doxyfile"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found in PATH." >&2
  exit 1
fi
if ! command -v doxygen >/dev/null 2>&1; then
  echo "doxygen not found in PATH." >&2
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi
# shellcheck source=/dev/null
. "${VENV_DIR}/bin/activate"
python3 -m pip install -r "${REQUIREMENTS_FILE}"

if ! command -v sphinx-build >/dev/null 2>&1; then
  echo "sphinx-build not found after installing documentation requirements." >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}" "${DOXYGEN_DIR}"
rm -rf "${SITE_DIR}" "${API_STUB_DIR}"

# Doxygen resolves relative INPUT values against its invocation context. Append
# absolute overrides so the pipeline is independent of the caller's directory,
# and keep XML exactly where doc/conf.py expects it.
cp "${DOXYFILE}" "${TMP_DOXYFILE}"
cat >> "${TMP_DOXYFILE}" <<EOF
OUTPUT_DIRECTORY = ${DOXYGEN_DIR}
INPUT = ${ROOT_DIR}/source ${ROOT_DIR}/include ${ROOT_DIR}/README.md
STRIP_FROM_PATH = ${ROOT_DIR}
GENERATE_HTML = NO
GENERATE_XML = YES
XML_OUTPUT = xml
EOF
(
  cd "${ROOT_DIR}"
  doxygen "${TMP_DOXYFILE}"
)

sphinx-build -b html -W "${DOCS_SOURCE_DIR}" "${SITE_DIR}"
echo "Documentation site generated in ${SITE_DIR}"
