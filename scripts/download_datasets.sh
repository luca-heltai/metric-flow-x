#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="luca-heltai/metric-flow-x"
RELEASE_TAG="datasets-final-version-v1"
NETWORK_ASSET="blood-flow-datasets-v0.1.0.tar.gz"
NETWORK_SHA256="a9bef4aa466cb263d8c0bdf7503f175a659c7b22a3a812d2a9d95d3ed0988f47"
CEREBRAL_ASSET="cerebral-datasets-v1.tar.gz"
CEREBRAL_SHA256="49a1d6a6789975ffb4f8657f8a725e3ce5664e2997a7a239c041719329380b09"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DESTINATION="${1:-${REPOSITORY_ROOT}}"
ARCHIVE="$(mktemp "${TMPDIR:-/tmp}/blood-flow-datasets.XXXXXX")"

cleanup() {
  rm -f "${ARCHIVE}"
}
trap cleanup EXIT

# Refuse to extract unverified data if no checksum tool is installed.
if command -v shasum >/dev/null 2>&1; then
  CHECKSUM_COMMAND=(shasum -a 256 -c -)
elif command -v sha256sum >/dev/null 2>&1; then
  CHECKSUM_COMMAND=(sha256sum -c -)
else
  echo "Error: install shasum or sha256sum to verify the datasets." >&2
  exit 1
fi

download_and_extract() {
  local asset="$1"
  local checksum="$2"
  local url="https://github.com/${REPOSITORY}/releases/download/${RELEASE_TAG}/${asset}"

  echo "Downloading ${asset} from ${REPOSITORY} (${RELEASE_TAG})..."
  curl --fail --location --retry 3 --output "${ARCHIVE}" "${url}"
  printf '%s  %s\n' "${checksum}" "${ARCHIVE}" | "${CHECKSUM_COMMAND[@]}"
  mkdir -p "${DESTINATION}"
  tar -xzf "${ARCHIVE}" -C "${DESTINATION}"
}

download_and_extract "${NETWORK_ASSET}" "${NETWORK_SHA256}"
download_and_extract "${CEREBRAL_ASSET}" "${CEREBRAL_SHA256}"
echo "Datasets extracted to ${DESTINATION}."
