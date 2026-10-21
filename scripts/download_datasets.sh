#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="luca-heltai/metric-flow-x"
RELEASE_TAG="datasets-final-version-v1"
NETWORK_ASSET="metric_flow_x_datasets_si_v1.tar.gz"
NETWORK_SHA256="ee7c831a626109639c32edeea5085e5cb8a6cd15cd6dce31cbd75bf173787c1a"
CEREBRAL_ASSET="cerebral_datasets_v2.tar.gz"
CEREBRAL_SHA256="3f94081c6a256e2d341a45c51889e9090056273664cc9503cb22a2ffa98c5ee1"

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
