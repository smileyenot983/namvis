#!/usr/bin/env bash
# Image-only replacement for the original Infinity3D/inference_geo/video.sh.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${REPO_ROOT}/inference/video.sh" "$@"
