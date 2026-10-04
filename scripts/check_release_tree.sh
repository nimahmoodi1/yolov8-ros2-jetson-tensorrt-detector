#!/usr/bin/env bash
set -euo pipefail

echo "Checking tracked release tree..."

bad_files="$(
  git ls-files |
  grep -E '(^|/)\.env$|\.(engine|plan|onnx|partial|zip)$|(^|/)(__pycache__|build|install|log)/' \
  || true
)"

if [[ -n "${bad_files}" ]]; then
  echo "ERROR: generated/private files are tracked:"
  echo "${bad_files}"
  exit 1
fi

stale_docs="$(
  git grep -nE \
  'Publishing checklist|Before making this repository public|vision_detection_container_jetson_v[0-9]+|vision_feedback_ros2_package_v[0-9]+|/home/(flyby|intplatform)' \
  -- . 2>/dev/null |
  grep -v '^scripts/check_release_tree\.sh:' \
  || true
)"

if [[ -n "${stale_docs}" ]]; then
  echo "ERROR: stale public documentation found:"
  echo "${stale_docs}"
  exit 1
fi

echo "PASS: release tree is clean"
