#!/usr/bin/env bash
# Download NVIDIA's 10 teleop demonstrations for Isaac-Stack-Cube-Franka-IK-Rel-v0 into this directory and verify them.
# Usage: datasets/fetch.sh
set -euo pipefail
cd "$(dirname "$0")"
URL="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/IsaacLab/Mimic/franka_stack_datasets/dataset.hdf5"
OUT=nvidia_teleop_10.hdf5
SHA_PREFIX=e0954647328801b0
if [ ! -f "$OUT" ]; then
  curl -L --fail -o "$OUT" "$URL"
fi
got=$(sha256sum "$OUT" | cut -c1-16)
if [ "$got" != "$SHA_PREFIX" ]; then
  echo "checksum mismatch: got $got, expected $SHA_PREFIX..." >&2
  exit 1
fi
echo "$OUT ok ($(du -h "$OUT" | cut -f1), sha256 $got...)"
