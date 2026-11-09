#!/usr/bin/env bash
# Record the demo clips: four policies x seven conditions, three episodes each chosen successes-first from up to
# ATTEMPTS rollouts (scripts/eval/record_demo.py --attempts), at most MAX recorders on the GPU. When every clip is
# done, scripts/eval/clips_postprocess.py rebuilds the GIFs, the seven-condition figure and docs/demos.md from them.
# Usage: [POLICIES="p4 p3"] [CONDS="distractors all"] [ATTEMPTS=10] scripts/eval/record_demos_all.sh <out_dir> [MAX=3]
# The checkpoint paths below are placeholders: export CKPT_P1..CKPT_P4 or fill them in.
set -u
# the output directory is made absolute, because the script cd-s into Isaac Lab below
out="$(realpath -m "$1")"; max="${2:-3}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
L=${ISAACLAB_DIR:-$HOME/IsaacLab}/logs/robomimic/Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0
# the four checkpoints: export CKPT_P1 .. CKPT_P4, or fill in the <timestamp> placeholders below
declare -A CKPT=(
  [p1]="${CKPT_P1:-$L/p1_teleop10_v2_600/<timestamp>/models/model_epoch_160.pth}"
  [p2]="${CKPT_P2:-$L/p2_mimic500_v2_600/<timestamp>/models/model_epoch_600.pth}"
  [p3]="${CKPT_P3:-$L/p3_mimic500_v2_dr_600/<timestamp>/models/model_epoch_600.pth}"
  [p4]="${CKPT_P4:-$L/p4_mimic500_v2_cosmos_600/<timestamp>/models/model_epoch_600.pth}"
)
declare -A LABEL=([p1]="Policy 1: 10 demonstrations" [p2]="Policy 2: + Mimic 500" [p3]="Policy 3: + domain randomization" [p4]="Policy 4: + Cosmos Transfer1")
declare -A COND=([indist]="in-distribution" [camera_offset]="camera offset 5 cm" [light_color]="new light colour"
                 [light_intensity]="new light intensity" [light_texture]="new light texture (HDR)" [distractors]="two distractors" [all]="all five changes")
IN=Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0
SH=CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0
running() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . ; }
mkdir -p "$out"
cd "${ISAACLAB_DIR:-$HOME/IsaacLab}" || exit 1
unset CONDA_PREFIX VIRTUAL_ENV; export PYTHONUNBUFFERED=1
for p in ${POLICIES:-p4 p3 p2 p1}; do
  if [ ! -f "${CKPT[$p]}" ]; then echo "$(date '+%F %T') SKIP $p: checkpoint not found: ${CKPT[$p]} (export CKPT_${p^^})" >> "$out/record.log"; continue; fi
  for c in ${CONDS:-indist distractors light_intensity camera_offset light_texture light_color all}; do
    while [ "$(running)" -ge "$max" ]; do sleep 30; done
    if [ "$c" = indist ]; then task=$IN; et=""; else task=$SH; et="--eval_type $c"; fi
    echo "$(date '+%F %T') launch ${p}_${c}" >> "$out/record.log"
    "$root/scripts/run_with_vram.sh" "$out/${p}_${c}.log" ./isaaclab.sh -p "$root/scripts/eval/record_demo.py" --task "$task" \
      --checkpoint "${CKPT[$p]}" $et --label "${LABEL[$p]}" --condition "${COND[$c]}" --episodes 3 --attempts "${ATTEMPTS:-10}" \
      --out "$out/${p}_${c}.mp4" --headless --enable_cameras < /dev/null > /dev/null 2>&1 &
    sleep 90
  done
done
wait
echo "$(date '+%F %T') all clips done" >> "$out/record.log"
./isaaclab.sh -p "$root/scripts/eval/clips_postprocess.py" all --clips "$out" >> "$out/record.log" 2>&1
