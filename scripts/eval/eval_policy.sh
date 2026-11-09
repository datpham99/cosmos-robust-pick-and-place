#!/usr/bin/env bash
# Evaluate one robomimic checkpoint with Isaac Lab's play.py under the fixed protocol
# (100 rollouts, seed 0, cameras on, headless) and append the result with a 95% Wilson interval.
# Usage: scripts/eval/eval_policy.sh <label> <task_id> <checkpoint.pth> [num_rollouts] [seed] [eval_type]
# With a 6th argument the rollouts run through scripts/eval/shift_eval.py (same protocol, eval_mode on,
# that shift setting active) and a first-frame contact sheet lands in results/figures/shift_frames_<label>.png.
set -u
label="$1"; task="$2"; ckpt="$3"; n="${4:-100}"; seed="${5:-0}"; eval_type="${6:-}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
log="$root/results/eval_${label}.log"
cd "${ISAACLAB_DIR:-$HOME/IsaacLab}" || exit 1
unset CONDA_PREFIX VIRTUAL_ENV
export PYTHONUNBUFFERED=1  # per-trial lines reach the log as they happen
if [ -n "$eval_type" ]; then
  "$root/scripts/run_with_vram.sh" "$log" ./isaaclab.sh -p "$root/scripts/eval/shift_eval.py" \
    --task "$task" --headless --enable_cameras --num_rollouts "$n" --seed "$seed" --checkpoint "$ckpt" \
    --eval_type "$eval_type" --frames_out "$root/results/figures/shift_frames_${label}.png"
else
  "$root/scripts/run_with_vram.sh" "$log" ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
    --task "$task" --headless --enable_cameras --num_rollouts "$n" --seed "$seed" --checkpoint "$ckpt"
fi
rc=$?
python3 - "$label" "$task" "$ckpt" "$log" "$seed" "$root" <<'EOF'
import math, re, sys, datetime
label, task, ckpt, log, seed, root = sys.argv[1:7]
txt = open(log).read()
m = re.search(r"Successful trials: (\d+), out of (\d+)", txt)
if not m:
    print(f"[EVAL] no result line in {log}"); sys.exit(1)
k, n = int(m.group(1)), int(m.group(2))
p = k / n; z = 1.96
den = 1 + z*z/n; c = (p + z*z/(2*n)) / den; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
lo, hi = max(0, c-h), min(1, c+h)
vram = re.search(r"vram_peak_mib=(\d+)", txt); vram = vram.group(1) if vram else "?"
row = f"| {datetime.date.today()} | {label} | {task} | {ckpt.split('/')[-1]} | seed {seed} | {k}/{n} | {p:.3f} | [{lo:.3f}, {hi:.3f}] | {vram} MiB |\n"
out = f"{root}/results/eval_results.md"
try: hdr = open(out).read()
except FileNotFoundError: hdr = ""
if not hdr:
    hdr = "# Evaluation results\n\nIn-distribution unless the label says otherwise. Wilson 95% interval. VRAM peak is the whole GPU during the run, other jobs included.\n\n| date | label | task | checkpoint | seed | successes | rate | 95% CI | VRAM peak |\n|---|---|---|---|---|---|---|---|---|\n"
    open(out, "w").write(hdr)
open(out, "a").write(row)
print("[EVAL]", row.strip())
EOF
exit $rc
