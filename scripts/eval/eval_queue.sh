#!/usr/bin/env bash
# Run a list of evaluations with at most MAX evaluators on the GPU at once (T15: four exceed 32 GB).
# Counts the GPU compute processes (nvidia-smi), so evaluations or trainings started by hand are respected.
# Usage: scripts/eval/eval_queue.sh <queue.txt> [MAX=3]   lines: <label> <task> <checkpoint> <eval_type or ->
# ~ and any $VARIABLE in the checkpoint column are expanded from the environment (the shipped queue files use
# $CKPT_P1..$CKPT_P4 and $D). A row whose checkpoint does not exist is skipped, printed and logged.
set -u
queue="$1"; max="${2:-3}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
running() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . ; }
while read -r label task ckpt etype; do
  [ -z "$label" ] && continue
  raw="$ckpt"; ckpt="${ckpt/#\~/$HOME}"
  while [[ "$ckpt" =~ \$([A-Za-z_][A-Za-z0-9_]*) ]]; do v="${BASH_REMATCH[1]}"; ckpt="${ckpt//\$$v/${!v:-}}"; done
  if [ ! -f "$ckpt" ]; then echo "$(date '+%F %T') SKIP $label: checkpoint not found: $raw -> $ckpt" | tee -a "$root/results/eval_queue.log" >&2; continue; fi
  while [ "$(running)" -ge "$max" ]; do sleep 60; done
  echo "$(date '+%F %T') launch $label" >> "$root/results/eval_queue.log"
  if [ "$etype" = "-" ]; then
    "$root/scripts/eval/eval_policy.sh" "$label" "$task" "$ckpt" 100 0 < /dev/null > /dev/null 2>&1 &
  else
    "$root/scripts/eval/eval_policy.sh" "$label" "$task" "$ckpt" 100 0 "$etype" < /dev/null > /dev/null 2>&1 &
  fi
  sleep 240  # let the new evaluator load before counting again
done < "$queue"
wait
echo "$(date '+%F %T') queue done" >> "$root/results/eval_queue.log"
