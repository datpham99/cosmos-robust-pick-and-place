#!/usr/bin/env bash
# Run a command, log its output, and sample total GPU memory every 5 s so the peak is on record.
# Usage: scripts/run_with_vram.sh <log_file> <command> [args...]
# Writes <log_file> and <log_file>.vram (one MiB sample per line). Exits with the command's exit code.
set -u
log="$1"; shift
vram="${log}.vram"
: > "$vram"
echo "start $(date '+%Y-%m-%d %H:%M:%S')" > "$log"
"$@" >> "$log" 2>&1 &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$vram"
  sleep 5
done
wait "$pid"; rc=$?
# Isaac Sim's python.sh wrapper can return 0 after a Python traceback, so check the log too.
tb=$(grep -c "^Traceback" "$log")
[ "$rc" -eq 0 ] && [ "$tb" -gt 0 ] && rc=99
echo "exit=$rc tracebacks=$tb end $(date '+%Y-%m-%d %H:%M:%S') vram_peak_mib=$(sort -n "$vram" | tail -1)" >> "$log"
exit $rc
