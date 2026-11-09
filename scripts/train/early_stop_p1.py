"""Early-stop rule for the ten-demonstration policy (policy 1) at the 600-epoch budget: live monitor or replay.

The rule reads per-epoch training "Loss" values from a robomimic train log and stops when the 10-epoch running
MEDIAN has not improved on its best by >= 0.1 for 30 consecutive epochs ("flat"), or has sat >= 0.5 above its best
for 10 consecutive epochs ("rising"). It then picks the saved checkpoint (every 20 epochs) nearest the best-median
epoch and writes the decision to <decision_file>. Training loss only; no evaluation data is used.

  live    watch a running training, kill it (pkill -f <pattern>) when the rule fires or training ends by itself
  replay  apply the rule epoch by epoch to a finished log: the epoch at which the live monitor would have stopped
          and the checkpoint it would have chosen. <train.log> may also be a plain list of per-epoch losses, one
          per line, such as results/records/p1_v2_600_loss.txt.

Usage: python3 scripts/train/early_stop_p1.py live   <train.log> <run_dir> <pkill_pattern> <decision_file>
       python3 scripts/train/early_stop_p1.py replay <train.log> <run_dir> <decision_file> [--flat-only]

--flat-only applies the plateau ("flat") clause alone. On policy 1's loss curve the rising clause trips on an early
plateau and selects checkpoint 20; the flat clause alone selects checkpoint 160, the reported checkpoint
(docs/writeup.md, Notes before you start). Both were evaluated: 0 of 100.
"""
import glob
import re
import statistics
import subprocess
import sys
import time

PATIENCE, MIN_DELTA, RISE, RISE_N, WIN = 30, 0.1, 0.5, 10, 10
FLAT_ONLY = "--flat-only" in sys.argv


def losses(log):
    text = open(log).read()
    found = re.findall(r'"Loss": ([-0-9.]+)', text)
    if found:
        return [float(x) for x in found]
    return [float(line) for line in text.split() if re.fullmatch(r"[-0-9.]+", line)]


def medians(L, n):
    """10-epoch running medians over the first n losses."""
    return [statistics.median(L[i - WIN + 1:i + 1]) for i in range(WIN - 1, n)]


def fired(means):
    """The rule on a list of running medians: 'flat', 'rising' or None."""
    best = min(means)
    since_best = len(means) - 1 - means.index(best)
    improved_recently = any(means[j] < min(means[:j]) - MIN_DELTA for j in range(max(1, len(means) - PATIENCE), len(means)))
    if since_best >= PATIENCE and not improved_recently:
        return "flat"
    if not FLAT_ONLY and len(means) >= RISE_N and all(m >= best + RISE for m in means[-RISE_N:]):
        return "rising"
    return None


def decide(L, stop, reason, run_dir, decision):
    """Choose the checkpoint for a run stopped at epoch `stop` and write the decision line."""
    ckpts = sorted(int(m.group(1)) for p in glob.glob(f"{run_dir}/models/model_epoch_*.pth") for m in [re.search(r"epoch_(\d+)", p)])
    means = medians(L, stop) if stop >= WIN else L[:stop]
    best_ep = (means.index(min(means)) + WIN) if stop >= WIN else stop
    chosen = min((c for c in ckpts if c <= stop), key=lambda e: abs(e - best_ep), default=None)
    line = (f"stopped_at_epoch={stop} reason={reason} best_median_epoch={best_ep} best_median_loss={min(means):.3f} "
            f"chosen_checkpoint={chosen} final_epoch={len(L)} final_loss={L[-1]:.3f}\n")
    open(decision, "w").write(line)
    print(line, end="")


def live(log, run_dir, pattern, decision):
    while True:
        L = losses(log)
        if len(L) >= WIN:
            reason = fired(medians(L, len(L)))
            if reason:
                subprocess.run(["pkill", "-f", pattern])
                decide(L, len(L), reason, run_dir, decision)
                return
        if "exit=" in open(log).read():  # run_with_vram.sh wrote its last line: training ended by itself
            L = losses(log)
            decide(L, len(L), "finished", run_dir, decision)
            return
        time.sleep(60)


def replay(log, run_dir, decision):
    L = losses(log)
    stop, reason = len(L), "finished"
    for n in range(WIN, len(L) + 1):
        r = fired(medians(L, n))
        if r:
            stop, reason = n, r
            break
    decide(L, stop, reason, run_dir, decision)


if __name__ == "__main__":
    sys.argv = [a for a in sys.argv if a != "--flat-only"]
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "live" and len(sys.argv) == 6:
        live(*sys.argv[2:6])
    elif mode == "replay" and len(sys.argv) == 5:
        replay(*sys.argv[2:5])
    else:
        sys.exit(__doc__)
