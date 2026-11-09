"""Build the Cosmos prompt file: one templated prompt per subset episode (row i is demo_i's), seeded, with three filters.
scripts/cosmos/prep_cosmos_inputs.py reads it and attaches the clips, so no video is needed at this step.

Isaac Lab's cosmos_prompt_gen.py draws one line per section (env/robot/table/cubes/light) at random. This does the
same with three filters applied to the template lines first:
1. Lines naming a held-out shift asset are dropped (the protocol keeps the shift set out of Cosmos prompts). Held-out
   HDR environments: evening road, auto shop, hospital, bus depot, surgery. Lines naming bamboo, cherry, oak, timber,
   walnut or stainless/polished steel are dropped as well.
2. Lines outside the `cubes` section that name a colour (blue LED strips on the robot, a red table, ...) are dropped,
   because the edge-only distilled model takes object colours from the prompt and the first frame and such words
   bleed into the cubes.
3. "chevron-style" is replaced by "V-shaped": Cosmos-Guardrail1's blocklist contains "Chevron" (the company), and a
   blocked prompt skips the whole segment with exit code 0. The word is substituted rather than the line dropped so
   the seeded draw of the other prompts is unchanged.

  python3 scripts/cosmos/make_cosmos_prompts.py --out_jsonl <batch200.jsonl> [--num 200] [--seed 0]
"""
import argparse
import json
import os
import random
import re

TEMPLATES = os.path.join(os.environ.get("ISAACLAB_DIR", os.path.expanduser("~/IsaacLab")),
                         "scripts/tools/cosmos/transfer1_templates.json")
BANNED = [r"\boak\b", r"\bcherry\b", r"\bwalnut\b", r"\bbamboo\b", r"\btimber\b", r"stainless", r"polished steel",
          r"\bhospital", r"\bsurg", r"auto ?shop", r"\bgarage", r"bus depot", r"evening road", r"\broad at dusk"]
SUBST = {r"\bchevron-style\b": "V-shaped"}  # words on the Cosmos guardrail blocklist
COLOUR_WORDS = r"\b(blue|red|green|teal|cyan|crimson|scarlet|emerald|lime|navy|azure|turquoise|magenta|purple|violet|pink|orange|yellow)\b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    templates = json.load(open(TEMPLATES))
    kept = {}
    for section, options in templates.items():
        if not isinstance(options, list):
            continue
        for pat, rep in SUBST.items():
            options = [re.sub(pat, rep, o, flags=re.I) for o in options]
        ok = [o for o in options if not any(re.search(b, o, re.I) for b in BANNED)
              and (section == "cubes" or not re.search(COLOUR_WORDS, o, re.I))]
        dropped = [o for o in options if o not in ok]
        kept[section] = ok
        print(f"{section}: kept {len(ok)}/{len(options)}" + (f"; dropped: {[d[:60] for d in dropped]}" if dropped else ""))

    random.seed(a.seed)
    rows = []
    for i in range(a.num):
        rows.append({"prompt": " ".join(random.choice(kept[s]) for s in kept)})
    with open(a.out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {a.out_jsonl} (seed {a.seed})")


if __name__ == "__main__":
    main()
