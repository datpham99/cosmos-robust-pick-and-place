"""Selective checkpoint download for Cosmos Transfer1 7B edge-control distilled inference.

NVIDIA's scripts/download_checkpoints.py snapshots every repo in full (~300 GB). The edge-distilled
single-control path needs about 80 GB: the distilled checkpoint, the T5-11B text encoder (PyTorch weights
only), the Cosmos video tokenizer, and the two guardrail models (Cosmos-Guardrail1 and Llama Guard 3, which
transfer.py always loads). Layout matches what transfer.py expects: <checkpoint_dir>/<org>/<repo>/...

Requires a Hugging Face read token passed per process (HF_TOKEN env var or --hf_token; nothing is written to disk) and accepted licenses for
nvidia/Cosmos-Transfer1-7B, nvidia/Cosmos-Guardrail1, nvidia/Cosmos-Tokenize1-CV8x8x8-720p and
meta-llama/Llama-Guard-3-8B.

  conda activate cosmos-transfer1
  python scripts/setup/download_cosmos_checkpoints.py --output_dir ~/cosmos-transfer1/checkpoints
"""
import argparse
import os

from huggingface_hub import snapshot_download

# repo -> allow_patterns (None = whole repo; these are small)
PLAN = {
    "nvidia/Cosmos-Transfer1-7B": ["edge_control_distilled.pt", "config.json", "README.md"],
    "google-t5/t5-11b": ["pytorch_model.bin", "config.json", "spiece.model", "tokenizer.json", "*.txt"],
    "nvidia/Cosmos-Tokenize1-CV8x8x8-720p": None,
    "nvidia/Cosmos-Guardrail1": None,
    "meta-llama/Llama-Guard-3-8B": ["*.safetensors", "*.json", "*.txt", "*.model"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default=os.path.expanduser("~/cosmos-transfer1/checkpoints"))
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--only", nargs="*", default=None, help="subset of repo ids to fetch")
    a = ap.parse_args()
    failed = []
    for repo, patterns in PLAN.items():
        if a.only and repo not in a.only:
            continue
        dst = os.path.join(a.output_dir, repo)
        print(f"==> {repo} -> {dst}  patterns={patterns}", flush=True)
        try:
            snapshot_download(repo_id=repo, local_dir=dst, allow_patterns=patterns, max_workers=4, token=a.hf_token)
            print(f"OK   {repo}", flush=True)
        except Exception as e:  # keep going; a gated repo whose license is not yet accepted returns 403
            failed.append(repo)
            print(f"FAIL {repo}: {type(e).__name__}: {str(e).splitlines()[0][:160]}", flush=True)
    print("done; failed:", failed or "none")


if __name__ == "__main__":
    main()
