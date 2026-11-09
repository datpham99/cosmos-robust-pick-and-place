#!/usr/bin/env bash
# Run Cosmos Transfer1 7B edge-control distilled over a batch JSONL on the single RTX 5090.
# Never run while Isaac Sim is on the GPU (docs/tutorial.md, section 0.4). Flags follow NVIDIA issue #205's only working single-5090
# invocation (all three offload flags + --use_distilled) and Isaac Lab's recommended sigma_max/negative prompt.
# Usage: scripts/cosmos/run_cosmos_batch.sh <batch.jsonl> <out_dir> [extra transfer.py args]
# Resume: START=<n> ... runs rows n.. of the batch file (0-based) and renumbers the output folders to video_<row>.
# n is the number of finished rows: ls <out_dir>/video_*/output.mp4 | wc -l. A folder left without output.mp4 by
# the interrupted row is replaced.
set -u
BATCH="$1"; OUT="$2"; shift 2
START="${START:-0}"
REPO="${COSMOS_REPO:-$HOME/cosmos-transfer1}"
CKPT="${COSMOS_CKPT_DIR:-$HOME/cosmos-transfer1/checkpoints}"
NEG="The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic and appears cg. Plane background."
SPEC="$OUT/controlnet_spec_edge.json"
mkdir -p "$OUT"
if [ "$START" -gt 0 ]; then
  RUN_BATCH="$OUT/resume_from_$START.jsonl"; RUN_OUT="$OUT/resume_from_$START"
  tail -n +"$((START + 1))" "$BATCH" > "$RUN_BATCH"
else
  RUN_BATCH="$BATCH"; RUN_OUT="$OUT"
fi
cat > "$SPEC" <<EOF
{
    "negative_prompt": "$NEG",
    "edge": {"control_weight": ${EDGE_WEIGHT:-1.0}}
}
EOF
source "${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")}/etc/profile.d/conda.sh"; set +u; conda activate cosmos-transfer1; set -u
unset CONDA_PREFIX_1 2>/dev/null || true
cd "$REPO"
# validate_controlnet_specs() requires --input_video_path whenever sigma_max < 80, even in batch mode where each
# row's visual_input is what is actually used; the first clip is passed as that placeholder.
FIRST=$(head -1 "$BATCH" | sed -E 's/.*"visual_input": *"([^"]*)".*/\1/')
export PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
  --checkpoint_dir "$CKPT" --tokenizer_dir nvidia/Cosmos-Tokenize1-CV8x8x8-720p \
  --controlnet_specs "$SPEC" --input_video_path "$FIRST" --batch_input_path "$RUN_BATCH" --video_save_folder "$RUN_OUT" \
  --use_distilled --offload_text_encoder_model --offload_guardrail_models --offload_diffusion_transformer \
  --num_gpus 1 --sigma_max 50 --fps 30 --seed 1 "$@"
rc=$?
if [ "$START" -gt 0 ]; then
  for d in "$RUN_OUT"/video_*; do
    [ -d "$d" ] || continue
    j="${d##*video_}"; dst="$OUT/video_$((START + j))"
    [ -d "$dst" ] && [ ! -f "$dst/output.mp4" ] && rm -rf "$dst"
    mv -n "$d" "$dst"
  done
fi
exit $rc
