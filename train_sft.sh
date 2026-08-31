#!/bin/bash
# Stage 1: QLoRA SFT on Qwen2.5-Coder-7B via MLX (Apple Silicon).
# Prereq: pip install mlx-lm ; python make_dataset.py ... first.
# ~2-4 hours on M-series for 100k examples. Adjust --iters to your dataset size
# (roughly dataset_size * epochs / batch_size).
set -e

# 4-bit base = true QLoRA: ~4GB weights instead of 15GB bf16. Combined with
# batch 1 + grad checkpointing + 1024 seq (prompts are ~500 tokens), peak
# memory stays ~8-12GB. Do NOT raise batch/seq on a 64GB machine -- the
# bf16 7B at batch 4 x 2048 without checkpointing peaks over 60GB and will
# freeze macOS.
MODEL="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
DATA="data"           # needs train.jsonl + valid.jsonl from make_dataset.py
ADAPTER="adapters_sft"

mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --data "$DATA" \
  --adapter-path "$ADAPTER" \
  --fine-tune-type lora \
  --num-layers 8 \
  --batch-size 1 \
  --grad-checkpoint \
  --iters 4000 \
  --learning-rate 1e-4 \
  --steps-per-eval 200 \
  --save-every 500 \
  --max-seq-length 1024
# resume after an interruption by adding:
#   --resume-adapter-file "$ADAPTER/adapters.safetensors"

# No fuse step needed with a quantized base: point the proposer at the
# adapters directly (mlx_lm loads base + adapter together):
echo "done -> use with:"
echo "  python packer.py ... --proposer llm --model $ADAPTER"
echo "(llm_proposer resolves the base model from the adapter config; if your"
echo " mlx-lm version needs it fused instead, run:"
echo "  mlx_lm.fuse --model $MODEL --adapter-path $ADAPTER --save-path model_sft)"
