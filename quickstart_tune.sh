#!/bin/bash
# "Tune a bit, then run stuff" -- a 1-2 hour end-to-end taste of the pipeline.
# Uses the 1.5B coder for speed; swap MODEL for the 7B when doing the real run
# (train_sft.sh has the full-scale settings).
#
#   bash quickstart_tune.sh
#
# Prereqs: pip install mlx-lm  (plus the packer deps; validate.py must pass)
set -e

MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"

echo "== 1/5: generate ~30 min of training data (adds to starter data/) =="
python sweep.py --plan data --attempts 60 --elite-rounds 4 --minutes 30 \
    --log logs/moves.jsonl

echo "== 2/5: build dataset =="
python make_dataset.py --logs 'logs/moves.jsonl.*' --out data --holdout-n 6 13

# mlx_lm.lora needs a non-empty valid.jsonl; if the holdout is empty early on,
# split 5% off train.
python - <<'EOF'
import os, random
if os.path.getsize("data/valid.jsonl") == 0:
    lines = open("data/train.jsonl").readlines()
    random.Random(0).shuffle(lines)
    k = max(4, len(lines) // 20)
    open("data/valid.jsonl", "w").writelines(lines[:k])
    open("data/train.jsonl", "w").writelines(lines[k:])
    print(f"valid.jsonl was empty; moved {k} examples from train")
EOF

echo "== 3/5: short SFT (a few hundred iters, ~15-30 min on M-series) =="
N=$(wc -l < data/train.jsonl)
ITERS=$(( N > 400 ? 600 : 200 ))
mlx_lm.lora --model "$MODEL" --train --data data \
    --adapter-path adapters_quick --num-layers 8 --batch-size 4 \
    --iters $ITERS --learning-rate 1e-4 --max-seq-length 2048

echo "== 4/5: fuse for fast inference =="
mlx_lm.fuse --model "$MODEL" --adapter-path adapters_quick \
    --save-path model_quick

echo "== 5/5: run stuff -- tuned policy vs a real target =="
python packer.py --shape tan --n 5 --container tan --attempts 30 \
    --elite-rounds 3 --proposer llm --model model_quick --record 2.4058

echo ""
echo "Look at the [llm_proposer] line above: invalid-rate < 10% means the"
echo "tuning took. Compare against the random baseline with:"
echo "  python packer.py --shape tan --n 5 --container tan --attempts 30 \\"
echo "      --elite-rounds 3 --proposer random --record 2.4058"
echo "Then go full scale: RUNBOOK.md stages 1-5 (7B model, overnight data)."
