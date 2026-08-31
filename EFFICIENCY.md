# Memory & efficiency guide (64 GB Apple silicon)

Written after the bf16 incident: the old training config (7B bf16, batch 4,
seq 2048, no checkpointing) peaked >60 GB and froze macOS. This documents
what actually uses memory, the safe envelopes, and how to watch it.

## Run commands (foreground, no caffeinate)

    python make_dataset.py --logs 'logs/moves.jsonl.*' --out data --holdout-n 6 13
    python figures.py --only dataset moves
    bash train_sft.sh 2>&1 | tee sft.log

Keep the machine from sleeping yourself: System Settings → Battery →
Options → "Prevent automatic sleeping on power adapter when the display is
off", or just keep the display on. If the run is interrupted anyway, nothing
is lost — adapters checkpoint every 500 iters; resume with:

    --resume-adapter-file adapters_sft/adapters.safetensors

## Where training memory goes (mental model)

    peak ≈ weights + adapter/optimizer state + activations + logits

| Component | Scaling | 7B numbers |
|---|---|---|
| Weights | fixed per model | bf16 ≈ 15 GB; **4-bit ≈ 4.2 GB** |
| LoRA adapter + Adam state | rank × layers | < 0.5 GB (rank 32, 8 layers) |
| Activations | batch × seq × layers | the killer; grad-checkpoint cuts ~5–10× |
| Logits + loss | batch × seq × 152k vocab | why seq 1024 vs 2048 matters |

The freeze was activations + logits at batch 4 × seq 2048 across 28 layers
with bf16 weights: ~45 GB on top of 15 GB of weights. Unified memory then
swap-thrashes; macOS freezes rather than killing the process.

## Training knobs (memory / speed / quality)

| Knob | Memory | Speed | Quality |
|---|---|---|---|
| 4-bit base (QLoRA) | −11 GB vs bf16 | ≈ same | marginal loss; standard practice |
| `--batch-size` | linear | higher = faster | neutral (small-data regime) |
| `--max-seq-length` | ~linear | higher = slower | none beyond your longest prompt (~500 tok) |
| `--grad-checkpoint` | −5–10× activations | +25–35% step time | none |
| `--num-layers` (LoRA depth) | small | small | more layers = more capacity |
| `--iters` | none | linear time | until val loss flattens |

## Safe envelopes (this machine)

| Config | Peak | Verdict |
|---|---|---|
| 1.5B bf16, batch 4, seq 2048 | 9.1 GB (measured) | fine |
| **7B 4-bit, batch 1, seq 1024, ckpt (current default)** | ~8–12 GB | safe |
| 7B 4-bit, batch 2, seq 1024, ckpt | ~14 GB | max recommended |
| 7B bf16, batch 4, seq 2048, no ckpt | >60 GB | froze the machine — never |

Speed at the safe config is ~2–4× slower per iter than the reckless one;
with batch 1 you need proportionally more iters for the same epochs, so
expect a few hours. That's the trade for a usable machine.

## Monitoring

- **The authoritative number is MLX's own `Peak mem` printed every 10
  iters in the training log.** Abort rule: if it exceeds ~20 GB, Ctrl-C
  (or `pkill -f mlx_lm.lora`) and resume later with smaller settings —
  nothing is lost.
- Activity Monitor → Memory tab → **Memory Pressure graph**: green fine,
  yellow = swapping has started, red = kill the job now.
- First 10 minutes of any new config: watch it. After it plateaus, it
  stays there — MLX memory is stable across iters.
- Disk: `du -sh ~/.cache/huggingface` (4-bit 7B ≈ 4.3 GB; the old bf16
  download can be deleted: `hf cache scan` / delete the bf16 snapshot).

## Inference (the LLM proposer)

7B-4bit resident ≈ 4.5–5 GB, adapters add ~50 MB. The model loads once in
the main packer process; joblib workers do NOT duplicate it (they only
re-shrink configs). Proposal speed, not memory, is the constraint —
`--elite-variants` controls samples per state.

## Engine / sweep processes

Each joblib worker is an independent JAX process: ~300–600 MB each. So
`--workers -1` on a 12-core machine ≈ 4–7 GB total. CPU-bound, not
memory-bound.

## What can run simultaneously (64 GB)

| Combination | OK? | Why |
|---|---|---|
| training + sweep `--workers 4` | yes (~18 GB) | GPU vs CPU, little overlap |
| training + normal desktop use | yes | CPU mostly idle |
| training + ablation (`--proposer llm`) | **no** | two resident models + workers |
| sweep + ablation | tolerable | CPU contention only, halves both |

## Recovery checklist

1. Ctrl-C the trainer (or `pkill -f mlx_lm.lora`). Nothing corrupts.
2. `ls adapters_sft/` — the newest `.safetensors` is your checkpoint.
3. Re-run train_sft.sh with `--resume-adapter-file` added.
4. If the whole machine froze: after reboot, the HF download cache and all
   checkpoints are intact; only un-checkpointed iters (<500) are lost.
