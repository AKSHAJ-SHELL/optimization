# System design — LLM-as-proposal-policy for geometric packing

**One sentence:** a fine-tuned code LLM replaces the random perturbation
distribution inside a classical basin-hopping/elite-pool packing optimizer;
the LLM chooses *which discrete topology move to try*, the classical solver
resolves all precise coordinates, and Friedman's record registry provides
unambiguous external validation.

## Architecture

```
                       ┌─────────────────────────────────────────────┐
                       │                packer.py                    │
                       │                                             │
  restarts ──────────► │  shrink loop: L-BFGS-B on penalty(x, S)     │
                       │  (exact JAX gradients), S *= (1-δ)          │
                       │        │                                    │
                       │        ▼                                    │
                       │  elite pool (top-k configs)                 │
                       │        │  state ──► canonicalizer (moves.py)│
                       │        ▼                     │              │
                       │   PROPOSER ◄─────────────────┘              │
                       │   random (baseline)   OR   LLM (policy)     │
                       │        │ discrete move JSON                 │
                       │        ▼                                    │
                       │  apply_move ► re-shrink ► accept if better  │
                       │        │                                    │
                       │        ▼ (--log)                            │
                       │  (state, move, Δs) JSONL ───────────────┐   │
                       │        │                                │   │
                       │        ▼                                │   │
                       │  endgame: backoff shrink + strict audit │   │
                       └─────────────────────────────────────────┼───┘
                                                                 ▼
              make_dataset.py: SFT positives + DPO (chosen, rejected) pairs
                                          │
                        train_sft.sh (MLX QLoRA)  →  train_dpo.py (TRL)
                                          │
                              llm_proposer.py (mlx_lm)
                                          │
                              back into packer --proposer llm
```

## The central design decision

**The model never emits a precise float.** LLMs are unreliable at 4th-decimal
precision and packings die there. The action space (moves.py) is discrete
macro-moves with coarse parameters:

```json
{"op":"rotate","shape":4,"deg":45}
{"op":"swap","a":1,"b":3}
{"op":"relocate","shape":2,"fx":0.25,"fy":0.75}
{"op":"jiggle","shapes":[0,2],"mag":0.1}
```

The classical layer (penalty + L-BFGS shrink, already validated against proven
optima) resolves exact coordinates after every move. Division of labor:
classical optimizers are excellent at continuous refinement and terrible at
the combinatorial jumps needed to escape basins; the learned policy owns
exactly those jumps.

## State representation (input to the model)

`moves.canonical_state()` produces a symmetry-reduced, low-precision view:
shapes sorted by (distance-from-origin, angle) — kills relabeling symmetry —
coords rounded to 3 decimals, angles in degrees, plus the **contact graph**
(which shapes touch which, and which touch the wall) computed from signed SAT
gaps. Contacts are the structure topology moves operate on; giving them to
the model directly is worth more than any model-size bump.

## Data (generated, not curated)

The optimizer instruments itself: `--log` records every elite-round variant as
`(canonical state, move, s_before, s_after, improved)`. One overnight sweep
across families × n on all cores yields ~10^5 records at zero extra cost.

- **SFT positives:** moves whose re-shrunk s improved. (Imitation of
  successful search behavior.)
- **DPO pairs:** (improving move, non-improving move) *from the same state* —
  the pairing is the supervision; strictly stronger than labeled negatives.
- **Ground-truth anchor:** runs on validate.py families (known optima) tag
  trajectories that reached proven optima.
- **Splits:** hold out entire n values (e.g. 6, 13) and one family
  (e.g. "in L") to measure generalization across size and shape.

Generation command (repeat per family/n, overnight):

    python packer.py --shape tan --n 8 --container tan --attempts 500 \
        --elite-rounds 6 --log logs/moves.jsonl
    python make_dataset.py --logs 'logs/moves.jsonl.*' --out data --holdout-n 6 13

## Training (64GB Apple Silicon)

| Stage | Tool | Data | Time |
|-------|------|------|------|
| 1. SFT | MLX QLoRA, Qwen2.5-Coder-7B-Instruct, r=32, 16 layers | train.jsonl | ~2–4 h |
| 2. DPO | TRL + PEFT LoRA bf16 on MPS, β=0.1, lr 5e-6 | dpo_train.jsonl | ~4–8 h |
| 3. (opt) RSFT | re-mine logs with the improved model in the loop | new logs | iterate |

Tune by **acceptance rate and mean Δs per proposal on held-out states**, not
loss. Key knobs: sampling temperature (invalid-JSON rate vs diversity), K
proposals per elite state, re-mining cadence.

## Deployment

`packer.py --proposer llm --model model_sft` — the LLM samples K moves per
elite config in the main process; variants re-shrink in parallel workers;
invalid generations fall back to random moves (fallback rate is reported —
it's a health metric). `--proposer random` is the *exact* ablation: same
action space, same loop, same compute.

## Evaluation protocol

1. **Sanity gate:** validate.py must pass (proven optima recovered).
2. **Central table (ablation.py):** LLM vs random proposer, ≥3 seeds ×
   6 eval problems, equal budget. Metrics: best s, gap-to-record,
   acceptance rate, time-to-match-record. This is the paper's claim.
3. **Generalization:** held-out n and held-out family rows reported separately.
4. **Record hunt:** deploy the winner on TARGETS.md Tier 1–2 (tans-in-tans
   2005–2009 records, tans-in-L's 2012). Every beat: strict audit
   (violations < 1e-11) → independent mpmath re-verification → submit to
   Friedman. Each accepted record is external validation with your name on it.

## Why this beats the raw-coordinate design

| Failure mode | Raw-coordinate LLM | This design |
|---|---|---|
| Float precision | fatal (4th-decimal overlap) | eliminated (solver owns floats) |
| Invalid outputs | frequent, wasted | schema-validated, random fallback |
| Symmetry waste | model learns 8 views of same packing | canonicalized away |
| SFT ceiling | caps at teacher quality | DPO/search push past imitation |
| Baseline fairness | none | identical loop, identical action space |

## Risks

- **LLM must beat *good* random mutations** (the elite pool is strong). If the
  margin is thin, report acceptance-rate and per-move Δs gains too — a
  positive result at equal compute is publishable even without new records.
- **Distribution shift** small-n → large-n: measured explicitly by holdout-n;
  the search loop compensates when the policy transfers imperfectly.
- **Throughput:** a 7B proposer is slower than rng. Keep K modest (4–8),
  batch generation, and count proposals — not wall-clock only — in one
  reported variant of the ablation.
- **train_dpo.py is version-sensitive** (TRL API moves fast); expect minor
  arg fixes. SFT path (MLX) is stable.

## Milestones

1. Overnight data sweep (logs → ~100k records) and dataset build.  [1–2 days]
2. SFT; check invalid-JSON rate < 5%, acceptance rate vs random.   [1 day]
3. DPO; ablation.py full run.                                      [2–3 days]
4. Record hunt on Tier 1–2 with best proposer; submissions.        [ongoing]
5. Write-up: "Learned proposal distributions for basin hopping."   [after 3]
