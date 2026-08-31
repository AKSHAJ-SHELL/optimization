# friedman — LLM-guided packing record hunter

Shape-packing record hunter for [Erich's Packing Center](https://erich-friedman.github.io/packing/),
plus an LLM-as-proposal-policy research pipeline on top (see DESIGN.md).

Project layout:

- `packer.py` — the optimizer (JAX gradients, elite pool, endgame audit,
  `--log` data generation, `--proposer random|llm`)
- `moves.py` — discrete move schema + canonical state (the LLM action space)
- `llm_proposer.py` — MLX inference wrapper for the fine-tuned policy
- `make_dataset.py` — logs → SFT/DPO JSONL with held-out splits
- `train_sft.sh`, `train_dpo.py` — QLoRA SFT (MLX) and DPO (TRL/MPS)
- `ablation.py` — LLM vs random proposer, the paper's central experiment
- `sweep.py` — overnight driver: whole record tiers / data plans in one command
  (resume support, summary.csv, flags possible records)
- `verify.py` — independent mpmath re-verification (50 digits, no shared code
  with the engine); REQUIRED before submitting any record
- `validate.py` — known-optima gate; run before trusting anything
- `TARGETS.md` — the record hit list; `DESIGN.md` — full system design

Quick start (hunt records, no ML):

    python validate.py
    python sweep.py --plan tier1 --attempts 500 --elite-rounds 6   # overnight
    python verify.py runs/<any flagged beat>.json                  # then submit

Research pipeline: `python sweep.py --plan data` (overnight, logs training
data as a side effect) → `make_dataset.py` → `train_sft.sh` → `train_dpo.py`
→ `ablation.py` → re-run tiers with `--proposer llm`.

## The classical engine
Same proven architecture as Flamethrower's polygon-packer (penalty → shrink →
restart), upgraded where it counts:

1. **Exact gradients** (JAX autodiff) instead of finite differences — each
   L-BFGS-B step costs 1 evaluation instead of ~3N.
2. **Shapes his tool can't express**: tans, L's (non-convex), dominoes, circles,
   regular n-gons — the families whose records are 14–21 years old (see TARGETS.md).
3. **Non-convex containers**: the L container is modeled exactly as bounding
   square + forbidden corner obstacle.
4. **Elite pool**: best configs are perturbed (jiggle / rotation-snap / swap /
   teleport) and re-shrunk, instead of relying only on independent restarts.
5. **Endgame**: geometric-backoff shrink squeezes ~9 digits, then a strict
   feasibility audit (max overlap depth, max containment violation) runs before
   any value is reported — no silent 1e-4 overlaps.

## Install

    pip install jax scipy numpy matplotlib joblib

## Use

    python validate.py                      # run this first; all checks must PASS
    python packer.py --shape tan --n 10 --container tan \
        --attempts 3000 --workers -1 --record 3.3075

Shapes/containers: `tan`, `square`, `domino`, `circle`, `L`, `ngon:k`.
Sizes follow Friedman's conventions (tan leg 1, circle radius 1, L short side 1,
n-gon side 1); the reported `s` is directly comparable to his pages.

Key flags: `--attempts` (restarts; throughput is the resource — go big),
`--elite-rounds/--elite-k/--elite-variants` (exploitation of best configs),
`--record` (prints beat/miss margin), `--quick` (smoke tests only).

Outputs: `<prefix>.json` (coordinates + audit) and `<prefix>.png` (picture).

## Validation status (sandbox, small budgets)

- 2 squares in square → 2.0000000 (exact 2)
- 2 circles in square → 3.41421357 (exact 2+√2)
- 2 circles in tan → 4.82842713 (proven 2+2√2, matched to 9 digits)
- 5 squares in square → 2.70712 (proven 2.70711, right basin)
- 3 tans in tan → 1.96179 (record "1.961+", Pegg 2005 — already at record
  precision with 10 quick attempts)

## Before claiming a record

Audit line must show violations < 1e-11, then independently re-verify the JSON
coordinates in high precision (mpmath), then submit to Friedman. Quote s
conservatively (round UP at the digit you trust).
# optimization
