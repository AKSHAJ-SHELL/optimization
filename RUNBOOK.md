# RUNBOOK — exact execution order

Everything below runs on your Mac from the `friedman/` folder. Stages are
ordered; each has a gate — don't advance past a failed gate.

## Stage 0 — setup (15 min, once)

    pip install jax scipy numpy matplotlib joblib mpmath
    pip install mlx-lm                        # training + LLM proposer
    python validate.py

**Gate:** all validate checks PASS. (You've already passed this.)

## Stage 1 — first hunts + data, nights 1–2 (unattended)

    python sweep.py --plan tier1 --attempts 500 --elite-rounds 6   # night 1
    python sweep.py --plan tier2 --attempts 500 --elite-rounds 6   # night 2
    python sweep.py --plan data  --attempts 150 --elite-rounds 4 --minutes 240

Every run logs training data to `logs/` as a side effect. Check each morning:

    cat runs/summary.csv          # any beat=True rows?
    python verify.py runs/<beat>.json
    python submit.py runs/<beat>.json --record <page value> --name "Akshaj Shandilya"

Records can fall here already, before any ML — the classical engine is strong.
Send the `submissions/<name>/` bundle to the contact on the registry site.

**Gate for stage 2:** `ls logs/ | wc -l` shows files and
`python make_dataset.py --logs 'logs/moves.jsonl.*' --out data` reports
≥ 30k records with a sane improved-rate (5–30%). If short, run more
`--plan data` nights; it scales linearly with time.

## Stage 2 — dataset (10 min)

    python make_dataset.py --logs 'logs/moves.jsonl.*' --out data --holdout-n 6 13

Inspect: `head -1 data/train.jsonl | python -m json.tool`. Prompt should show
coordinates + contacts; completion is a single move JSON.

## Stage 3 — SFT (~2–4 h)

    bash train_sft.sh        # edit --iters ≈ dataset_size × 3 / batch_size first

Smoke-test the policy immediately (5 min):

    python packer.py --shape tan --n 5 --container tan --attempts 40 \
        --elite-rounds 3 --proposer llm --model model_sft --quick

**Gate:** the `[llm_proposer]` exit line reports **< 10% invalid** and the
run completes. If invalid-rate is high: lower `--llm-temp` (0.5), or train
longer. If it never improves on random in stage 5, more/cleaner data beats
more epochs.

## Stage 4 — DPO (~4–8 h; expect friction)

    python train_dpo.py --base <HF export of model_sft>
    mlx_lm.convert --hf-path model_dpo --mlx-path model_dpo_mlx

This is the version-sensitive step (TRL API churn). If it fights you for
more than an evening, skip to stage 5 with `model_sft` — SFT-only is a valid
paper configuration; DPO is the upgrade, not the prerequisite.

## Stage 5 — the ablation (overnight; this is the paper's main table)

    python ablation.py --model model_sft --seeds 3 --attempts 200 --rounds 4
    python figures.py --only ablation moves      # fig_ablation fills itself
    # repeat with model_dpo_mlx if stage 4 succeeded

**Gate:** compare mean gap and acceptance rate, llm vs random, in
`results.csv`. Three outcomes: (a) LLM wins → strong paper, proceed;
(b) tie → report acceptance-rate/equal-proposal variants, still publishable;
(c) LLM loses → re-mine data with more elite rounds, retrain once, and
report honestly whichever way it lands.

## Stage 6 — LLM-guided record hunts (nights, ongoing)

    python sweep.py --plan tier1 --attempts 300 --elite-rounds 8 \
        --proposer llm --model <best model>
    python sweep.py --plan tier3 --attempts 300 --elite-rounds 8 \
        --proposer llm --model <best model>     # L's in tans, incl. "Trivial" probes

Every beat: `verify.py` → `submit.py` → email. Log each accepted record in
the paper's Table 3.

## Stage 7 — finalize the paper (1–2 days)

1. `python figures.py --all` (full budgets; replaces the --quick sandbox figures)
2. Copy `figures/` into `paper/figures/`, fill red TODOs in `paper.tex`:
   affiliation, repo URL (push the folder to GitHub first), ablation +
   records tables, abstract/conclusion numbers.
3. Fix the two bib entries marked CHECK (AlphaEvolve, Xu 1996).
4. `cd paper && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper`
5. Venue: the sn-jnl class fits Springer journals — realistic targets:
   *Journal of Global Optimization*, *Optimization and Engineering*, or
   *Machine Learning* (Springer); workshop-first (NeurIPS OPT / ICML AI4Math)
   is a good de-risk if you want feedback before journal review.
6. Submit the .tex + separate figure .eps files + .bib, per the template's
   user-manual.pdf.

## Rough calendar

| Days | What |
|------|------|
| 1–3 | Stage 1 nights; possible first record submissions |
| 4 | Stages 2–3 (dataset + SFT + smoke) |
| 5–6 | Stage 4 (DPO) or skip; stage 5 ablation overnight |
| 7–14 | Stage 6 hunts every night; stage 7 paper during the day |

## If something breaks

- JAX/Metal: engine is CPU-JAX; do NOT install jax-metal, plain `pip install jax` is correct.
- Long runs: sweep.py resumes (skips finished combos); rerun the same command.
- A "beat" that fails verify.py: it's not a record; increase --attempts and endgame reruns.
- LLM proposer slow: lower --elite-variants (K), it's proposals-per-state.
