#!/usr/bin/env python3
"""Turn packer --log output into SFT + DPO training data.

Reads moves.jsonl.* (one file per worker process), writes:
  data/train.jsonl, data/valid.jsonl        (SFT: {"prompt","completion"})
  data/dpo_train.jsonl, data/dpo_valid.jsonl (DPO: {"prompt","chosen","rejected"})

Splits:
  --holdout-n:      these n values go to valid (generalization across n)
  --holdout-family: substring match, e.g. "in L" (generalization across family)

SFT positives = moves whose re-shrunk s improved on s_before.
DPO pairs     = (improving move, non-improving move) from the SAME state.

Usage:
  python make_dataset.py --logs 'logs/moves.jsonl.*' --out data \
      --holdout-n 6 13 --holdout-family 'in L'
"""

import argparse
import glob
import json
import os
import random
from collections import defaultdict

import moves as moves_mod


def load_records(pattern):
    recs = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return recs


def is_holdout(rec, holdout_n, holdout_family):
    if rec["n"] in holdout_n:
        return True
    return any(h and h in rec["family"] for h in holdout_family)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/moves.jsonl.*")
    ap.add_argument("--out", default="data")
    ap.add_argument("--holdout-n", type=int, nargs="*", default=[6, 13])
    ap.add_argument("--holdout-family", nargs="*", default=[])
    ap.add_argument("--max-pairs-per-state", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    recs = load_records(args.logs)
    if not recs:
        raise SystemExit(f"no records matched {args.logs!r} -- run packer "
                         "with --log logs/moves.jsonl first")
    os.makedirs(args.out, exist_ok=True)

    sft = {"train": [], "valid": []}
    by_state = defaultdict(lambda: {"good": [], "bad": [], "split": "train"})
    n_improved = 0
    for r in recs:
        prompt = moves_mod.render_prompt(r["state"])
        move_str = json.dumps(r["move"], separators=(",", ":"))
        split = "valid" if is_holdout(r, set(args.holdout_n),
                                      args.holdout_family) else "train"
        st = by_state[prompt]
        st["split"] = split
        if r["improved"]:
            n_improved += 1
            sft[split].append({"prompt": prompt, "completion": move_str})
            st["good"].append(move_str)
        else:
            st["bad"].append(move_str)

    dpo = {"train": [], "valid": []}
    for prompt, st in by_state.items():
        pairs = [(g, b) for g in st["good"] for b in st["bad"] if g != b]
        rng.shuffle(pairs)
        for g, b in pairs[:args.max_pairs_per_state]:
            dpo[st["split"]].append(
                {"prompt": prompt, "chosen": g, "rejected": b})

    for name, data in (("train", sft["train"]), ("valid", sft["valid"]),
                       ("dpo_train", dpo["train"]), ("dpo_valid", dpo["valid"])):
        rng.shuffle(data)
        with open(os.path.join(args.out, name + ".jsonl"), "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

    print(f"records: {len(recs)}  improved: {n_improved} "
          f"({100.0 * n_improved / len(recs):.1f}%)")
    print(f"SFT  train/valid: {len(sft['train'])}/{len(sft['valid'])}")
    print(f"DPO  train/valid: {len(dpo['train'])}/{len(dpo['valid'])}")
    print(f"wrote to {args.out}/")


if __name__ == "__main__":
    main()
