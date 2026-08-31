#!/usr/bin/env python3
"""The paper's central experiment: LLM proposer vs random proposer,
same loop, same compute. Writes results.csv.

Usage:
  python ablation.py --model model_sft --seeds 3 \
      --eval "tan:5:tan:2.4058" "tan:10:tan:3.3075" "tan:13:L:1.4995"
"""

import argparse
import csv
import json
import subprocess
import sys
import time

DEFAULT_EVAL = [
    "tan:5:tan:2.4058", "tan:7:tan:2.8245", "tan:10:tan:3.3075",
    "tan:12:tan:3.5315", "tan:10:L:1.4005", "tan:13:L:1.4995",
]


def run(shape, n, container, proposer, model, seed, attempts, rounds):
    out = f"_abl_{shape}_{n}_{container}_{proposer}_{seed}".replace(":", "")
    cmd = [sys.executable, "packer.py", "--shape", shape, "--n", str(n),
           "--container", container, "--attempts", str(attempts),
           "--elite-rounds", str(rounds), "--seed", str(seed),
           "--proposer", proposer, "--out", out]
    if proposer == "llm":
        cmd += ["--model", model]
    t0 = time.time()
    subprocess.run(cmd, check=True, capture_output=True)
    with open(out + ".json") as f:
        return json.load(f)["s"], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", nargs="*", default=DEFAULT_EVAL,
                    help="shape:n:container:record")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--attempts", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=4)
    args = ap.parse_args()

    rows = []
    for spec in args.eval:
        shape, n, container, record = spec.split(":")
        for proposer in ("random", "llm"):
            for seed in range(args.seeds):
                s, dt = run(shape, int(n), container, proposer,
                            args.model, seed, args.attempts, args.rounds)
                rows.append(dict(shape=shape, n=n, container=container,
                                 record=float(record), proposer=proposer,
                                 seed=seed, s=s, gap=s - float(record),
                                 seconds=round(dt, 1)))
                print(rows[-1])

    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("wrote results.csv")


if __name__ == "__main__":
    main()
