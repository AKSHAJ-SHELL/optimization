#!/usr/bin/env python3
"""Overnight driver: runs packer across a plan of (shape, n, container),
logging training data and hunting records in one pass.

  python sweep.py --plan tier1 --attempts 500 --elite-rounds 6
  python sweep.py --plan data  --attempts 200 --minutes 480
  python sweep.py --plan tier1 --proposer llm --model model_sft

Plans:
  tier1  tans in tans n=3..27      (soft 2005-2009 records, --record wired in)
  tier2  tans in L's  n=3..18      (soft 2012 records)
  tier3  L's in tans  n=3..22
  data   diverse small-n mix across families (training-data generation)
  smoke  tiny sanity plan

Resume: combos with an existing runs/<name>.json are skipped unless --redo.
Summary appended to runs/summary.csv; beats are flagged loudly. Verify every
beat with verify.py before submitting.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

# Friedman page values (see TARGETS.md). Beat the 4th decimal to be safe.
TANINTAN = {3: 1.9617, 5: 2.4057, 6: 2.6906, 7: 2.8246, 10: 3.3078,
            11: 3.4142, 12: 3.5317, 13: 3.7071, 14: 3.8284, 15: 3.9142,
            17: 4.1817, 19: 4.4137, 20: 4.5357, 21: 4.6817, 22: 4.7677,
            23: 4.8867, 24: 4.9477, 26: 5.1213, 27: 5.2417}
TANINL = {3: 0.9114, 4: 0.9659, 7: 1.1381, 8: 1.2761, 9: 1.3333,
          10: 1.4005, 13: 1.4995, 14: 1.6075, 15: 1.6805, 16: 1.7474,
          17: 1.8047}
LINTAN = {4: 5.6568, 5: 6.2426, 13: 9.7782, 17: 10.8284, 19: 11.3137,
          20: 11.7782}
# Triangles (unit equilateral) in tans -- 2026-rush family, page ends at 23;
# n>=24 are unclaimed new entries. Scraped 2026-07-20.
TRIINTAN = {4: 2.36602, 5: 2.53905, 6: 2.63895, 7: 2.78972, 8: 3.07313,
            9: 3.18252, 10: 3.34084, 11: 3.46951, 12: 3.62487, 13: 3.71163,
            14: 3.88268, 15: 3.98382, 16: 4.03901, 17: 4.20127, 18: 4.27120,
            19: 4.45701, 20: 4.56124, 21: 4.65799, 22: 4.73289, 23: 4.85470}


def plan_items(name):
    if name == "tier1":
        return [("tan", n, "tan", TANINTAN.get(n)) for n in range(3, 28)]
    if name == "tier2":
        return [("tan", n, "L", TANINL.get(n)) for n in range(3, 19)]
    if name == "tier3":
        return [("L", n, "tan", LINTAN.get(n)) for n in range(2, 23)]
    if name == "tri":
        # n=27..30 requested by the registry maintainer (July 2026)
        return [("ngon:3", n, "tan", TRIINTAN.get(n)) for n in range(4, 31)]
    if name == "data":
        items = []
        for shape, cont in [("tan", "tan"), ("tan", "L"), ("tan", "square"),
                            ("square", "tan"), ("circle", "tan"),
                            ("square", "square"), ("circle", "square"),
                            ("domino", "square"), ("L", "square")]:
            for n in range(2, 13):
                items.append((shape, n, cont, None))
        return items
    if name == "smoke":
        return [("tan", 3, "tan", TANINTAN[3]), ("square", 2, "square", 2.0)]
    raise SystemExit(f"unknown plan {name!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="data",
                    choices=["tier1", "tier2", "tier3", "tri", "data",
                             "smoke"])
    ap.add_argument("--attempts", type=int, default=300)
    ap.add_argument("--elite-rounds", type=int, default=5)
    ap.add_argument("--elite-variants", type=int, default=8)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--minutes", type=float, default=None,
                    help="stop starting new combos after this budget")
    ap.add_argument("--log", default="logs/moves.jsonl")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--proposer", default="random", choices=["random", "llm"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.runs, exist_ok=True)
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    summary_path = os.path.join(args.runs, "summary.csv")
    new_summary = not os.path.exists(summary_path)

    t0 = time.time()
    items = plan_items(args.plan)
    beats = []
    for k, (shape, n, cont, record) in enumerate(items):
        if args.minutes and (time.time() - t0) / 60 > args.minutes:
            print(f"[sweep] time budget reached; stopping before item {k}")
            break
        name = f"{n}_{shape}_in_{cont}".replace(":", "")
        out = os.path.join(args.runs, name)
        if os.path.exists(out + ".json") and not args.redo:
            print(f"[sweep] skip {name} (exists; --redo to rerun)")
            continue
        # protect any existing (possibly better) result from being clobbered
        old_s = None
        if os.path.exists(out + ".json"):
            with open(out + ".json") as f:
                old_s = json.load(f)["s"]
            os.replace(out + ".json", out + "_prev.json")
            if os.path.exists(out + ".png"):
                os.replace(out + ".png", out + "_prev.png")
        cmd = [sys.executable, "packer.py", "--shape", shape, "--n", str(n),
               "--container", cont, "--attempts", str(args.attempts),
               "--elite-rounds", str(args.elite_rounds),
               "--elite-variants", str(args.elite_variants),
               "--workers", str(args.workers), "--seed", str(args.seed),
               "--log", args.log, "--out", out]
        if record is not None:
            cmd += ["--record", str(record)]
        if args.quick:
            cmd += ["--quick"]
        if args.proposer == "llm":
            cmd += ["--proposer", "llm", "--model", args.model or ""]
        print(f"[sweep] {k + 1}/{len(items)} {name} "
              f"(record {record})", flush=True)
        tr = time.time()
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[sweep] {name} FAILED ({e}); continuing")
            if old_s is not None:                 # restore protected result
                os.replace(out + "_prev.json", out + ".json")
                if os.path.exists(out + "_prev.png"):
                    os.replace(out + "_prev.png", out + ".png")
            continue
        with open(out + ".json") as f:
            s = json.load(f)["s"]
        if old_s is not None:
            if old_s < s:                          # previous run was better
                os.replace(out + "_prev.json", out + ".json")
                if os.path.exists(out + "_prev.png"):
                    os.replace(out + "_prev.png", out + ".png")
                print(f"[sweep] kept previous better s={old_s:.9f} "
                      f"(this run: {s:.9f})")
                s = old_s
            else:
                for p in (out + "_prev.json", out + "_prev.png"):
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except OSError:
                        pass                       # stale _prev files are harmless
        beat = record is not None and s < record
        if beat:
            beats.append((name, s, record))
            print(f"[sweep] *** POSSIBLE RECORD: {name} s={s:.9f} < {record} "
                  f"-- run: python verify.py {out}.json ***")
        with open(summary_path, "a", newline="") as f:
            w = csv.writer(f)
            if new_summary:
                w.writerow(["name", "shape", "n", "container", "s", "record",
                            "beat", "seconds"])
                new_summary = False
            w.writerow([name, shape, n, cont, f"{s:.10f}", record, beat,
                        round(time.time() - tr, 1)])

    print(f"\n[sweep] done in {(time.time() - t0) / 60:.1f} min; "
          f"summary -> {summary_path}")
    if beats:
        print(f"[sweep] {len(beats)} possible record(s):")
        for name, s, r in beats:
            print(f"  {name}: {s:.9f} vs {r}  -> verify.py before submitting!")


if __name__ == "__main__":
    main()
