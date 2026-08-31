#!/usr/bin/env python3
"""Collect every packing result (json + png pair) scattered around the repo
and order them into results/<family>/, best-first.

  results/
    tan_in_tan/
      n05_s2.405970.json/.png        <- best known result for this instance
      alternates/n05_s2.409273__5_tan_in_tan.json/.png
    tan_in_L/ ...
    INDEX.md                         <- sortable table of everything

Copies only -- originals are left untouched (delete them yourself once happy).
Re-run anytime; it rebuilds results/ from whatever exists.

  python organize.py                     # scans . runs examples submissions
  python organize.py --scan . runs old   # custom dirs
"""

import argparse
import glob
import json
import os
import shutil
from collections import defaultdict

# registry page values for the beat column (tan/L families; see TARGETS.md)
RECORDS = {
    ("tan in tan", 3): 1.9617, ("tan in tan", 5): 2.4057,
    ("tan in tan", 6): 2.6906, ("tan in tan", 7): 2.8245,
    ("tan in tan", 10): 3.3078, ("tan in tan", 11): 3.4142,
    ("tan in tan", 12): 3.5317, ("tan in tan", 13): 3.7071,
    ("tan in tan", 14): 3.8284, ("tan in tan", 15): 3.9142,
    ("tan in L", 3): 0.9114, ("tan in L", 4): 0.9659,
    ("tan in L", 7): 1.1381, ("tan in L", 8): 1.2761,
    ("tan in L", 9): 1.3333, ("tan in L", 10): 1.4005,
    ("tan in L", 13): 1.4995, ("tan in L", 14): 1.6075,
    ("tan in L", 15): 1.6805,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", nargs="*",
                    default=[".", "runs", "examples", "submissions"])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    found = defaultdict(list)          # (family, n) -> [(s, json, png, src)]
    seen = set()
    for d in args.scan:
        for jf in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
            if os.path.abspath(jf).startswith(os.path.abspath(args.out)):
                continue
            try:
                with open(jf) as f:
                    data = json.load(f)
                fam, n, s = data["family"], data["n"], data["s"]
            except (json.JSONDecodeError, KeyError, TypeError, OSError):
                continue
            key = (fam, n, round(s, 12))
            if key in seen:            # exact duplicate copied elsewhere
                continue
            seen.add(key)
            png = os.path.splitext(jf)[0] + ".png"
            found[(fam, n)].append((s, jf, png if os.path.exists(png) else None))

    rows = []
    for (fam, n), items in sorted(found.items()):
        items.sort(key=lambda t: t[0])
        slug = fam.replace(" ", "_").replace("'", "")
        fdir = os.path.join(args.out, slug)
        os.makedirs(fdir, exist_ok=True)
        for rank, (s, jf, png) in enumerate(items):
            if rank == 0:
                base = os.path.join(fdir, f"n{n:02d}_s{s:.6f}")
            else:
                adir = os.path.join(fdir, "alternates")
                os.makedirs(adir, exist_ok=True)
                src = os.path.splitext(os.path.basename(jf))[0]
                base = os.path.join(adir, f"n{n:02d}_s{s:.6f}__{src}")
            shutil.copy(jf, base + ".json")
            if png:
                shutil.copy(png, base + ".png")
        best_s, best_src = items[0][0], items[0][1]
        rec = RECORDS.get((fam, n))
        beat = "" if rec is None else ("BEAT?" if best_s < rec else "")
        rows.append((fam, n, best_s, rec, len(items), best_src, beat))

    with open(os.path.join(args.out, "INDEX.md"), "w") as f:
        f.write("# Results index (best per instance; copies, originals "
                "untouched)\n\n")
        f.write("| family | n | best s | page value | runs | source | |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for fam, n, s, rec, cnt, src, beat in rows:
            f.write(f"| {fam} | {n} | {s:.9f} | {rec if rec else ''} "
                    f"| {cnt} | {src} | **{beat}** |\n" if beat else
                    f"| {fam} | {n} | {s:.9f} | {rec if rec else ''} "
                    f"| {cnt} | {src} | |\n")

    n_pairs = sum(len(v) for v in found.values())
    n_beats = sum(1 for r in rows if r[6])
    print(f"[organize] {n_pairs} results across {len(found)} instances -> "
          f"{args.out}/ ({n_beats} possible beat(s) flagged in INDEX.md)")


if __name__ == "__main__":
    main()
