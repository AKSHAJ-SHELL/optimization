#!/usr/bin/env python3
"""Validation gate for packer.py -- run before trusting any record claim.

Recovers known/proven optima. All should PASS in a few minutes on a laptop.
"""
import subprocess
import json
import math
import sys

CHECKS = [
    # (shape, n, container, known s, tolerance, note)
    ("square", 2, "square", 2.0,                    1e-5, "2 squares in square (trivial)"),
    ("circle", 2, "square", 2 + math.sqrt(2),       1e-5, "2 circles in square (proven)"),
    ("circle", 2, "tan",    2 + 2 * math.sqrt(2),   1e-5, "2 circles in tan (proven, Xu 1996)"),
    ("circle", 3, "tan",    4 + math.sqrt(2),       1e-4, "3 circles in tan (proven, Xu 1996)"),
    ("square", 5, "square", 2 + 1 / math.sqrt(2),   1e-3, "5 squares in square (proven, Friedman)"),
    ("tan",    3, "tan",    1.962,                  1e-2, "3 tans in tan (record 1.961+, Pegg 2005)"),
]

def run(shape, n, container, attempts):
    out = f"_val_{n}_{shape}_{container}".replace(":", "")
    cmd = [sys.executable, "packer.py", "--shape", shape, "--n", str(n),
           "--container", container, "--attempts", str(attempts),
           "--quick", "--elite-rounds", "1", "--out", out]
    subprocess.run(cmd, check=True, capture_output=True)
    with open(out + ".json") as f:
        d = json.load(f)
    return d["s"], max(d["audit_max_overlap"], d["audit_max_outside"])

def main():
    fails = 0
    for shape, n, container, known, tol, note in CHECKS:
        attempts = 16 if n >= 5 else 8
        s, viol = run(shape, n, container, attempts)
        ok = s <= known + tol and viol < 1e-9
        print(f"{'PASS' if ok else 'FAIL'}  {note}: got {s:.8f} "
              f"(known {known:.8f}, viol {viol:.1e})")
        fails += 0 if ok else 1
    print("\nall passed" if fails == 0 else f"\n{fails} FAILURES -- do not hunt records until fixed")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
