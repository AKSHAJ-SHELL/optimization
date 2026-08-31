#!/usr/bin/env python3
"""Package a verified packing into a registry submission bundle.

  python submit.py runs/10_tan_in_tan.json --record 3.3078

Refuses to package anything that fails the independent verifier. Produces
submissions/<name>/ containing:
  packing.png        clean picture (registry style)
  coordinates.txt    full-precision poses + shape/container conventions
  submission.txt     ready-to-send message body
  verified.json      the config + verifier report

Send picture + coordinates to the registry maintainer (contact on
https://erich-friedman.github.io/packing/).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file")
    ap.add_argument("--record", type=float, default=None,
                    help="current registry value being improved")
    ap.add_argument("--name", default=None, help="your name for the credit")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    # 1) independent verification gate
    print("running independent verifier...")
    r = subprocess.run([sys.executable, "verify.py", args.json_file,
                        "--tol", str(args.tol)], capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        raise SystemExit("verifier FAILED -- nothing packaged. Fix first.")
    safe_line = [l for l in r.stdout.splitlines() if "safe value" in l]

    fam = data["family"]
    n = data["n"]
    s = data["s"]
    name = f"{n}_{fam.replace(' ', '_')}"
    out = os.path.join("submissions", name)
    os.makedirs(out, exist_ok=True)

    # 2) picture (reuse engine rendering)
    png = args.json_file.replace(".json", ".png")
    if os.path.exists(png):
        shutil.copy(png, os.path.join(out, "packing.png"))

    # 3) coordinates at full precision
    shape, cont = fam.split(" in ")
    conv = {"tan": "right isosceles triangle, short side (leg) = 1",
            "square": "unit square (side 1)",
            "circle": "unit circle (radius 1)",
            "domino": "1 x 2 rectangle (short side 1)",
            "L": "L of short side 1 (2x2 square minus 1x1 corner)",
            "ngon:3": "unit equilateral triangle (side 1)"}
    display = {"ngon:3": "equilateral triangle"}
    shape_disp = display.get(shape, shape)
    with open(os.path.join(out, "coordinates.txt"), "w") as f:
        f.write(f"{n} {shape_disp}(s) in the smallest {cont}\n")
        f.write(f"shape convention: {conv.get(shape, shape)}\n")
        f.write(f"container convention: {conv.get(cont, cont)}, size s\n")
        f.write(f"container size s = {s!r}\n")
        f.write(f"audit: max overlap {data['audit_max_overlap']:.3e}, "
                f"max outside {data['audit_max_outside']:.3e} "
                f"(independently re-verified at 50 digits)\n\n")
        f.write("shape  x  y  theta_rad (pose of shape frame; see moves.py "
                "shape_def for vertex definitions)\n")
        for i, sh in enumerate(data["shapes"]):
            f.write(f"{i}  {sh['x']!r}  {sh['y']!r}  {sh['theta']!r}\n")

    # 4) message body
    who = args.name or "[your name]"
    with open(os.path.join(out, "submission.txt"), "w") as f:
        f.write(f"Subject: improved packing: {n} {shape_disp}s in a {cont}\n\n")
        f.write("Dear Erich,\n\n")
        f.write(f"I believe I have an improved packing of {n} unit {shape_disp}s "
                f"in a {cont}:\n\n")
        f.write(f"    s = {s:.9f}")
        if args.record is not None:
            f.write(f"   (current page value: {args.record})")
        f.write("\n\n")
        if safe_line:
            f.write(f"({safe_line[0].strip()})\n\n")
        f.write("The configuration was found by a basin-hopping optimizer "
                "with exact autodiff gradients and verified independently "
                "in 50-digit arithmetic (max overlap and containment "
                "violation below 1e-9; audit values in the attached "
                "coordinates file). Picture and full-precision coordinates "
                "attached.\n\n")
        f.write(f"Best regards,\n{who}\n")

    # 5) archive the verified config
    shutil.copy(args.json_file, os.path.join(out, "verified.json"))
    print(f"packaged -> {out}/  (picture, coordinates, message body)")
    if args.record is not None and s >= args.record:
        print(f"WARNING: s = {s} does NOT beat the stated record "
              f"{args.record}; double-check before sending.")


if __name__ == "__main__":
    main()
