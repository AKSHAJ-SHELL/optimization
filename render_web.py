#!/usr/bin/env python3
"""Render a packing JSON as a registry-style, web-ready picture.

Matches the Packing Center page style: same orientation as the engine
(right angle bottom-left for tans), lavender fill, thin dark edges, no
axes, no title, white background, consistent pixel size.

  python render_web.py runs/27_ngon3_in_tan.json            # -> *_web.png
  python render_web.py runs/*.json --px 420
"""

import argparse
import glob
import json
import math
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Circle as MplCirc

FILL = "#d9daeb"        # lavender used on the registry pages
EDGE = "#3b3b47"
LINE_W = 0.9


def render(json_file, px):
    import packer
    with open(json_file) as f:
        d = json.load(f)
    shape_name, cont_name = d["family"].split(" in ")
    S = d["s"]
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    cont = packer.container_def(cont_name)
    if cont["kind"] == "circle":
        ax.add_patch(MplCirc((0, 0), S, fill=False, lw=LINE_W, color="k"))
    else:
        outline = cont.get("outline", cont["verts"])
        ax.add_patch(MplPoly(np.asarray(outline) * S, closed=True,
                             fill=False, lw=LINE_W, color="k"))
    sd = packer.shape_def(shape_name)
    for sh in d["shapes"]:
        x, y, th = sh["x"], sh["y"], sh["theta"]
        c, s_ = math.cos(th), math.sin(th)
        R = np.array([[c, -s_], [s_, c]])
        for p in sd["parts"]:
            if p.get("circle"):
                cc = np.array([x, y]) + R @ p["center"]
                ax.add_patch(MplCirc(cc, p["radius"], facecolor=FILL,
                                     edgecolor=EDGE, lw=LINE_W * 0.7))
            else:
                ax.add_patch(MplPoly(np.array([x, y]) + p["verts"] @ R.T,
                                     closed=True, facecolor=FILL,
                                     edgecolor=EDGE, lw=LINE_W * 0.7))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.autoscale()
    out = json_file.replace(".json", "_web.png")
    fig.savefig(out, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"[render_web] {out}  (s = {d['s']:.9f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_files", nargs="+")
    ap.add_argument("--px", type=int, default=420,
                    help="output image width/height in pixels")
    args = ap.parse_args()
    files = []
    for pat in args.json_files:
        files += glob.glob(pat)
    if not files:
        sys.exit("no files matched")
    for f in files:
        render(f, args.px)


if __name__ == "__main__":
    main()
