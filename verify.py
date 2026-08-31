#!/usr/bin/env python3
"""Independent high-precision verifier for packer.py results.

Deliberately does NOT import packer/JAX: geometry is re-implemented from
scratch in mpmath at --dps digits, so a bug in the engine cannot certify
itself. Run this on every candidate before submitting to Friedman.

  python verify.py 10_tan_in_tan.json            # PASS/FAIL + safe quote
  python verify.py result.json --dps 60 --tol 1e-9

Checks: every pairwise separation (exact SAT for convex parts, exact
distances for circles) and every containment constraint (half-planes /
circle radius; the L container's notch is a forbidden rectangle).
"""

import argparse
import json
import sys

from mpmath import mp, mpf, cos, sin, sqrt, atan2, pi


# --- shape catalog (problem definitions, mpmath) -----------------------------

def _reg_ngon(k):
    rc = 1 / (2 * sin(pi / k))
    off = -pi / 2 + pi / k
    return [[rc * cos(2 * pi * i / k + off), rc * sin(2 * pi * i / k + off)]
            for i in range(k)]


def shape_parts(name):
    """Parts in shape frame (centroid at origin). Polygon lists or
    ('circle', radius)."""
    if name == "square":
        h = mpf(1) / 2
        return [[[-h, -h], [h, -h], [h, h], [-h, h]]]
    if name == "domino":
        return [[[-1, mpf(-1) / 2], [1, mpf(-1) / 2],
                 [1, mpf(1) / 2], [-1, mpf(1) / 2]]]
    if name == "tan":
        t = mpf(1) / 3
        return [[[-t, -t], [1 - t, -t], [-t, 1 - t]]]
    if name == "circle":
        return [("circle", mpf(1))]
    if name == "L":
        c = mpf(5) / 6
        r1 = [[0 - c, 0 - c], [2 - c, 0 - c], [2 - c, 1 - c], [0 - c, 1 - c]]
        r2 = [[0 - c, 1 - c], [1 - c, 1 - c], [1 - c, 2 - c], [0 - c, 2 - c]]
        return [r1, r2]
    if name.startswith("ngon:"):
        return [_reg_ngon(int(name.split(":")[1]))]
    raise SystemExit(f"unknown shape {name!r}")


def container_geom(name, S):
    """Returns (kind, boundary_polygon_or_None, notches). Boundary CCW."""
    S = mpf(S)
    if name == "square":
        return "poly", [[0, 0], [S, 0], [S, S], [0, S]], []
    if name == "tan":
        return "poly", [[0, 0], [S, 0], [0, S]], []
    if name == "domino":
        return "poly", [[0, 0], [2 * S, 0], [2 * S, S], [0, S]], []
    if name == "L":
        bounds = [[0, 0], [2 * S, 0], [2 * S, 2 * S], [0, 2 * S]]
        notch = [[S, S], [2 * S, S], [2 * S, 2 * S], [S, 2 * S]]
        return "poly", bounds, [notch]
    if name == "circle":
        return "circle", None, []
    if name.startswith("ngon:"):
        return "poly", [[S * vx, S * vy]
                        for vx, vy in _reg_ngon(int(name.split(":")[1]))], []
    raise SystemExit(f"unknown container {name!r}")


# --- exact geometry ----------------------------------------------------------

def transform(part, x, y, th):
    c, s = cos(th), sin(th)
    return [[x + vx * c - vy * s, y + vx * s + vy * c] for vx, vy in part]


def edge_normals(poly):
    out = []
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        L = sqrt(dx * dx + dy * dy)
        out.append([dy / L, -dx / L])       # outward for CCW
    return out


def sat_overlap(p1, p2):
    """Signed: >0 penetration depth (along best axis), <=0 separated."""
    best = None
    for axis in edge_normals(p1) + edge_normals(p2):
        d1 = [vx * axis[0] + vy * axis[1] for vx, vy in p1]
        d2 = [vx * axis[0] + vy * axis[1] for vx, vy in p2]
        ov = min(max(d1), max(d2)) - max(min(d1), min(d2))
        if best is None or ov < best:
            best = ov
        if best <= 0:
            return best
    return best


def point_poly_signed(p, poly):
    """Signed distance point->convex polygon boundary (<0 inside)."""
    px, py = p
    inside = True
    best = None
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = ((px - ax) * dx + (py - ay) * dy) / L2
        t = max(mpf(0), min(mpf(1), t))
        cx, cy = ax + t * dx, ay + t * dy
        d = sqrt((px - cx) ** 2 + (py - cy) ** 2)
        best = d if best is None else min(best, d)
        if dx * (py - ay) - dy * (px - ax) < 0:   # CCW: right of edge -> outside
            inside = False
    return -best if inside else best


def circle_poly_overlap(c, r, poly):
    return r - point_poly_signed(c, poly)


# --- verification ------------------------------------------------------------

def verify(data, dps, tol):
    mp.dps = dps
    shape_name, container_name = data["family"].split(" in ")
    S = mpf(repr(data["s"]))
    n = data["n"]
    base = shape_parts(shape_name)
    placed = []                       # list of lists of (poly | circle tuple)
    for sh in data["shapes"]:
        x, y, th = (mpf(repr(sh[k])) for k in ("x", "y", "theta"))
        parts = []
        for p in base:
            if isinstance(p, tuple):
                cx = x + p[1] * 0            # circle center offset is origin
                parts.append(("circle", [x, y], p[1]))
            else:
                parts.append(transform(p, x, y, th))
        placed.append(parts)

    max_overlap = mpf(-1e30)
    for i in range(n):
        for j in range(i + 1, n):
            for pa in placed[i]:
                for pb in placed[j]:
                    ca, cb = isinstance(pa, tuple), isinstance(pb, tuple)
                    if ca and cb:
                        (_, c1, r1), (_, c2, r2) = pa, pb
                        d = sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
                        ov = r1 + r2 - d
                    elif ca:
                        ov = circle_poly_overlap(pa[1], pa[2], pb)
                    elif cb:
                        ov = circle_poly_overlap(pb[1], pb[2], pa)
                    else:
                        ov = sat_overlap(pa, pb)
                    max_overlap = max(max_overlap, ov)

    kind, bound, notches = container_geom(container_name, S)
    max_out = mpf(-1e30)
    for parts in placed:
        for p in parts:
            if isinstance(p, tuple):
                _, c, r = p
                if kind == "circle":
                    v = sqrt(c[0] ** 2 + c[1] ** 2) + r - S
                else:
                    v = r + point_poly_signed(c, bound)
                    for nt in notches:
                        v = max(v, circle_poly_overlap(c, r, nt))
                max_out = max(max_out, v)
            else:
                for vx, vy in p:
                    if kind == "circle":
                        v = sqrt(vx * vx + vy * vy) - S
                    else:
                        v = point_poly_signed([vx, vy], bound)
                    max_out = max(max_out, v)
                for nt in notches:
                    max_out = max(max_out, sat_overlap(p, nt))
    return max_overlap, max_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file")
    ap.add_argument("--dps", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)
    mo, mc = verify(data, args.dps, args.tol)
    s = mpf(repr(data["s"]))

    print(f"family: {data['family']}  n={data['n']}")
    print(f"claimed s            = {data['s']}")
    print(f"max pairwise overlap = {mp.nstr(mo, 6)}")
    print(f"max outside/notch    = {mp.nstr(mc, 6)}")
    worst = max(mo, mc, mpf(0))
    ok = worst < mpf(repr(args.tol))
    if ok:
        # safe quote: round s UP where the remaining slack cannot flip a digit
        pad = worst * 10 + mpf("1e-12")
        quote = mp.nstr(s + pad, 12)
        print(f"VERDICT: PASS (violations < {args.tol})")
        print(f"safe value to submit: s = {quote} (rounded up past worst-case "
              f"violation; quote fewer digits to be safer)")
    else:
        print(f"VERDICT: FAIL -- violations exceed {args.tol}; re-run the "
              f"endgame or increase --attempts. DO NOT SUBMIT.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
