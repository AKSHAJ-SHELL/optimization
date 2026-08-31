#!/usr/bin/env python3
"""
recordpacker -- shape-packing record hunter for Erich's Packing Center.

Upgrades over Flamethr0wer/polygon-packer:
  1. Exact gradients via JAX autodiff (theirs used finite differences through
     L-BFGS-B: ~3N function evals per gradient step).
  2. General shapes: tans, dominoes, L's (non-convex, convex-decomposed),
     circles, regular n-gons -- the families polygon-packer cannot express.
  3. General containers, including the non-convex L (modeled as bounding
     square + forbidden corner obstacle).
  4. Elite pool: the best configs found are perturbed (jiggle / rotation snap /
     swap / teleport) and re-shrunk, instead of only independent restarts.
  5. Endgame: tight re-polish with tiny shrink steps + strict feasibility
     audit (max overlap depth, max containment violation) before reporting.

Size conventions match Friedman's pages:
  tan       = right isosceles triangle, short side (leg) = 1
  square    = side 1
  domino    = 1 x 2 rectangle, short side 1
  L         = 2x2 square minus 1x1 corner, short side 1
  circle    = radius 1
  ngon:k    = regular k-gon with side 1
Containers use the same measure for s (reported value).

Examples:
  python packer.py --shape tan   --n 3  --container tan --attempts 50
  python packer.py --shape tan   --n 10 --container L   --attempts 200
  python packer.py --shape circle --n 12 --container tan --attempts 200
"""

import argparse
import functools
import json
import math
import os
import sys
import time

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize

try:
    from joblib import Parallel, delayed
    HAVE_JOBLIB = True
except ImportError:
    HAVE_JOBLIB = False

BIG = 1e18


# ----------------------------------------------------------------------------
# Shape / container catalog
# ----------------------------------------------------------------------------

def _centered(verts):
    v = np.asarray(verts, dtype=float)
    # centroid of polygon (area-weighted)
    x, y = v[:, 0], v[:, 1]
    xs, ys = np.roll(x, -1), np.roll(y, -1)
    cr = x * ys - xs * y
    a = cr.sum() / 2.0
    cx = ((x + xs) * cr).sum() / (6 * a)
    cy = ((y + ys) * cr).sum() / (6 * a)
    return v - np.array([cx, cy]), np.array([cx, cy])


def _poly_area(verts):
    v = np.asarray(verts, dtype=float)
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def shape_def(name):
    """Return dict(parts=[...], sym, area). Parts are given in the shape frame
    (about the shape's overall centroid). Part = dict(verts=(nv,2)) or
    dict(circle=True, center=(2,), radius=r)."""
    if name == "square":
        v = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
        return dict(parts=[dict(verts=np.array(v))], sym=4, area=1.0)
    if name == "domino":
        v = [(-1.0, -0.5), (1.0, -0.5), (1.0, 0.5), (-1.0, 0.5)]
        return dict(parts=[dict(verts=np.array(v))], sym=2, area=2.0)
    if name == "tan":
        v = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
        v = v - np.array([1.0 / 3, 1.0 / 3])
        return dict(parts=[dict(verts=v)], sym=1, area=0.5)
    if name == "circle":
        return dict(parts=[dict(circle=True, center=np.zeros(2), radius=1.0)],
                    sym=0, area=math.pi)
    if name == "L":
        c = np.array([5.0 / 6, 5.0 / 6])
        r1 = np.array([(0, 0), (2, 0), (2, 1), (0, 1)], dtype=float) - c
        r2 = np.array([(0, 1), (1, 1), (1, 2), (0, 2)], dtype=float) - c
        return dict(parts=[dict(verts=r1), dict(verts=r2)], sym=1, area=3.0)
    if name.startswith("ngon:"):
        k = int(name.split(":")[1])
        assert k >= 3
        rc = 1.0 / (2 * math.sin(math.pi / k))
        ang = 2 * math.pi * np.arange(k) / k - math.pi / 2 + math.pi / k
        v = rc * np.column_stack([np.cos(ang), np.sin(ang)])
        return dict(parts=[dict(verts=v)], sym=k,
                    area=k / (4 * math.tan(math.pi / k)))
    raise ValueError(f"unknown shape {name!r}")


def container_def(name):
    """Return dict(kind, verts?, obstacles=[verts], area1, mirror). All at
    S=1 in the family's s-measure. Polygon containers scale about origin."""
    if name == "square":
        v = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=float)
        return dict(kind="poly", verts=v, obstacles=[], area1=1.0)
    if name == "tan":
        v = np.array([(0, 0), (1, 0), (0, 1)], dtype=float)
        return dict(kind="poly", verts=v, obstacles=[], area1=0.5)
    if name == "domino":
        v = np.array([(0, 0), (2, 0), (2, 1), (0, 1)], dtype=float)
        return dict(kind="poly", verts=v, obstacles=[], area1=2.0)
    if name == "L":
        v = np.array([(0, 0), (2, 0), (2, 2), (0, 2)], dtype=float)
        ob = np.array([(1, 1), (2, 1), (2, 2), (1, 2)], dtype=float)
        return dict(kind="poly", verts=v, obstacles=[ob], area1=3.0,
                    outline=np.array([(0, 0), (2, 0), (2, 1), (1, 1),
                                      (1, 2), (0, 2)], dtype=float))
    if name == "circle":
        return dict(kind="circle", obstacles=[], area1=math.pi)
    if name.startswith("ngon:"):
        k = int(name.split(":")[1])
        rc = 1.0 / (2 * math.sin(math.pi / k))
        ang = 2 * math.pi * np.arange(k) / k - math.pi / 2 + math.pi / k
        v = rc * np.column_stack([np.cos(ang), np.sin(ang)])
        return dict(kind="poly", verts=v, obstacles=[],
                    area1=k / (4 * math.tan(math.pi / k)))
    raise ValueError(f"unknown container {name!r}")


# ----------------------------------------------------------------------------
# Problem build: flatten everything into padded arrays for JAX
# ----------------------------------------------------------------------------

def _edge_normals(verts):
    """Outward unit normals of a CCW convex polygon, one per edge."""
    v = np.asarray(verts, dtype=float)
    d = np.roll(v, -1, axis=0) - v
    n = np.column_stack([d[:, 1], -d[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return n


class Problem:
    def __init__(self, shape_name, n, container_name):
        self.shape_name, self.n, self.container_name = shape_name, n, container_name
        sd = shape_def(shape_name)
        cd = container_def(container_name)
        self.sym = sd["sym"]
        self.cont = cd
        self.N = n

        parts = []          # (shape_index, part dict)
        for i in range(n):
            for p in sd["parts"]:
                parts.append((i, p))
        n_real_parts = len(parts)
        for ob in cd["obstacles"]:
            parts.append((n, dict(verts=ob)))     # virtual shape n = obstacles
        P = len(parts)

        Vmax = max((len(p["verts"]) if "verts" in p else 1) for _, p in parts)
        pv = np.zeros((P, Vmax, 2))
        vmask = np.zeros((P, Vmax), dtype=bool)
        pnorm = np.zeros((P, Vmax, 2))
        nmask = np.zeros((P, Vmax), dtype=bool)
        ea = np.zeros((P, Vmax), dtype=int)
        eb = np.zeros((P, Vmax), dtype=int)
        emask = np.zeros((P, Vmax), dtype=bool)
        pcirc = np.zeros(P, dtype=bool)
        prad = np.zeros(P)
        pshape = np.zeros(P, dtype=int)

        for pi, (si, p) in enumerate(parts):
            pshape[pi] = si
            if p.get("circle"):
                pv[pi, 0] = p["center"]
                vmask[pi, 0] = True
                pcirc[pi] = True
                prad[pi] = p["radius"]
            else:
                v = np.asarray(p["verts"], dtype=float)
                nv = len(v)
                pv[pi, :nv] = v
                vmask[pi, :nv] = True
                pnorm[pi, :nv] = _edge_normals(v)
                nmask[pi, :nv] = True
                ea[pi, :nv] = np.arange(nv)
                eb[pi, :nv] = (np.arange(nv) + 1) % nv
                emask[pi, :nv] = True

        # pair lists (between different shapes; obstacles pair with all real)
        pp, cp, cc = [], [], []
        for i in range(P):
            for j in range(i + 1, P):
                if pshape[i] == pshape[j]:
                    continue
                if pshape[i] == n and pshape[j] == n:
                    continue
                ci, cj = pcirc[i], pcirc[j]
                if not ci and not cj:
                    pp.append((i, j))
                elif ci and cj:
                    cc.append((i, j))
                elif ci:
                    cp.append((i, j))
                else:
                    cp.append((j, i))

        self.arr = dict(
            pv=jnp.array(pv), vmask=jnp.array(vmask),
            pnorm=jnp.array(pnorm), nmask=jnp.array(nmask),
            ea=jnp.array(ea), eb=jnp.array(eb), emask=jnp.array(emask),
            pcirc=jnp.array(pcirc), prad=jnp.array(prad),
            pshape=jnp.array(pshape),
            pp=jnp.array(pp, dtype=int).reshape(-1, 2),
            cp=jnp.array(cp, dtype=int).reshape(-1, 2),
            cc=jnp.array(cc, dtype=int).reshape(-1, 2),
            n_real_parts=n_real_parts,
        )
        self._np = dict(pp=np.array(pp, dtype=int).reshape(-1, 2),
                        cp=np.array(cp, dtype=int).reshape(-1, 2),
                        cc=np.array(cc, dtype=int).reshape(-1, 2),
                        pshape=pshape.copy(), n_real=n_real_parts)

        if cd["kind"] == "poly":
            U = _edge_normals(cd["verts"])
            o1 = np.einsum("kd,kd->k", U, cd["verts"])
            self.arr["contU"] = jnp.array(U)
            self.arr["conto1"] = jnp.array(o1)
        self.kind = cd["kind"]

        self.shape_area = sd["area"]
        self.S_lower = math.sqrt(n * sd["area"] / cd["area1"])
        self._build_fns()

    # ------------------------------------------------------------------
    def _build_fns(self):
        a = self.arr
        N = self.N
        kind = self.kind
        n_real = a["n_real_parts"]

        def world(params, S):
            pose = jnp.concatenate([params.reshape(N, 3),
                                    jnp.zeros((1, 3))], axis=0)
            scales = jnp.concatenate([jnp.ones(N), jnp.array([S])])
            th = pose[:, 2]
            c, s = jnp.cos(th), jnp.sin(th)
            R = jnp.stack([jnp.stack([c, -s], -1),
                           jnp.stack([s, c], -1)], -2)      # (N+1,2,2)
            Rp = R[a["pshape"]]
            T = pose[a["pshape"], 0:2]
            sc = scales[a["pshape"]]
            W = T[:, None, :] + sc[:, None, None] * jnp.einsum(
                "pij,pvj->pvi", Rp, a["pv"])
            Nw = jnp.einsum("pij,pvj->pvi", Rp, a["pnorm"])
            rw = a["prad"] * sc
            return W, Nw, rw

        def pair_pp(W, Nw):
            ii, jj = a["pp"][:, 0], a["pp"][:, 1]
            if ii.shape[0] == 0:
                return jnp.zeros(0)
            axes = jnp.concatenate([Nw[ii], Nw[jj]], axis=1)          # (M,2V,2)
            amask = jnp.concatenate([a["nmask"][ii], a["nmask"][jj]], axis=1)

            def proj(Wk, mk):
                d = jnp.einsum("max,mvx->mav", axes, Wk)              # (M,A,V)
                mx = jnp.max(jnp.where(mk[:, None, :], d, -BIG), axis=2)
                mn = jnp.min(jnp.where(mk[:, None, :], d, BIG), axis=2)
                return mn, mx

            mn1, mx1 = proj(W[ii], a["vmask"][ii])
            mn2, mx2 = proj(W[jj], a["vmask"][jj])
            ov = jnp.minimum(mx1, mx2) - jnp.maximum(mn1, mn2)        # (M,A)
            ov = jnp.where(amask, ov, BIG)
            return jnp.min(ov, axis=1)          # signed: >0 overlap, <0 gap

        def pair_cp(W, Nw, rw):
            ci, pj = a["cp"][:, 0], a["cp"][:, 1]
            if ci.shape[0] == 0:
                return jnp.zeros(0)
            cen = W[ci, 0, :]                                          # (M,2)
            r = rw[ci]
            Wp = W[pj]                                                 # (M,V,2)
            nm = a["nmask"][pj]
            npx = Nw[pj]                                               # (M,V,2)
            # signed side: center . n_k - vert_k . n_k
            off = jnp.einsum("mvd,mvd->mv", npx, Wp)
            sdk = jnp.einsum("mvd,md->mv", npx, cen) - off
            inside_ind = jnp.max(jnp.where(nm, sdk, -BIG), axis=1)     # <=0 inside
            # distance to boundary (over edges)
            av = jnp.take_along_axis(Wp, a["ea"][pj][..., None], axis=1)
            bv = jnp.take_along_axis(Wp, a["eb"][pj][..., None], axis=1)
            em = a["emask"][pj]
            d = bv - av
            L2 = jnp.maximum(jnp.sum(d * d, axis=2), 1e-30)
            t = jnp.clip(jnp.einsum("mvd,mvd->mv",
                                    cen[:, None, :] - av, d) / L2, 0.0, 1.0)
            close = av + t[..., None] * d
            dist = jnp.sqrt(jnp.maximum(jnp.sum(
                (cen[:, None, :] - close) ** 2, axis=2), 1e-30))
            bd = jnp.min(jnp.where(em, dist, BIG), axis=1)
            sd = jnp.where(inside_ind <= 0.0, -bd, bd)
            return r - sd                        # signed

        def pair_cc(W, rw):
            ii, jj = a["cc"][:, 0], a["cc"][:, 1]
            if ii.shape[0] == 0:
                return jnp.zeros(0)
            d = W[ii, 0, :] - W[jj, 0, :]
            dist = jnp.sqrt(jnp.maximum(jnp.sum(d * d, axis=1), 1e-30))
            return rw[ii] + rw[jj] - dist        # signed

        def containment(W, rw, S):
            Wr = W[:n_real]
            vm = a["vmask"][:n_real]
            circ = a["pcirc"][:n_real]
            rr = rw[:n_real]
            if kind == "circle":
                dist = jnp.sqrt(jnp.maximum(jnp.sum(Wr * Wr, axis=2), 1e-30))
                viol = dist + jnp.where(circ[:, None], rr[:, None], 0.0) - S
                viol = jnp.where(vm, viol, -BIG)
                return viol.reshape(-1), jnp.max(viol, axis=1)
            U, o1 = a["contU"], a["conto1"]
            d = jnp.einsum("pvd,kd->pvk", Wr, U) - S * o1[None, None, :]
            d = d + jnp.where(circ[:, None, None], rr[:, None, None], 0.0)
            d = jnp.where(vm[:, :, None], d, -BIG)
            return d.reshape(-1), jnp.max(d, axis=(1, 2))

        def penalty(params, S):
            W, Nw, rw = world(params, S)
            cflat, _ = containment(W, rw, S)
            t = 0.0
            t = t + jnp.sum(jnp.maximum(pair_pp(W, Nw), 0.0) ** 2)
            t = t + jnp.sum(jnp.maximum(pair_cp(W, Nw, rw), 0.0) ** 2)
            t = t + jnp.sum(jnp.maximum(pair_cc(W, rw), 0.0) ** 2)
            t = t + jnp.sum(jnp.maximum(cflat, 0.0) ** 2)
            return t

        def audit(params, S):
            W, Nw, rw = world(params, S)
            mo = 0.0
            for arrv in (pair_pp(W, Nw), pair_cp(W, Nw, rw), pair_cc(W, rw)):
                if arrv.shape[0]:
                    mo = jnp.maximum(mo, jnp.max(arrv))
            mo = jnp.maximum(mo, 0.0)
            cflat, _ = containment(W, rw, S)
            mc = jnp.maximum(jnp.max(cflat), 0.0) if cflat.shape[0] else 0.0
            return mo, mc

        def gaps_raw(params, S):
            W, Nw, rw = world(params, S)
            _, cpart = containment(W, rw, S)
            return (pair_pp(W, Nw), pair_cp(W, Nw, rw),
                    pair_cc(W, rw), cpart)

        self._vg = jax.jit(jax.value_and_grad(penalty, argnums=0))
        self._audit = jax.jit(audit)
        self._gaps_raw = jax.jit(gaps_raw)
        self._world = world

    # ------------------------------------------------------------------
    def fun(self, x, S):
        f, g = self._vg(jnp.asarray(x), S)
        return float(f), np.asarray(g)

    def audit(self, x, S):
        mo, mc = self._audit(jnp.asarray(x), S)
        return float(mo), float(mc)

    def gaps(self, x, S):
        """Shape-level signed gaps: dict[(si,sj)] -> max gap value (>0 overlap,
        ~0 contact, <0 separated), plus per-shape wall gap (incl. L notch)."""
        pp, cp, cc, cpart = self._gaps_raw(jnp.asarray(x), S)
        pp, cp, cc, cpart = (np.asarray(v) for v in (pp, cp, cc, cpart))
        ps, N = self._np["pshape"], self.N
        pair_gaps, wall = {}, [-BIG] * N
        for arr, idx in ((pp, self._np["pp"]), (cp, self._np["cp"]),
                         (cc, self._np["cc"])):
            for g, (i, j) in zip(arr, idx):
                si, sj = int(ps[i]), int(ps[j])
                if si == N or sj == N:          # obstacle part -> wall contact
                    k = sj if si == N else si
                    wall[k] = max(wall[k], float(g))
                    continue
                key = (min(si, sj), max(si, sj))
                pair_gaps[key] = max(pair_gaps.get(key, -BIG), float(g))
        for p in range(self._np["n_real"]):
            si = int(ps[p])
            wall[si] = max(wall[si], float(cpart[p]))
        return pair_gaps, wall

    # -- geometry helpers (numpy, for init / plotting) ------------------
    def container_bbox(self, S):
        if self.kind == "circle":
            return -S, S, -S, S
        v = np.asarray(self.cont["verts"]) * S
        return v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()

    def point_ok(self, p, S, margin):
        if self.kind == "circle":
            if np.linalg.norm(p) > S - margin:
                return False
        else:
            U = np.asarray(self.arr["contU"])
            o1 = np.asarray(self.arr["conto1"])
            if np.any(p @ U.T > S * o1 - margin):
                return False
            for ob in self.cont["obstacles"]:
                Uo = _edge_normals(ob)
                oo = np.einsum("kd,kd->k", Uo, ob)
                if np.all(p @ Uo.T < S * oo + margin):
                    return False
        return True

    def shape_radius(self):
        sd = shape_def(self.shape_name)
        r = 0.0
        for p in sd["parts"]:
            if p.get("circle"):
                r = max(r, np.linalg.norm(p["center"]) + p["radius"])
            else:
                r = max(r, np.max(np.linalg.norm(p["verts"], axis=1)))
        return r


# ----------------------------------------------------------------------------
# Optimizer
# ----------------------------------------------------------------------------

def init_config(prob, S, rng, grid=False):
    N = prob.N
    x = np.zeros(N * 3)
    x0, x1, y0, y1 = prob.container_bbox(S)
    margin = 0.4 * prob.shape_radius()
    pts = []
    if grid:
        k = int(math.ceil(math.sqrt(N * (x1 - x0) / max(y1 - y0, 1e-9))))
        gx = np.linspace(x0, x1, k + 2)[1:-1]
        gy = np.linspace(y0, y1, k + 2)[1:-1]
        cand = [np.array([a, b]) for a in gx for b in gy
                if prob.point_ok(np.array([a, b]), S, margin)]
        rng.shuffle(cand)
        pts = cand[:N]
    while len(pts) < N:
        p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        if prob.point_ok(p, S, margin * 0.5):
            pts.append(p)
    for i, p in enumerate(pts):
        x[3 * i], x[3 * i + 1] = p
        x[3 * i + 2] = rng.uniform(0, 2 * math.pi)
    return x


def local_min(prob, x0, S, maxiter=400):
    res = minimize(prob.fun, x0, args=(S,), jac=True,
                   method="L-BFGS-B", tol=1e-16,
                   options=dict(maxiter=maxiter))
    return res.fun, res.x


def jitter(x, S, rng, pos_sig, ang_sig):
    x = x.copy()
    n = len(x) // 3
    x[0::3] += rng.normal(0, pos_sig * S, n)
    x[1::3] += rng.normal(0, pos_sig * S, n)
    x[2::3] += rng.normal(0, ang_sig, n)
    return x


def shrink_mult(S, S_low, range0, final_step):
    frac = final_step + max(S - S_low, 0.0) * (0.01 - final_step) / range0
    return 1.0 - frac


def solve_attempt(prob, seed, opts, x_init=None, S_init=None):
    rng = np.random.default_rng(seed)
    S_low = prob.S_lower
    if S_init is None:
        S = S_low * (1.35 + rng.random() * 1.0)
    else:
        S = S_init
    if x_init is None:
        x0 = init_config(prob, S, rng, grid=rng.random() < 0.5)
    else:
        x0 = x_init.copy()
    range0 = max(S - S_low, 1e-9)
    best = None
    steps = 0
    while steps < opts["max_steps"]:
        steps += 1
        f, x = local_min(prob, x0, S)
        if f < opts["ptol"]:
            best = (S, x.copy())
        else:
            fb, xb = f, x
            for _ in range(opts["bh_iters"]):
                ft, xt = local_min(prob, jitter(x, S, rng, 0.08, 0.5), S)
                if ft < fb:
                    fb, xb = ft, xt
                if fb < opts["ptol"]:
                    break
            if fb < opts["ptol"]:
                best, x = (S, xb.copy()), xb
            else:
                break
        m = shrink_mult(S, S_low, range0, opts["final_step"])
        x0 = x.copy()
        x0[0::3] *= m
        x0[1::3] *= m
        S *= m
    return best


# Elite-pool perturbations use the shared discrete move schema in moves.py.
# The same action space serves the random proposer, the LLM proposer, and the
# training data -- so "--proposer random" is an exact ablation baseline.
import moves as moves_mod


def endgame(prob, S, x, opts):
    """Tight polish: geometric-backoff shrink to squeeze digits, then audit."""
    ptol = 1e-18
    f, x = local_min(prob, x, S, maxiter=2000)
    while f >= ptol:                       # ensure a strictly clean start
        S *= 1.0 + 1e-6
        f, x = local_min(prob, x, S, maxiter=2000)
        if S > 10 * prob.S_lower:
            break
    delta = 1e-3
    while delta > 1e-11:
        St = S * (1 - delta)
        xt = x.copy()
        xt[0::3] *= (1 - delta)
        xt[1::3] *= (1 - delta)
        ft, xt = local_min(prob, xt, St, maxiter=2000)
        if ft < ptol:
            S, x = St, xt
        else:
            rng = np.random.default_rng(int(1 / delta))
            for _ in range(3):
                fr, xr = local_min(prob, jitter(xt, St, rng, 0.005, 0.03),
                                   St, maxiter=2000)
                if fr < ptol:
                    S, x, ft = St, xr, fr
                    break
            if ft >= ptol:
                delta /= 4.0
    # feasibility audit: inflate S until violations are negligible
    for _ in range(30):
        mo, mc = prob.audit(x, S)
        v = max(mo, mc)
        if v < 1e-11:
            break
        S *= (1.0 + 2.0 * v / max(S, 1e-9) + 1e-13)
        _, x = local_min(prob, x, S, maxiter=800)
    mo, mc = prob.audit(x, S)
    return S, x, mo, mc


# ----------------------------------------------------------------------------
# Worker plumbing
# ----------------------------------------------------------------------------

# Manual cache instead of @functools.lru_cache: the lru_cache C wrapper is not
# picklable, which breaks joblib/loky process workers (their __main__ differs,
# so the wrapper can't be resolved by reference or serialized by value). A plain
# dict cache is picklable and still lets each worker build the Problem (and its
# JIT-compiled JAX fns) once and reuse it across tasks.
_PROBLEM_CACHE = {}


def get_problem(shape, n, container):
    key = (shape, n, container)
    p = _PROBLEM_CACHE.get(key)
    if p is None:
        p = Problem(shape, n, container)
        _PROBLEM_CACHE[key] = p
    return p


def worker_attempt(shape, n, container, seed, opts_t):
    prob = get_problem(shape, n, container)
    return solve_attempt(prob, seed, dict(opts_t))


def worker_elite(shape, n, container, seed, opts_t, x, S,
                 move=None, log=None):
    """One elite variant: apply a move (given canonical move from the LLM, or
    a random one), re-shrink, optionally log the (state, move, outcome)."""
    prob = get_problem(shape, n, container)
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    state = moves_mod.canonical_state(prob, x, S) if (log or move) else None
    if move is not None:
        mv_canon = move
        mv_orig = moves_mod.unrelabel(move, state["order"])
    else:
        mv_orig = moves_mod.sample_move(rng, n)
        if state is not None:
            inv = {o: c for c, o in enumerate(state["order"])}
            mv_canon = moves_mod.relabel(mv_orig, inv)
    xp = moves_mod.apply_move(x, S, prob, mv_orig, rng)
    res = solve_attempt(prob, seed, dict(opts_t), x_init=xp, S_init=S * 1.02)
    if log:
        rec = dict(family=f"{shape} in {container}", n=n,
                   state={k: v for k, v in state.items() if k != "order"},
                   move=mv_canon, s_before=float(S),
                   s_after=(float(res[0]) if res else None),
                   improved=bool(res and res[0] < S - 1e-9))
        with open(f"{log}.{os.getpid()}", "a") as f:
            f.write(json.dumps(rec) + "\n")
    return res


def run_parallel(jobs, workers):
    if HAVE_JOBLIB and workers != 1:
        return Parallel(n_jobs=workers, prefer="processes")(
            delayed(fn)(*args) for fn, args in jobs)
    return [fn(*args) for fn, args in jobs]


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

def save_outputs(prob, S, x, mo, mc, prefix, record=None):
    data = dict(
        family=f"{prob.shape_name} in {prob.container_name}",
        n=prob.N, s=S,
        audit_max_overlap=mo, audit_max_outside=mc,
        record_reference=record,
        shapes=[dict(x=x[3 * i], y=x[3 * i + 1], theta=x[3 * i + 2])
                for i in range(prob.N)],
    )
    with open(prefix + ".json", "w") as f:
        json.dump(data, f, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPoly, Circle as MplCirc
        fig, ax = plt.subplots(figsize=(6, 6))
        if prob.kind == "circle":
            ax.add_patch(MplCirc((0, 0), S, fill=False, lw=1, color="k"))
        else:
            outline = prob.cont.get("outline", prob.cont["verts"])
            ax.add_patch(MplPoly(np.asarray(outline) * S, closed=True,
                                 fill=False, lw=1, color="k"))
        sd = shape_def(prob.shape_name)
        for i in range(prob.N):
            px, py, th = x[3 * i], x[3 * i + 1], x[3 * i + 2]
            c, s_ = math.cos(th), math.sin(th)
            R = np.array([[c, -s_], [s_, c]])
            for p in sd["parts"]:
                if p.get("circle"):
                    cc = np.array([px, py]) + R @ p["center"]
                    ax.add_patch(MplCirc(cc, p["radius"], facecolor="#cccccc",
                                         edgecolor="k", lw=0.5))
                else:
                    w = np.array([px, py]) + p["verts"] @ R.T
                    ax.add_patch(MplPoly(w, closed=True, facecolor="#cccccc",
                                         edgecolor="k", lw=0.5))
        ax.set_aspect("equal")
        ax.autoscale()
        ax.set_title(f"{prob.N} {prob.shape_name} in {prob.container_name}: "
                     f"s = {S:.9f}")
        fig.savefig(prefix + ".png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:  # plotting is best-effort
        print("plot failed:", e)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", required=True,
                    help="tan | square | domino | circle | L | ngon:k")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--container", required=True,
                    help="tan | square | domino | circle | L | ngon:k")
    ap.add_argument("--attempts", type=int, default=100)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ptol", type=float, default=1e-10)
    ap.add_argument("--finalstep", type=float, default=1e-5)
    ap.add_argument("--bh-iters", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--elite-rounds", type=int, default=2)
    ap.add_argument("--elite-k", type=int, default=6)
    ap.add_argument("--elite-variants", type=int, default=8)
    ap.add_argument("--record", type=float, default=None,
                    help="current record s to compare against")
    ap.add_argument("--out", default=None, help="output file prefix")
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke-test settings")
    ap.add_argument("--log", default=None,
                    help="JSONL path prefix: log (state, move, outcome) from "
                         "every elite variant for policy training")
    ap.add_argument("--proposer", choices=["random", "llm"], default="random",
                    help="who proposes elite-pool moves")
    ap.add_argument("--model", default=None,
                    help="MLX model path/repo for --proposer llm")
    ap.add_argument("--llm-temp", type=float, default=0.8)
    args = ap.parse_args()

    opts = dict(ptol=args.ptol, final_step=args.finalstep,
                bh_iters=args.bh_iters, max_steps=args.max_steps)
    if args.quick:
        opts.update(final_step=1e-3, bh_iters=4, max_steps=400)
        args.elite_rounds = min(args.elite_rounds, 1)
        args.elite_variants = min(args.elite_variants, 4)

    prob = get_problem(args.shape, args.n, args.container)
    opts_t = tuple(sorted(opts.items()))
    t0 = time.time()

    print(f"[{args.n} x {args.shape} in {args.container}]  "
          f"area lower bound s >= {prob.S_lower:.6f}")

    jobs = [(worker_attempt, (args.shape, args.n, args.container,
                              args.seed * 100003 + i, opts_t))
            for i in range(args.attempts)]
    results = [r for r in run_parallel(jobs, args.workers) if r is not None]
    if not results:
        print("no feasible packing found; increase --attempts")
        sys.exit(1)
    results.sort(key=lambda r: r[0])
    print(f"initial pool best s = {results[0][0]:.9f}  "
          f"({len(results)} feasible, {time.time()-t0:.0f}s)")

    # elite rounds
    proposer = None
    if args.proposer == "llm":
        from llm_proposer import LLMProposer
        proposer = LLMProposer(args.model, temperature=args.llm_temp)
    pool = results[:max(args.elite_k, 1)]
    for rnd in range(args.elite_rounds):
        jobs = []
        for ei, (S, x) in enumerate(pool):
            if proposer is not None:
                st = moves_mod.canonical_state(prob, np.asarray(x), S)
                mvs = proposer.propose(st, args.elite_variants)
            else:
                mvs = [None] * args.elite_variants
            for v, mv in enumerate(mvs):
                sd_ = args.seed * 900001 + rnd * 7919 + ei * 613 + v + 1
                jobs.append((worker_elite, (args.shape, args.n, args.container,
                                            sd_, opts_t, tuple(x), S,
                                            mv, args.log)))
        newr = [r for r in run_parallel(jobs, args.workers) if r is not None]
        pool = sorted(pool + newr, key=lambda r: r[0])[:max(args.elite_k, 1)]
        print(f"elite round {rnd + 1}: best s = {pool[0][0]:.9f}")

    S, x = pool[0]
    S, x, mo, mc = endgame(prob, S, np.asarray(x), opts)

    print()
    print(f"BEST:  s = {S:.10f}")
    print(f"audit: max pairwise overlap = {mo:.2e}, "
          f"max containment violation = {mc:.2e}")
    if args.record is not None:
        d = args.record - S
        tag = "BEATS RECORD by" if d > 0 else "misses record by"
        print(f"record s = {args.record}:  {tag} {abs(d):.6g}")

    prefix = args.out or f"{args.n}_{args.shape.replace(':', '')}_in_" \
                         f"{args.container.replace(':', '')}"
    save_outputs(prob, S, x, mo, mc, prefix, args.record)
    print(f"wrote {prefix}.json and {prefix}.png  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
