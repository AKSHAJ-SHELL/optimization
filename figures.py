#!/usr/bin/env python3
"""Springer-format figure + data generator for the manuscript.

Every figure is written as .eps + .pdf (vector, for submission) and .png
(600 dpi, for preview), sized to Springer column widths (84 mm single /
174 mm double), sans-serif labels, thin lines. Each figure also writes its
underlying numbers to figures/data/<name>.csv for the data-availability
declaration.

Figures (all from real runs/logs except the ablation placeholder):
  gallery      best-known packings produced by the engine (vector redraws)
  landscape    record age by family/n from the registry scrape (why tans/L's)
  restarts     distribution of per-restart final s (tan family instance)
  convergence  shrink-loop trace: gap to area lower bound vs step
  moves        elite-move improvement rate by op, from --log data
  gradcheck    autodiff vs finite-difference gradient agreement
  ablation     LLM vs random (reads results.csv if present, else placeholder)

Usage:
  python figures.py --all            # full budgets (minutes)
  python figures.py --all --quick    # small budgets (sandbox/smoke)
  python figures.py --only restarts convergence
"""

import argparse
import csv
import glob
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Circle as MplCirc

MM = 1 / 25.4
SINGLE = 84 * MM          # Springer single-column width
DOUBLE = 174 * MM         # full text width
FIGDIR = "figures"
DATADIR = os.path.join(FIGDIR, "data")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 0.9, "axes.linewidth": 0.5,
    "grid.linewidth": 0.3, "grid.alpha": 0.4,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

C = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


def save(fig, name, data_rows=None, data_header=None):
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(DATADIR, exist_ok=True)
    for ext in ("eps", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"))
    fig.savefig(os.path.join(FIGDIR, f"{name}.png"), dpi=600)
    plt.close(fig)
    if data_rows is not None:
        with open(os.path.join(DATADIR, f"{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            if data_header:
                w.writerow(data_header)
            w.writerows(data_rows)
    print(f"[figures] wrote {name}.eps/.pdf/.png"
          + (" + data csv" if data_rows is not None else ""))


# ---------------------------------------------------------------- gallery ---
def fig_gallery(args):
    import packer
    files = [f for f in args.gallery_jsons if os.path.exists(f)]
    if not files:
        print("[figures] gallery: no packing JSONs found; skipping")
        return
    k = len(files)
    fig, axes = plt.subplots(1, k, figsize=(DOUBLE, DOUBLE / k * 0.95))
    if k == 1:
        axes = [axes]
    rows = []
    for ax, f in zip(axes, files):
        d = json.load(open(f))
        shape_name, cont_name = d["family"].split(" in ")
        S = d["s"]
        cont = packer.container_def(cont_name)
        if cont["kind"] == "circle":
            ax.add_patch(MplCirc((0, 0), S, fill=False, lw=0.7, color="k"))
        else:
            outline = cont.get("outline", cont["verts"])
            ax.add_patch(MplPoly(np.asarray(outline) * S, closed=True,
                                 fill=False, lw=0.7, color="k"))
        sd = packer.shape_def(shape_name)
        for sh in d["shapes"]:
            x, y, th = sh["x"], sh["y"], sh["theta"]
            c_, s_ = math.cos(th), math.sin(th)
            R = np.array([[c_, -s_], [s_, c_]])
            for p in sd["parts"]:
                if p.get("circle"):
                    cc = np.array([x, y]) + R @ p["center"]
                    ax.add_patch(MplCirc(cc, p["radius"], facecolor="#d9d9d9",
                                         edgecolor="k", lw=0.5))
                else:
                    ax.add_patch(MplPoly(np.array([x, y]) + p["verts"] @ R.T,
                                         closed=True, facecolor="#d9d9d9",
                                         edgecolor="k", lw=0.5))
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.autoscale()
        ax.set_title(f'$n={d["n"]}$ {shape_name} in {cont_name}\n'
                     f'$s={S:.6f}$', fontsize=7)
        rows.append([d["family"], d["n"], f"{S:.10f}",
                     d.get("audit_max_overlap"), d.get("audit_max_outside")])
    save(fig, "fig_gallery", rows,
         ["family", "n", "s", "audit_overlap", "audit_outside"])


# -------------------------------------------------------------- landscape ---
REG = {  # record year by n, from the 2026-07 registry scrape (see TARGETS.md)
    "tans in tans": {3: 2005, 5: 2005, 6: 2007, 7: 2007, 10: 2007, 11: 2005,
                     12: 2007, 13: 2005, 14: 2005, 15: 2007, 17: 2007,
                     19: 2005, 20: 2007, 21: 2007, 22: 2009, 23: 2007,
                     24: 2007, 26: 2009, 27: 2009},
    "tans in L's": {3: 2011, 4: 2012, 7: 2012, 8: 2012, 9: 2012, 10: 2012,
                    13: 2012, 14: 2012, 15: 2012, 16: 2012, 17: 2012},
    "L's in tans": {4: 2012, 5: 2012, 13: 2012, 17: 2025, 19: 2025, 20: 2025},
    "circles in tans": {2: 1996, 3: 1996, 4: 1996, 5: 1997, 6: 1997, 7: 1996,
                        8: 1997, 11: 2005, 12: 2005, 13: 1997, 14: 2005,
                        16: 2006, 17: 2006, 18: 2009, 19: 2006, 20: 2008},
    "octagons in tans": {5: 2026, 7: 2026, 8: 2026, 11: 2026, 12: 2026,
                         13: 2026, 14: 2026, 17: 2026, 18: 2026, 19: 2026,
                         20: 2026, 22: 2026},
}


def fig_landscape(args):
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.5))
    rows = []
    for i, (fam, d) in enumerate(REG.items()):
        ns, ys = zip(*sorted(d.items()))
        ax.scatter(ns, ys, s=11, color=C[i], label=fam, zorder=3,
                   marker="osD^v"[i])
        rows += [[fam, n, y] for n, y in sorted(d.items())]
    ax.axhspan(2025.5, 2026.6, color="#d62728", alpha=0.08, lw=0)
    ax.text(2.2, 2023.6, "2026 rush", fontsize=6.5, color="#d62728",
            va="center")
    ax.annotate("", xy=(2.7, 2025.7), xytext=(3.4, 2024.1),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.6))
    ax.set_xlabel("$n$")
    ax.set_ylabel("year of current record")
    ax.set_ylim(1994.5, 2027.5)
    ax.grid(True)
    ax.legend(ncol=5, frameon=False, fontsize=6.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    save(fig, "fig_landscape", rows, ["family", "n", "record_year"])


# --------------------------------------------------------------- restarts ---
def fig_restarts(args):
    import packer
    shape, n, cont = args.inst
    prob = packer.get_problem(shape, int(n), cont)
    opts = dict(ptol=1e-10, final_step=1e-3 if args.quick else 1e-5,
                bh_iters=3 if args.quick else 20,
                max_steps=300 if args.quick else 4000)
    seeds = args.restart_seeds
    vals = []
    for sd in range(seeds):
        r = packer.solve_attempt(prob, sd, opts)
        if r is not None:
            vals.append(r[0])
    vals = np.array(sorted(vals))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE, 1.9))
    a1.hist(vals, bins=max(6, seeds // 4), color=C[0], edgecolor="k",
            linewidth=0.4)
    a1.axvline(prob.S_lower, color=C[1], ls="--", lw=0.8,
               label="area lower bound")
    a1.set_xlabel("final $s$ per restart")
    a1.set_ylabel("count")
    a1.legend(frameon=False, fontsize=6, loc="upper right")
    a2.plot(vals, np.arange(1, len(vals) + 1) / len(vals), color=C[0],
            drawstyle="steps-post")
    a2.axvline(vals[0], color=C[2], ls=":", lw=0.8, label="best restart")
    a2.set_xlabel("final $s$")
    a2.set_ylabel("empirical CDF")
    a2.legend(frameon=False, fontsize=6, loc="lower right")
    fig.suptitle(f"{n} {shape}s in {cont}: {len(vals)} restarts", fontsize=8)
    save(fig, "fig_restarts", [[i, v] for i, v in enumerate(vals)],
         ["restart_rank", "final_s"])


# ------------------------------------------------------------ convergence ---
def fig_convergence(args):
    import packer
    shape, n, cont = args.inst
    prob = packer.get_problem(shape, int(n), cont)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.1))
    rows = []
    for k, sd in enumerate(range(args.trace_seeds)):
        rng = np.random.default_rng(sd)
        S = prob.S_lower * (1.35 + rng.random())
        x0 = packer.init_config(prob, S, rng, grid=sd % 2 == 0)
        range0 = max(S - prob.S_lower, 1e-9)
        trace = []
        for step in range(300 if args.quick else 3000):
            f, x = packer.local_min(prob, x0, S)
            if f < 1e-10:
                trace.append((step, S))
                m = packer.shrink_mult(S, prob.S_lower, range0,
                                       1e-3 if args.quick else 1e-5)
                x0 = x.copy()
                x0[0::3] *= m
                x0[1::3] *= m
                S *= m
            else:
                ok = False
                for _ in range(3 if args.quick else 20):
                    f2, x2 = packer.local_min(
                        prob, packer.jitter(x, S, rng, 0.08, 0.5), S)
                    if f2 < 1e-10:
                        trace.append((step, S))
                        x0, ok = x2, True
                        break
                if not ok:
                    break
        t = np.array(trace)
        ax.semilogy(t[:, 0], t[:, 1] - prob.S_lower, color=C[k % len(C)],
                    lw=0.8, label=f"restart {sd}")
        rows += [[sd, int(s_), f"{v:.10f}"] for s_, v in trace]
    ax.set_xlabel("accepted shrink step")
    ax.set_ylabel(r"$s - s_{\mathrm{lower}}$")
    ax.grid(True, which="both")
    ax.legend(frameon=False)
    ax.set_title(f"{n} {shape}s in {cont}", fontsize=8)
    save(fig, "fig_convergence", rows, ["seed", "step", "s"])


# ------------------------------------------------------------------ moves ---
def fig_moves(args):
    recs = []
    for path in glob.glob(args.logs):
        with open(path) as f:
            recs += [json.loads(l) for l in f if l.strip()]
    if not recs:
        print("[figures] moves: no logs found; skipping")
        return
    ops = {}
    for r in recs:
        op = r["move"]["op"]
        tot, imp = ops.get(op, (0, 0))
        ops[op] = (tot + 1, imp + (1 if r["improved"] else 0))
    names = sorted(ops)
    rates = [ops[o][1] / ops[o][0] for o in names]
    counts = [ops[o][0] for o in names]
    fig, ax = plt.subplots(figsize=(SINGLE, 1.9))
    bars = ax.bar(names, rates, color=C[0], edgecolor="k", linewidth=0.4)
    top = max(rates)
    for b, c_ in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * 0.03,
                f"{c_:,}", ha="center", fontsize=6)
    ax.set_ylabel("improvement rate")
    ax.set_ylim(0, top * 1.18)
    ax.grid(True, axis="y")
    save(fig, "fig_moves",
         [[o, ops[o][0], ops[o][1], ops[o][1] / ops[o][0]] for o in names],
         ["op", "proposals", "improvements", "rate"])


# ---------------------------------------------------------------- dataset ---
def fig_dataset(args):
    """Training-data figure: composition, improvement magnitudes, signal vs n."""
    recs = []
    for path in glob.glob(args.logs):
        with open(path) as f:
            recs += [json.loads(l) for l in f if l.strip()]
    if not recs:
        print("[figures] dataset: no logs found; skipping")
        return
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(DOUBLE, 2.1),
                                     constrained_layout=True)
    rows = []

    # (a) records per family, stacked improved / not
    ab = {"circle": "cir", "square": "sq", "domino": "dom", "tan": "tan",
          "L": "L"}
    fams = sorted({r["family"] for r in recs})
    imp = [sum(1 for r in recs if r["family"] == f and r["improved"])
           for f in fams]
    non = [sum(1 for r in recs if r["family"] == f and not r["improved"])
           for f in fams]
    short = ["/".join(ab.get(w, w) for w in f.split(" in ")) for f in fams]
    a1.bar(short, non, color="#c7c7c7", edgecolor="k", linewidth=0.4,
           label="non-improving")
    a1.bar(short, imp, bottom=non, color=C[2], edgecolor="k", linewidth=0.4,
           label="improving")
    a1.tick_params(axis="x", labelsize=5.5, rotation=60)
    a1.set_ylabel("logged records")
    a1.legend(frameon=False, fontsize=6)
    rows += [["records_by_family", f, i, nn] for f, i, nn in
             zip(fams, imp, non)]

    # (b) distribution of relative improvement among successful moves
    dimp = [(r["s_before"] - r["s_after"]) / r["s_before"]
            for r in recs if r["improved"] and r["s_after"]]
    if dimp:
        a2.hist(np.log10(np.clip(dimp, 1e-12, None)),
                bins=15, color=C[0], edgecolor="k", linewidth=0.4)
    a2.set_xlabel(r"$\log_{10}$ relative improvement $\Delta s / s$")
    a2.set_ylabel("count")
    rows += [["rel_improvement", f"{v:.3e}", "", ""] for v in dimp]

    # (c) improvement rate vs n (does the signal survive larger instances?)
    ns = sorted({r["n"] for r in recs})
    rate = [np.mean([r["improved"] for r in recs if r["n"] == n_])
            for n_ in ns]
    cnt = [sum(1 for r in recs if r["n"] == n_) for n_ in ns]
    a3.plot(ns, rate, "o-", color=C[1], ms=3, lw=0.8)
    a3.set_xlabel("$n$")
    a3.set_ylabel("improvement rate")
    a3.set_ylim(0, None)
    a3.grid(True)
    rows += [["rate_by_n", n_, f"{r_:.4f}", c_] for n_, r_, c_ in
             zip(ns, rate, cnt)]

    fig.suptitle(f"training data: {len(recs)} logged (state, move, outcome) "
                 f"records", fontsize=8)
    save(fig, "fig_dataset", rows, ["series", "key", "value", "count"])


# -------------------------------------------------------------- gradcheck ---
def fig_gradcheck(args):
    import packer
    cases = [("tan", 3, "tan"), ("circle", 3, "tan"),
             ("tan", 4, "L"), ("square", 4, "square")]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.0))
    rows = []
    for ci, fam in enumerate(cases):
        prob = packer.get_problem(*fam)
        rng = np.random.default_rng(1)
        errs = []
        for t in range(args.grad_states):
            S = prob.S_lower * (1.5 + rng.random())
            x = packer.init_config(prob, S, rng)
            f, g = prob.fun(x, S)
            eps = 1e-7
            gfd = np.zeros_like(x)
            for i in range(len(x)):
                xp, xm = x.copy(), x.copy()
                xp[i] += eps
                xm[i] -= eps
                gfd[i] = (prob.fun(xp, S)[0] - prob.fun(xm, S)[0]) / (2 * eps)
            denom = max(1e-12, np.max(np.abs(gfd)))
            err = np.max(np.abs(g - gfd)) / denom
            errs.append(err)
            rows.append([f"{fam[1]} {fam[0]} in {fam[2]}", t, f"{err:.3e}"])
        ax.scatter([ci] * len(errs), errs, s=8, color=C[0])
    ax.set_yscale("log")
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([f"{n} {s}\nin {c}" for s, n, c in cases], fontsize=6)
    ax.set_ylabel("max rel. gradient error\n(autodiff vs central diff)")
    ax.grid(True, axis="y", which="both")
    save(fig, "fig_gradcheck", rows, ["instance", "state", "rel_error"])


# --------------------------------------------------------------- ablation ---
def fig_ablation(args):
    fig, ax = plt.subplots(figsize=(SINGLE, 2.0))
    if os.path.exists(args.results):
        rows = list(csv.DictReader(open(args.results)))
        insts = sorted({(r["shape"], r["n"], r["container"]) for r in rows})
        xs = np.arange(len(insts))
        for j, prop in enumerate(("random", "llm")):
            gaps = []
            for inst in insts:
                g = [float(r["gap"]) for r in rows
                     if (r["shape"], r["n"], r["container"]) == inst
                     and r["proposer"] == prop]
                gaps.append(np.mean(g) if g else np.nan)
            ax.bar(xs + (j - 0.5) * 0.38, gaps, width=0.36,
                   color=C[j], edgecolor="k", linewidth=0.4, label=prop)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{n} {s}\nin {c}" for s, n, c in insts],
                           fontsize=6)
        ax.set_ylabel("mean gap to registry $s$")
        ax.legend(frameon=False)
        data = [[*i] for i in insts]
    else:
        ax.text(0.5, 0.5, "PENDING\nrun ablation.py to populate\n"
                          "(reads results.csv)", ha="center", va="center",
                fontsize=9, color="#d62728", transform=ax.transAxes)
        ax.set_axis_off()
        data = [["pending"]]
    ax.grid(True, axis="y")
    save(fig, "fig_ablation", data)


# --------------------------------------------------------------- training ---
def fig_training(args):
    """Parse the MLX training log: train/val loss curves, best checkpoint,
    divergence point. Real data from sft.log."""
    import re
    if not os.path.exists(args.sft_log):
        print(f"[figures] training: {args.sft_log} not found; skipping")
        return
    tr, va = [], []
    for line in open(args.sft_log):
        m = re.match(r"Iter (\d+): Train loss ([\d.]+|nan)", line)
        if m:
            tr.append((int(m.group(1)), float(m.group(2))))
        m = re.match(r"Iter (\d+): Val loss ([\d.]+|nan)", line)
        if m:
            va.append((int(m.group(1)), float(m.group(2))))
    if not tr:
        print("[figures] training: no loss lines parsed; skipping")
        return
    tr_ok = [(i, v) for i, v in tr if np.isfinite(v)]
    va_ok = [(i, v) for i, v in va if np.isfinite(v)]
    nan_start = min([i for i, v in tr if not np.isfinite(v)] or [None],
                    key=lambda x: x if x is not None else 1 << 30)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.2))
    ax.plot(*zip(*tr_ok), color=C[0], lw=0.7, alpha=0.8, label="train loss")
    ax.plot(*zip(*va_ok), "o-", color=C[1], lw=0.9, ms=3, label="val loss")
    best_i, best_v = min(va_ok, key=lambda t: t[1])
    ax.axvline(500, color=C[2], ls="--", lw=0.8,
               label="deployed checkpoint (500)")
    ax.annotate(f"best val {best_v:.3f}", (best_i, best_v), fontsize=6,
                textcoords="offset points", xytext=(6, -10))
    if nan_start is not None:
        ax.axvspan(nan_start, max(i for i, _ in tr), color="#d62728",
                   alpha=0.10, lw=0)
        ax.text(nan_start + 30, ax.get_ylim()[1] * 0.75, "NaN\ndivergence",
                fontsize=6, color="#d62728")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid(True, which="both")
    ax.legend(frameon=False, fontsize=6)
    save(fig, "fig_training",
         [["train", i, v] for i, v in tr] + [["val", i, v] for i, v in va],
         ["series", "iter", "loss"])


# ---------------------------------------------------------------- records ---
def fig_records(args):
    """Gallery of the registry results (beats + new entries), page-style."""
    import packer
    items = []
    for n in args.record_ns:
        f = f"runs/{n}_ngon3_in_tan.json"
        if os.path.exists(f):
            items.append(json.load(open(f)))
    if not items:
        print("[figures] records: no run JSONs found; skipping")
        return
    cols = 3 if len(items) <= 6 else 5
    rows_n = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols,
                             figsize=(DOUBLE, DOUBLE / cols * rows_n * 1.08))
    axes = np.atleast_2d(axes)
    data = []
    for k, d in enumerate(items):
        ax = axes[k // cols][k % cols]
        S = d["s"]
        cont = packer.container_def("tan")
        ax.add_patch(MplPoly(np.asarray(cont["verts"]) * S, closed=True,
                             fill=False, lw=0.7, color="k"))
        sd = packer.shape_def("ngon:3")
        for sh in d["shapes"]:
            x, y, th = sh["x"], sh["y"], sh["theta"]
            c_, s_ = math.cos(th), math.sin(th)
            R = np.array([[c_, -s_], [s_, c_]])
            ax.add_patch(MplPoly(np.array([x, y]) + sd["parts"][0]["verts"] @ R.T,
                                 closed=True, facecolor="#d9daeb",
                                 edgecolor="#3b3b47", lw=0.4))
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.autoscale()
        ax.set_title(f'$n={d["n"]}$: $s={S:.6f}$', fontsize=6.5)
        data.append([d["n"], f"{S:.10f}"])
    for k in range(len(items), rows_n * cols):
        axes[k // cols][k % cols].set_axis_off()
    save(fig, "fig_records", data, ["n", "s"])


# ---------------------------------------------------------------- summary ---
def fig_summary(args):
    """Best found s vs n per family, against area lower bound and registry
    page values. Ingests every result JSON in runs/."""
    from sweep import TANINTAN, TANINL, TRIINTAN
    fams = {"tan in tan": ("tans in tans", TANINTAN,
                           lambda n: math.sqrt(n)),
            "tan in L": ("tans in L's", TANINL,
                         lambda n: math.sqrt(n / 6.0)),
            "ngon:3 in tan": ("triangles in tans", TRIINTAN,
                              lambda n: math.sqrt(n * math.sqrt(3) / 2))}
    best = {k: {} for k in fams}
    for f in glob.glob("runs/*.json") + glob.glob("*.json"):
        try:
            d = json.load(open(f))
            fam, n, s = d["family"], d["n"], d["s"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if fam in best and (n not in best[fam] or s < best[fam][n]):
            best[fam][n] = s
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.1),
                             constrained_layout=True)
    rows = []
    for ax, (fam, (title, rec, lb)) in zip(axes, fams.items()):
        ns = sorted(best[fam])
        if not ns:
            ax.set_axis_off()
            continue
        xs = np.linspace(min(ns), max(ns), 100)
        ax.plot(xs, [lb(x) for x in xs], color="#999999", lw=0.7,
                label="area lower bound")
        rn = sorted(k for k in rec if min(ns) <= k <= max(ns))
        ax.plot(rn, [rec[k] for k in rn], "x", color=C[1], ms=4,
                label="registry page value")
        ax.plot(ns, [best[fam][n] for n in ns], "o", color=C[0], ms=3,
                label="this work (best)")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("$n$")
        ax.grid(True)
        rows += [[fam, n, f"{best[fam][n]:.9f}", rec.get(n, "")] for n in ns]
    axes[0].set_ylabel("$s$")
    axes[0].legend(frameon=False, fontsize=6)
    save(fig, "fig_summary", rows, ["family", "n", "best_s", "page_value"])


# ------------------------------------------------------------------ seeds ---
def fig_seeds(args):
    """Per-seed ablation outcomes + wall-clock comparison, from results.csv."""
    if not os.path.exists(args.results):
        print("[figures] seeds: no results.csv; skipping")
        return
    rows = list(csv.DictReader(open(args.results)))
    insts = sorted({(r["shape"], int(r["n"]), r["container"]) for r in rows},
                   key=lambda t: (t[2], t[1]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.2),
                                 gridspec_kw={"width_ratios": [2.2, 1]},
                                 constrained_layout=True)
    data = []
    for xi, inst in enumerate(insts):
        for prop, col, mk, dx in (("random", C[0], "o", -0.13),
                                  ("llm", C[1], "s", 0.13)):
            g = [max(float(r["gap"]), 1e-5) for r in rows
                 if (r["shape"], int(r["n"]), r["container"]) == inst
                 and r["proposer"] == prop]
            a1.scatter([xi + dx] * len(g), g, s=14, color=col, marker=mk,
                       alpha=0.85, label=prop if xi == 0 else None)
            data += [[*inst, prop, v] for v in g]
    a1.set_yscale("log")
    a1.set_xticks(range(len(insts)))
    a1.set_xticklabels([f"{n} {s}\nin {c}" for s, n, c in insts], fontsize=6)
    a1.set_ylabel("gap to page value (log)")
    a1.grid(True, axis="y", which="both")
    a1.legend(frameon=False, fontsize=6, loc="lower right")
    secs = {p: [float(r["seconds"]) for r in rows if r["proposer"] == p]
            for p in ("random", "llm")}
    bp = a2.boxplot([secs["random"], secs["llm"]], tick_labels=["random", "llm"],
                    widths=0.5, patch_artist=True, medianprops=dict(color="k"))
    for patch, col in zip(bp["boxes"], (C[0], C[1])):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    a2.set_ylabel("wall-clock per run (s)")
    a2.grid(True, axis="y")
    save(fig, "fig_seeds", data, ["shape", "n", "container", "proposer", "gap"])


# ------------------------------------------------------------------ beats ---
def fig_beats(args):
    """Margins by which the three 2026 records were beaten."""
    page = {21: 4.65799, 22: 4.73289, 23: 4.85470}
    ours, rows = {}, []
    for n in page:
        f = f"runs/{n}_ngon3_in_tan.json"
        if os.path.exists(f):
            ours[n] = json.load(open(f))["s"]
    if not ours:
        print("[figures] beats: no run JSONs; skipping")
        return
    fig, ax = plt.subplots(figsize=(SINGLE, 1.8))
    ns = sorted(ours)
    margins = [page[n] - ours[n] for n in ns]
    bars = ax.barh([f"$n={n}$" for n in ns], margins, color=C[2],
                   edgecolor="k", linewidth=0.4, height=0.55)
    for b, n, m in zip(bars, ns, margins):
        ax.text(m * 1.15, b.get_y() + b.get_height() / 2,
                f"{page[n]:.5f} $\\to$ {ours[n]:.5f}", va="center", fontsize=6)
        rows.append([n, page[n], f"{ours[n]:.9f}", f"{m:.6f}"])
    ax.set_xscale("log")
    ax.set_xlim(right=max(margins) * 30)
    ax.set_xlabel("improvement over June--July 2026 record (log)")
    ax.grid(True, axis="x", which="both")
    save(fig, "fig_beats", rows, ["n", "page_s", "our_s", "margin"])


# ---------------------------------------------------------------- density ---
def fig_density(args):
    """Packing density vs n, triangles-in-tans: this work vs registry."""
    from sweep import TRIINTAN
    tri_area = math.sqrt(3) / 4
    dens = lambda n, s: n * tri_area / (s * s / 2)
    ours, rows = {}, []
    for f in glob.glob("runs/*_ngon3_in_tan.json"):
        d = json.load(open(f))
        n, s = d["n"], d["s"]
        if n not in ours or s < ours[n]:
            ours[n] = s
    if not ours:
        print("[figures] density: no runs; skipping")
        return
    fig, ax = plt.subplots(figsize=(SINGLE, 2.1))
    rn = sorted(k for k in TRIINTAN)
    ax.plot(rn, [dens(k, TRIINTAN[k]) for k in rn], "x", color=C[1], ms=4,
            label="registry page value")
    on = sorted(ours)
    ax.plot(on, [dens(k, ours[k]) for k in on], "o-", color=C[0], ms=3,
            lw=0.7, label="this work (best)")
    rows += [[k, f"{dens(k, ours[k]):.4f}",
              f"{dens(k, TRIINTAN[k]):.4f}" if k in TRIINTAN else ""]
             for k in on]
    ax.set_xlabel("$n$")
    ax.set_ylabel("packing density")
    ax.grid(True)
    ax.legend(frameon=False, fontsize=6)
    save(fig, "fig_density", rows, ["n", "density_ours", "density_page"])


# -------------------------------------------------------------- opsfamily ---
def fig_opsfamily(args):
    """Heatmap: improvement rate of each move operator per family."""
    recs = []
    for path in glob.glob(args.logs):
        with open(path) as f:
            recs += [json.loads(l) for l in f if l.strip()]
    if not recs:
        print("[figures] opsfamily: no logs; skipping")
        return
    ab = {"circle": "cir", "square": "sq", "domino": "dom", "tan": "tan",
          "L": "L", "ngon:3": "tri"}
    fams = sorted({r["family"] for r in recs})
    ops = ["jiggle", "relocate", "rotate", "swap"]
    M = np.zeros((len(ops), len(fams)))
    for oi, op in enumerate(ops):
        for fi, fam in enumerate(fams):
            sel = [r["improved"] for r in recs
                   if r["family"] == fam and r["move"]["op"] == op]
            M[oi, fi] = np.mean(sel) if sel else np.nan
    fig, ax = plt.subplots(figsize=(DOUBLE, 1.9))
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(["/".join(ab.get(w, w) for w in f.split(" in "))
                        for f in fams], fontsize=6, rotation=45, ha="right")
    ax.set_yticks(range(len(ops)))
    ax.set_yticklabels(ops, fontsize=7)
    for oi in range(len(ops)):
        for fi in range(len(fams)):
            if np.isfinite(M[oi, fi]):
                ax.text(fi, oi, f"{100 * M[oi, fi]:.1f}", ha="center",
                        va="center", fontsize=5.5,
                        color="w" if M[oi, fi] > np.nanmax(M) * 0.6 else "k")
    fig.colorbar(im, ax=ax, label="improvement rate", fraction=0.03)
    save(fig, "fig_opsfamily",
         [[op, fam, f"{M[oi, fi]:.5f}"] for oi, op in enumerate(ops)
          for fi, fam in enumerate(fams)],
         ["op", "family", "rate"])


# ------------------------------------------------------------------ slack ---
def fig_slack(args):
    """How close each family's best packings get to the area lower bound."""
    AS = {"tan": 0.5, "square": 1.0, "circle": math.pi, "domino": 2.0,
          "L": 3.0, "ngon:3": math.sqrt(3) / 4}
    AC = {"tan": 0.5, "square": 1.0, "circle": math.pi, "domino": 2.0,
          "L": 3.0}
    per_fam, rows = {}, []
    for f in glob.glob("runs/*.json"):
        try:
            d = json.load(open(f))
            shape, cont = d["family"].split(" in ")
            lb = math.sqrt(d["n"] * AS[shape] / AC[cont])
            slack = d["s"] / lb - 1
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
        per_fam.setdefault(d["family"], []).append(slack)
        rows.append([d["family"], d["n"], f"{slack:.5f}"])
    if not per_fam:
        print("[figures] slack: no runs; skipping")
        return
    fams = sorted(per_fam, key=lambda k: np.median(per_fam[k]))
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.1))
    bp = ax.boxplot([per_fam[f] for f in fams],
                    tick_labels=[f.replace(" in ", "\nin ") for f in fams],
                    patch_artist=True, medianprops=dict(color="k"))
    for patch in bp["boxes"]:
        patch.set_facecolor(C[0])
        patch.set_alpha(0.55)
    ax.set_ylabel("slack over area bound\n$s/s_{\\mathrm{lower}} - 1$")
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(True, axis="y")
    save(fig, "fig_slack", rows, ["family", "n", "slack"])


# ------------------------------------------------------------------ audit ---
def fig_audit(args):
    """Feasibility discipline: audit violations of every reported packing."""
    xs, ys, rows = [], [], []
    for f in glob.glob("runs/*.json"):
        try:
            d = json.load(open(f))
            xs.append(max(d["audit_max_overlap"], 1e-16))
            ys.append(max(d["audit_max_outside"], 1e-16))
            rows.append([os.path.basename(f), d["audit_max_overlap"],
                         d["audit_max_outside"]])
        except (KeyError, json.JSONDecodeError):
            continue
    if not xs:
        print("[figures] audit: no runs; skipping")
        return
    fig, ax = plt.subplots(figsize=(SINGLE, 2.1))
    ax.scatter(xs, ys, s=8, color=C[0], alpha=0.6)
    ax.axvline(1e-9, color=C[1], ls="--", lw=0.7)
    ax.axhline(1e-9, color=C[1], ls="--", lw=0.7,
               label="verifier tolerance ($10^{-9}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("max pairwise overlap")
    ax.set_ylabel("max containment violation")
    ax.grid(True, which="both")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    save(fig, "fig_audit", rows, ["run", "max_overlap", "max_outside"])


# --------------------------------------------------------------- contacts ---
def fig_contacts(args):
    """Contact-graph size vs instance size in the logged training states."""
    per_n = {}
    for path in glob.glob(args.logs):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    per_n.setdefault(r["n"], []).append(
                        len(r["state"].get("contacts", [])))
                except (json.JSONDecodeError, KeyError):
                    continue
    if not per_n:
        print("[figures] contacts: no logs; skipping")
        return
    ns = sorted(per_n)
    mean = [np.mean(per_n[n]) for n in ns]
    sd = [np.std(per_n[n]) for n in ns]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.0))
    ax.errorbar(ns, mean, yerr=sd, fmt="o-", color=C[0], ms=3, lw=0.8,
                elinewidth=0.5, capsize=1.5)
    ax.plot(ns, ns, ls=":", color="#999999", lw=0.8, label="$n$ (reference)")
    ax.set_xlabel("$n$")
    ax.set_ylabel("contacts per state\n(mean $\\pm$ s.d.)")
    ax.grid(True)
    ax.legend(frameon=False, fontsize=6)
    save(fig, "fig_contacts",
         [[n, f"{m:.2f}", f"{s:.2f}"] for n, m, s in zip(ns, mean, sd)],
         ["n", "mean_contacts", "sd"])


# ------------------------------------------------------------------- main ---
ALL = ["gallery", "landscape", "restarts", "convergence", "moves",
       "dataset", "gradcheck", "ablation", "training", "records", "summary",
       "seeds", "beats", "density", "opsfamily", "slack", "audit", "contacts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, choices=ALL)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--inst", nargs=3, default=["tan", "4", "tan"],
                    metavar=("SHAPE", "N", "CONTAINER"))
    ap.add_argument("--restart-seeds", type=int, default=None)
    ap.add_argument("--trace-seeds", type=int, default=None)
    ap.add_argument("--grad-states", type=int, default=None)
    ap.add_argument("--logs", default="logs/moves.jsonl*")
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--sft-log", default="sft.log")
    ap.add_argument("--record-ns", type=int, nargs="*",
                    default=[21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
    ap.add_argument("--gallery-jsons", nargs="*",
                    default=["3_tan_in_tan.json", "4_tan_in_L.json",
                             "2_circle_in_tan.json", "_val_5_square_square.json"])
    args = ap.parse_args()
    args.restart_seeds = args.restart_seeds or (16 if args.quick else 200)
    args.trace_seeds = args.trace_seeds or (3 if args.quick else 6)
    args.grad_states = args.grad_states or (3 if args.quick else 10)

    todo = args.only if args.only else ALL
    for name in todo:
        globals()[f"fig_{name}"](args)
    print(f"[figures] done -> {FIGDIR}/ (vector .eps/.pdf + 600dpi .png; "
          f"underlying numbers in {DATADIR}/)")


if __name__ == "__main__":
    main()
