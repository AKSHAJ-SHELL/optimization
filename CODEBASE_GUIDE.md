# The `friedman` codebase — complete guide

*An exhaustive, function-by-function walkthrough of the LLM-guided packing
record hunter, plus the math and papers behind it. Written so that if you open
any file and land on any function, you already know what it does, why it
exists, and how it connects to everything else.*

Companion docs already in the repo: `EXPLAINER.md` (the narrative story),
`DESIGN.md` (architecture), `RUNBOOK.md` (operations), `EFFICIENCY.md`
(memory), `TARGETS.md` (the record hit-list), and `paper/paper.pdf` (the
formal writeup). This guide unifies all of them at the level of individual
functions.

---

## 0. The one-paragraph mental model

You are trying to fit `n` copies of a shape (a triangle, square, circle, L,
etc.) into the smallest possible container of a given form. That is two
problems braided together: a **continuous** problem (given a rough layout,
nudge every coordinate to squeeze out slack — calculus is great at this) and a
**combinatorial** problem (which layout? which piece in which corner, which
pair is flipped — calculus cannot hop between these). This repo builds a
**classical optimizer** (`packer.py`) that owns the continuous half perfectly
(a physics-style penalty function, exact autodiff gradients, an L-BFGS
minimizer, a shrink loop, an elite pool, and a paranoid endgame + audit), and
then bolts on a **learned policy** (a fine-tuned LLM) that owns the
combinatorial half — it proposes *which discrete move to try*, and the
classical solver resolves all the precise numbers. The LLM never emits a
float. The optimizer generates its own training data about itself, the LLM is
fine-tuned on it (SFT then DPO via QLoRA), and the whole thing is validated
against known-proven optima and an independent 50-digit verifier. Real
outcome: ten packing records credited to Akshaj Shandilya on Erich Friedman's
Packing Center (July 2026).

---

## 1. Repository map

### Core engine and action space
| File | Role |
|---|---|
| `packer.py` | The optimizer. Geometry catalog, penalty function, JAX gradients, shrink loop, elite pool, endgame + audit, logging, the `random`/`llm` proposers, and the CLI. **The heart of everything.** |
| `moves.py` | The discrete move schema + canonical-state builder. The shared "action space" used by the random proposer, the LLM proposer, and the dataset builder. |
| `llm_proposer.py` | MLX inference wrapper: loads the fine-tuned model and samples moves from a canonical-state prompt. |

### The ML pipeline (logs → data → tuned model → experiment)
| File | Role |
|---|---|
| `make_dataset.py` | Turns `--log` output into SFT (`train/valid.jsonl`) and DPO (`dpo_train/valid.jsonl`) with held-out splits. |
| `train_sft.sh` | Stage-1 QLoRA supervised fine-tune (MLX, Apple Silicon). |
| `train_dpo.py` | Stage-2 Direct Preference Optimization (TRL/PEFT on MPS). |
| `quickstart_tune.sh` | 1–2 hour end-to-end taste of the whole pipeline with a small 1.5B model. |
| `ablation.py` | The paper's central experiment: LLM proposer vs random proposer, same everything → `results.csv`. |

### Correctness, operations, and output
| File | Role |
|---|---|
| `validate.py` | Sanity gate: recover known/proven optima. Run first; must PASS. |
| `verify.py` | Independent 50-digit re-verifier (mpmath, zero shared code with the engine). Required before submitting a record. |
| `sweep.py` | Overnight driver: run whole record tiers / data plans in one command, with resume + `summary.csv` + beat flagging. |
| `submit.py` | Package a *verified* packing into a registry submission bundle. |
| `organize.py` | Collect scattered result JSON/PNG pairs into `results/<family>/`, best-first, with an `INDEX.md`. |
| `render_web.py` | Redraw a packing JSON in the registry's page style (lavender fill). |
| `figures.py` | Generates every figure + data CSV for the Springer manuscript. |

### Docs and artifacts
`README.md`, `DESIGN.md`, `EXPLAINER.md`, `RUNBOOK.md`, `EFFICIENCY.md`,
`TARGETS.md`; `paper/` (LaTeX manuscript + figures + bibliography);
`data/` (the built SFT/DPO JSONL); `runs/`, `logs/`, `results/`,
`submissions/`; `adapters_sft/`, `adapters_quick/`, `model_quick/` (trained
adapters/models). The many top-level `*.json`/`*.png` files are individual
packing results and ablation outputs (`_abl_*`), plus validation outputs
(`_val_*`).

---

## 2. Math foundations (read this once; every formula in the code maps here)

### 2.1 The optimization problem
Each shape `i` has a pose `(xᵢ, yᵢ, θᵢ)` (position + rotation); the container
has size `S`. For fixed `n` you solve:

```
minimize   S
subject to   no two shapes overlap
             every shape stays inside the container C(S)
```

**Area lower bound.** Shapes' total area can't exceed the container's area, so
`S ≥ √(n · shape_area / container_area_at_S=1)`. In code this is
`Problem.S_lower` (`packer.py`), e.g. `√n` for tans-in-tans. You can never beat
it; good packings hug it. The gap `S − S_lower` is "wasted space" and is what
every figure and trace plots.

### 2.2 Overlap as a smooth number — the Separating Axis Theorem (SAT)
Two convex shapes are disjoint **iff** there is some axis on which their
projected shadows don't overlap ("shine a flashlight; if from some angle the
shadows separate, the shapes are apart"). For polygons you only need the
directions perpendicular to each edge. Projecting both shapes onto axis `a`
gives intervals; the overlap on that axis is

```
ov(a) = min(max₁, max₂) − max(min₁, min₂)
```

and the penetration depth is `ω = minₐ ov(a)`. `ω > 0` ⇒ real overlap; `ω ≤ 0`
⇒ a separating axis exists ⇒ disjoint. This is exactly the `pair_pp` function
in `packer.py` and `sat_overlap` in `verify.py`. Circles skip the flashlight:
two circles overlap iff center-distance `<` sum of radii (`pair_cc`), and a
circle-polygon pair uses the signed distance from the center to the polygon
boundary (`pair_cp`).

### 2.3 The penalty function Φ ("springs everywhere")
Sum every violation, squared:

```
Φ(x, S) = Σ max(0, ω_pair)²  +  Σ max(0, containment_violation)²
```

Any overlapping pair gets a spring pushing them apart; any shape poking out of
the container gets a spring pushing it in. `Φ = 0` exactly when the packing is
legal. Squaring makes it smooth (C¹) at zero, which the gradient minimizer
needs. This is `penalty()` in `packer.py`.

### 2.4 Exact gradients via automatic differentiation (JAX)
To minimize Φ you need `∇Φ` — which way to nudge each of the `3n` coordinates.
Two ways: **finite differences** (wiggle each coordinate, re-measure — costs
~`3n`–`6n` evaluations per step; what the prior open-source solver did), or
**automatic differentiation** (JAX applies the chain rule through Φ's exact
formula and returns the exact gradient for ~the cost of one evaluation). The
whole engine is built so Φ is one differentiable JAX expression; `jax.value_and_grad`
gives value + gradient together. This is *the* biggest speed upgrade in the
project. `fig_gradcheck` confirms autodiff agrees with central differences to
~10⁻⁹.

### 2.5 L-BFGS-B — the inner minimizer
A quasi-Newton method: a ball rolling downhill that *remembers* the recent
curvature of the terrain (a limited-memory approximation of the Hessian), so it
takes smart curvature-aware steps instead of timid straight-downhill ones. It
crushes Φ from a messy start to ~10⁻¹⁰ in a few hundred steps. Called via
`scipy.optimize.minimize(method="L-BFGS-B", jac=True)` in `local_min()`.

### 2.6 Basin hopping + the shrink loop — the outer game
Start with a comfortably large container, find a legal packing (`Φ ≈ 0`),
shrink `S` by ~1%, re-relax, repeat. When shrinking fails, give the config a
random *kick* (`jitter`) and re-minimize — that's **basin hopping** (kick the
ball out of its valley, hope it rolls into a deeper one; from computational
chemistry, Wales & Doye 1997). When even kicks fail, the restart is done.
Hundreds of restarts run in parallel. This is `solve_attempt()`.

### 2.7 The elite pool
Independent restarts throw away everything they learn. Instead, keep the best
`k` configs ("elites") and spend compute perturbing *them* with a discrete move
(rotate a piece, swap two, relocate one, jiggle a few), then re-shrink each
variant. "Repack your best suitcase with one smart change" instead of packing
from scratch. **That "one smart change" is exactly the slot the LLM fills.**

### 2.8 The endgame and the audit
Search-phase tolerance allows ~10⁻⁵ overlaps (invisible, but fatal to a record
claim). The endgame re-polishes at tolerance 10⁻¹⁸, shrinking with steps that
decay geometrically from 10⁻³ to 10⁻¹¹, then an audit measures the true worst
violation and inflates `S` just enough to clear it, so reported violations are
< 10⁻¹¹. Then a *separate program* (`verify.py`), sharing zero code, re-checks
everything in 50-digit arithmetic. (This paranoia caught a real sign bug in the
verifier during development.)

### 2.9b How a pose becomes world coordinates
Every shape is stored once in its own "shape frame" (centered on its centroid).
To place it, you rotate by `θ` and translate by `(x, y)`. For a vertex `v`:
`world = T + R(θ)·v`, where `R(θ) = [[cos, −sin],[sin, cos]]`. The container is
treated as a virtual "shape" with scale `S` and zero rotation. This single
`world()` transform (in `packer.py`) is what makes Φ differentiable in the
poses *and* in `S` at once.

### 2.9 The learning math: SFT, DPO, QLoRA
- **SFT (supervised fine-tuning):** standard next-token cross-entropy — given
  the state prompt, make the good move's JSON more likely. "Here's the
  situation, here's what worked, imitate it."
- **DPO (Direct Preference Optimization):** for a chosen move `y⁺` and rejected
  move `y⁻` from the *same* state `x`, the loss is
  ```
  L = −log σ( β·[ log π(y⁺|x)/π₀(y⁺|x) − log π(y⁻|x)/π₀(y⁻|x) ] )
  ```
  where `π` is the model being trained, `π₀` the frozen reference, `σ` the
  logistic function, `β` a temperature. In words: raise the winning move's
  probability relative to the losing one, *without* needing a numeric reward.
  A pair "this worked, that didn't from the identical state" is a strictly
  stronger signal than a pile of unlabeled examples.
- **LoRA / QLoRA:** freeze the 7B model, learn tiny low-rank adapter matrices
  (~0.076% of params); QLoRA additionally stores the frozen weights in 4-bit
  (~4GB vs ~15GB). Final recipe fit in ~6GB.

---

## 3. `packer.py` — the engine (function by function)

This is the largest and most important file. Top-level constant `BIG = 1e18`
is a sentinel for "infinity" used to mask padded array slots so they never win
a `min`/`max`.

### 3.1 Geometry catalog

**`_centered(verts)`** — Takes a polygon's vertices and returns them shifted so
the polygon's *area-weighted centroid* sits at the origin, plus the centroid it
removed. Uses the standard shoelace centroid formula: signed area
`a = ½Σ(xᵢ·yᵢ₊₁ − xᵢ₊₁·yᵢ)`, then `cx, cy` from the polygon-centroid formula.
Centering every shape on its centroid is what makes rotations behave sanely
(a shape rotates about its own middle, not some arbitrary corner). *(Note: in
the current catalog most shapes are hand-centered inline, so `_centered` is a
utility/helper kept for completeness.)*

**`_poly_area(verts)`** — Shoelace area of a polygon,
`½|Σ(xᵢ·yᵢ₊₁) − Σ(xᵢ₊₁·yᵢ)|`. A small helper for area bookkeeping.

**`shape_def(name)`** — The **shape catalog**. Returns a dict
`{parts, sym, area}` describing one copy of a shape *in its own frame*, centered
on its centroid. A "part" is either a polygon (`verts`) or a circle
(`circle=True, center, radius`). Non-convex shapes are given as *multiple
convex parts* rigidly welded to one pose.
- `square`: unit square, `sym=4` (4-fold rotational symmetry), area 1.
- `domino`: 1×2 rectangle, `sym=2`, area 2.
- `tan`: right isosceles triangle with legs = 1, shifted by `−(⅓,⅓)` to center
  on its centroid, `sym=1`, area ½.
- `circle`: radius 1, `sym=0` (continuous symmetry), area π.
- `L`: a 2×2 square minus a 1×1 corner, decomposed into **two rectangles**
  `r1, r2`, centered on the L's centroid `(5/6, 5/6)`, area 3.
- `ngon:k`: regular k-gon with side 1. Circumradius `rc = 1/(2·sin(π/k))`; the
  angle offset `−π/2 + π/k` orients a flat side at the bottom; `sym=k`,
  area `k/(4·tan(π/k))`.
The `sym` field records each shape's rotational symmetry (used conceptually to
understand redundant orientations; the engine mostly relies on the optimizer to
resolve orientation).

**`container_def(name)`** — The **container catalog**, everything given at size
`S=1` and scaled about the origin. Returns `{kind, verts?, obstacles, area1,
...}`.
- Polygon containers (`square`, `tan`, `domino`, regular `ngon:k`) return their
  boundary polygon and an empty obstacle list.
- `circle`: `kind="circle"`, no polygon.
- `L`: modeled **exactly** as a bounding 2×2 square **plus a forbidden corner
  rectangle** (`obstacles=[ob]`) — the notch. It also carries an `outline` (the
  true 6-vertex L outline) purely for drawing. This "bounding box + obstacle"
  trick is how a non-convex container is handled with only convex machinery: a
  shape is inside the L iff it's inside the square **and** outside the notch.
`area1` is the container's area at `S=1`, used for the area lower bound.

**`_edge_normals(verts)`** — Outward unit normals of a CCW convex polygon, one
per edge. For edge vector `d = (dx, dy)`, the outward normal is
`(dy, −dx)` normalized. These normals *are* the SAT flashlight axes and the
half-plane directions for containment. Used everywhere.

### 3.2 `class Problem` — one packing instance, compiled

`Problem(shape_name, n, container_name)` builds everything needed to evaluate
and differentiate a specific instance (e.g. "10 tans in a tan"), then JIT-compiles
the JAX functions. Building it is moderately expensive (JIT compile), so
instances are cached (see `get_problem`).

**`Problem.__init__`** — The heavy lifting of "flatten geometry into padded
arrays JAX can vectorize over." Step by step:
1. Look up `shape_def` and `container_def`.
2. Build a flat list of **parts**: `n` copies of the shape's parts, each tagged
   with its shape index `i`; then append the container's obstacles as a
   *virtual shape* with index `n`.
3. Pad every part to `Vmax` vertices and pack into fixed-shape arrays:
   `pv` (vertices), `vmask` (which slots are real), `pnorm` (edge normals),
   `nmask`, `ea`/`eb`/`emask` (edge endpoint indices for circle-polygon distance),
   `pcirc` (is this part a circle), `prad` (radius), `pshape` (which shape each
   part belongs to). Padding + masks is the trick that lets one JAX kernel handle
   shapes of different vertex counts without Python loops.
4. Build **pair lists** by type so each collision kind is a separate vectorized
   kernel: `pp` (polygon-polygon), `cp` (circle-polygon), `cc` (circle-circle).
   Parts of the *same* shape never pair (a shape can't overlap itself); two
   obstacle parts never pair with each other.
5. Store both JAX arrays (`self.arr`) and NumPy copies (`self._np`, for the
   plotting/`gaps` path).
6. For polygon containers, precompute the container edge normals `contU` and
   offsets `conto1` (so containment is a batched dot product).
7. Compute `S_lower` (area bound) and call `_build_fns()`.

**`Problem._build_fns()`** — Defines the differentiable core as closures over
the arrays, then JIT-compiles them. The inner functions:

- **`world(params, S)`** — The pose→world transform (see §2.9b). Reshapes the
  `3n` params into `(n, 3)` poses, appends a zero pose for the container/obstacle
  shape, builds each shape's rotation matrix `R`, and computes world-frame
  vertices `W`, world-frame normals `Nw`, and world-frame radii `rw`. The
  container "shape" gets scale `S`; real shapes get scale 1. This is the one
  place `S` enters the geometry, which is why Φ is differentiable in `S` too.
- **`pair_pp(W, Nw)`** — Vectorized SAT for all polygon-polygon pairs. For each
  pair it gathers the union of both parts' edge normals as candidate axes,
  projects both parts onto every axis (masking padded vertices with ±BIG so they
  never affect min/max), computes `ov = min(max₁,max₂) − max(min₁,min₂)` per
  axis, and returns `min over axes` = signed penetration depth (>0 overlap,
  <0 gap). This is §2.2 in code.
- **`pair_cp(W, Nw, rw)`** — Circle-polygon signed overlap. Computes whether the
  circle center is inside the polygon (via the max signed half-plane distance),
  finds the true distance from the center to the nearest polygon edge (project
  center onto each edge segment, clamp `t∈[0,1]`, take min distance), assembles a
  signed center-to-boundary distance `sd` (negative inside), and returns
  `r − sd` (>0 means the circle penetrates).
- **`pair_cc(W, rw)`** — Circle-circle: `r_i + r_j − distance` (>0 overlap).
- **`containment(W, rw, S)`** — How far each real part pokes *outside* the
  container. Circle container: `‖vertex‖ (+ radius) − S`. Polygon container:
  half-plane violations `vertex·Uₖ − S·o1ₖ` for every container edge `k` (plus
  radius for circles). Returns a flat vector (every part×constraint, for the
  penalty) and a per-part max (for the `gaps`/contact graph). *Note the notch/
  obstacle is not enforced here — it's handled as an ordinary pair via the
  virtual obstacle-shape in `pp`/`cp`.*
- **`penalty(params, S)`** — Assembles Φ: `Σ max(0, ·)²` over `pair_pp`,
  `pair_cp`, `pair_cc`, and `containment`. This is the single scalar JAX
  differentiates. §2.3 in code.
- **`audit(params, S)`** — The **feasibility inspector**: returns the raw
  `max` pairwise overlap and `max` containment violation (not squared, not
  summed) — the true worst constraint breach, for deciding whether a packing is
  legal enough to report.
- **`gaps_raw(params, S)`** — Returns the signed gap arrays per pair and per
  part, unreduced, so the Python layer can build the contact graph.

Then it compiles: `self._vg = jax.jit(jax.value_and_grad(penalty))` (value +
gradient in one call), `self._audit`, `self._gaps_raw`, and keeps `world`.

**`Problem.fun(x, S)`** — The scipy-facing objective: calls `_vg`, returns
`(float value, numpy gradient)`. This is what `L-BFGS-B` calls each step.

**`Problem.audit(x, S)`** — Python wrapper returning `(max_overlap,
max_outside)` as floats.

**`Problem.gaps(x, S)`** — Builds **shape-level signed gaps**: a dict
`(si, sj) → max gap` for shape pairs, plus a per-shape wall gap (including the
L notch, since the obstacle is virtual shape `n`). `>0` = overlap/penetration,
`~0` = touching, `<0` = separated. This feeds the contact graph in
`canonical_state` — the structure the LLM reads.

**`Problem.container_bbox(S)`** — Axis-aligned bounding box of the container at
size `S` (used to place random init points and to interpret `relocate`
fractions).

**`Problem.point_ok(p, S, margin)`** — Is point `p` a legal *center* location
(inside the container, `margin` away from walls, and outside any obstacle)? Used
to seed initial configurations without immediately-illegal placements.

**`Problem.shape_radius()`** — The shape's circumradius (farthest vertex/circle
extent from its center). Used to size the placement margin so shapes don't start
half-outside.

### 3.3 The optimizer functions

**`init_config(prob, S, rng, grid=False)`** — Produce a random starting layout:
`3n` numbers `(x, y, θ)` per shape. Either scatters points on a jittered grid
(`grid=True`) or samples uniformly in the bbox, keeping only points that pass
`point_ok`, then assigns random angles. Half of restarts use the grid seeding
(chosen randomly in `solve_attempt`), which tends to find orderly packings; the
other half start fully random.

**`local_min(prob, x0, S, maxiter=400)`** — One L-BFGS-B minimization of Φ from
`x0` at fixed `S`. Returns `(final Φ, final x)`. Thin wrapper over
`scipy.optimize.minimize` with `jac=True` (uses the exact gradient) and a very
tight `tol=1e-16`.

**`jitter(x, S, rng, pos_sig, ang_sig)`** — The basin-hopping kick: add Gaussian
noise to positions (scaled by `pos_sig·S`) and angles (scaled by `ang_sig`).
Returns a perturbed copy.

**`shrink_mult(S, S_low, range0, final_step)`** — Compute the multiplicative
shrink factor `m = 1 − frac` for the current `S`. `frac` interpolates from ~1%
when far above the area bound down to `final_step` (e.g. 1e-5) as `S`
approaches `S_low` — big confident steps early, tiny careful steps near the
limit.

**`solve_attempt(prob, seed, opts, x_init=None, S_init=None)`** — **One full
restart** (or one elite variant re-shrink). The core loop:
1. Pick a starting `S` (random 1.35–2.35× the area bound, unless `S_init` given)
   and a starting layout (`init_config`, unless `x_init` given).
2. Repeat up to `max_steps`: minimize Φ at current `S`. If `Φ < ptol`, the
   packing is legal — record it as `best`. Otherwise try up to `bh_iters`
   jittered basin-hops to recover feasibility; if none works, stop.
3. On success, shrink (`S *= m`, scale positions by `m` too so the layout stays
   inside) and continue.
Returns `(S, x)` of the smallest legal packing this restart found, or `None`.

**`endgame(prob, S, x, opts)`** — The precision + safety finisher (§2.8):
1. Minimize hard (`maxiter=2000`) to get a strictly clean start; if not clean,
   nudge `S` up slightly and retry.
2. **Geometric-backoff shrink:** try shrinking by `delta`; if it stays feasible,
   accept; if not, try a few tiny jittered re-minimizations; if still stuck,
   cut `delta` by 4× and continue — from `delta=1e-3` down to `1e-11`.
3. **Feasibility audit + inflate:** up to 30 times, measure the worst violation
   `v`; if `v ≥ 1e-11`, inflate `S` by just over `2v/S` and re-minimize, until
   violations clear. Returns `(S, x, max_overlap, max_outside)`.

### 3.4 Worker plumbing (parallelism)

**`_PROBLEM_CACHE` / `get_problem(shape, n, container)`** — A plain-dict cache of
built `Problem` objects. The comment explains *why a dict and not
`functools.lru_cache`*: the lru_cache C wrapper isn't picklable, which breaks
joblib/loky process workers; a dict is picklable and lets each worker build the
(JIT-compiled) Problem once and reuse it.

**`worker_attempt(shape, n, container, seed, opts_t)`** — A picklable top-level
entry point for a single independent restart in a worker process. Rebuilds/looks
up the Problem and calls `solve_attempt`.

**`worker_elite(shape, n, container, seed, opts_t, x, S, move=None, log=None)`**
— **One elite variant**, and the hinge where the AI plugs in:
1. Build the `canonical_state` (if logging or an LLM move is involved).
2. If a `move` was given (from the LLM, in canonical indices), map it back to
   solver indices via `unrelabel`. Otherwise sample a **random** move
   (`moves.sample_move`) and relabel it to canonical indices for logging.
3. `apply_move` to get a perturbed layout, then `solve_attempt` from it (with
   `S_init = S·1.02`, a little slack to let the move settle).
4. If `--log`, append a JSONL record `(family, n, state, move, s_before,
   s_after, improved)` to `logs/moves.jsonl.<pid>` — **this is how the training
   data is generated as a side effect.** Note it strips `order` from the logged
   state (it's solver-internal) and writes per-PID files to avoid clashes.

Because random and LLM proposers go through the *identical* `worker_elite` path
and action space, `--proposer random` is an *exact* ablation baseline.

**`run_parallel(jobs, workers)`** — Fan a list of `(fn, args)` jobs across
processes with joblib (or run serially if joblib is missing or `workers==1`).

### 3.5 Output

**`save_outputs(prob, S, x, mo, mc, prefix, record=None)`** — Writes
`<prefix>.json` (family, n, `s`, audit values, record reference, and each
shape's `x/y/theta`) and, best-effort, `<prefix>.png` (matplotlib drawing of the
container outline + every placed shape). Plotting failures are swallowed so a
headless run never dies on a rendering issue.

### 3.6 CLI — `main()`

Parses all flags (`--shape`, `--n`, `--container`, `--attempts`, `--workers`,
`--elite-rounds/-k/-variants`, `--record`, `--log`, `--proposer`, `--model`,
`--llm-temp`, `--quick`, …), then:
1. Print the area lower bound.
2. Run `--attempts` independent restarts in parallel (`worker_attempt`); keep
   feasible results, sort by `s`, print the initial pool best.
3. **Elite rounds:** keep the top `elite_k` as the pool. Each round, for each
   elite, either the LLM proposes `elite_variants` moves (`--proposer llm`) or
   `None`×variants (random). Run all variants via `worker_elite`, merge into the
   pool, keep top-k. Print best each round.
4. Run `endgame` on the best config; print `s`, the audit line, and (if
   `--record`) the beat/miss margin.
5. `save_outputs`.

Key knobs to understand: `--attempts` is raw exploration budget (throughput is
the resource — go big); `--elite-rounds/-k/-variants` control exploitation of
the best configs (the slot where a learned policy acts); `--quick` swaps in
fast smoke-test settings.


## 4. `moves.py` — the action space + canonical state

This module *is* the shared vocabulary between the random proposer, the LLM
proposer, and the dataset builder. Constants:
`ROT_CHOICES = [45, -45, 90, -90, 180]`, `MAG_CHOICES = [0.05, 0.1, 0.2]`,
`OPS = ("rotate", "swap", "relocate", "jiggle")`. The four move types are the
entire discrete action space — coarse enough that an LLM can pick reliably,
expressive enough to change a packing's topology.

**`sample_move(rng, n)`** — The **random proposer / behavior policy**. Draws a
move type by probability (jiggle ~35% and always if `n<2`, rotate ~25%, swap
~15%, relocate ~25%) in **original solver indices**. Jiggle picks a small random
subset of shapes and a magnitude; rotate picks a shape and a snap angle; swap
picks two distinct shapes; relocate picks a shape and target bbox-fractions.
This is what the ablation's random arm uses — and what generated most of the
training logs.

**`apply_move(x, S, prob, move, rng)`** — Execute a move (in original indices)
on a config `x`:
- `rotate`: add `radians(deg)` to shape `i`'s angle.
- `swap`: exchange the full `(x, y, θ)` of shapes `a` and `b`.
- `relocate`: move shape `i` to `(fx, fy)` as fractions of the container bbox,
  with a fresh random angle.
- `jiggle`: add Gaussian noise (scaled by `mag·S` in position, `5·mag` in angle)
  to each listed shape.
All indices are taken `% n` defensively. Returns the perturbed copy. This is the
bridge from an abstract move to actual coordinates the solver then re-optimizes.

**`relabel(move, inv)`** — Rewrite a move's indices from **original → canonical**
via `inv[orig] = canon`. Used when logging a random move so the recorded state
and move share the canonical numbering.

**`unrelabel(move, order)`** — Rewrite a move's indices from **canonical →
original** via `order[canon] = orig`. Used when the LLM (which sees canonical
indices) proposes a move and the solver needs original indices to apply it.
`relabel` and `unrelabel` are inverses; keeping them separate makes the two
directions explicit and hard to confuse.

**`validate_move(move, n)`** — **Schema guard** for a possibly-LLM-emitted move.
Returns a cleaned move or `None`. Checks op type, coerces indices `% n`, verifies
`deg ∈ ROT_CHOICES`, that swap indices differ, that relocate fractions are in
`[0,1]` (rounded to 2 decimals), and snaps jiggle magnitude to the nearest legal
choice. Any malformed field → `None` (which triggers random fallback). This is
what keeps a hallucinating model from ever corrupting the search.

**`canonical_state(prob, x, S, tol=1e-5)`** — Builds the **symmetry-reduced,
low-precision view the policy sees**. Steps:
1. Sort shapes by a fixed geometric key `(distance-from-origin, angle)` — this
   kills the `n!` relabeling symmetry so two identical packings with shuffled
   labels look identical to the model. `order[canon]=orig`, `inv[orig]=canon`.
2. Emit each shape in canonical order with coordinates rounded to 3 decimals and
   angle in degrees (`deg % 360`).
3. Build the **contact graph** from `prob.gaps`: any shape pair whose signed gap
   `> −tol` is "touching" → a `[i, j]` contact; any shape touching a wall/notch →
   `[i, "wall"]`. Contacts are the load-bearing structure that topology moves
   manipulate — handing them over directly is worth more than a bigger model.
Returns `{family, n, s, lower_bound, shapes, contacts, order}`.

**`PROMPT_TEMPLATE` / `render_prompt(state)`** — Renders the canonical state into
the exact text prompt the model is trained and queried on: the instance
description, the shape list `i:(x, y, deg)`, the touching pairs, the allowed-move
JSON schema, and a trailing `JSON:` cue. (Appendix A of the paper shows a real
example.) Keeping training and inference prompts identical is why the model
behaves in deployment.

**`parse_move_json(text, n)`** — Extract the first well-formed JSON object from
raw LLM output (brace-matching, bounded to 500 chars so it can't scan forever),
then run it through `validate_move`. Returns the cleaned move or `None`. Robust
to the model adding prose around the JSON.

## 5. `llm_proposer.py` — the fine-tuned policy at inference

**`class LLMProposer`** — Wraps MLX inference so `packer.py` can ask for moves.

- **`__init__(model_path, temperature=0.8, max_tokens=80)`** — Loads the model
  via `mlx_lm`. Cleverly accepts **either** a full model dir/HF repo **or** an
  `mlx_lm` adapter directory (`adapters_sft/`): if it finds `adapter_config.json`
  it resolves the base model from it and loads base + adapter together (no fuse
  step needed with a quantized base). Sets up a temperature sampler, initializes
  `n_calls`/`n_fallback` counters, and registers `_report` to run at exit.
- **`_report()`** — At process exit, prints the invalid-generation rate
  (`n_fallback / n_calls`). This is a **health metric**: a well-trained policy
  should be <10% (the paper's deployed run was 0.0%, 144/144 valid).
- **`propose(state, k)`** — Renders the prompt (applying the tokenizer chat
  template if present), samples `k` completions, parses each into a validated
  move, counts fallbacks, and returns a length-`k` list where invalid slots are
  `None` (later replaced by a random move in `worker_elite`). So a degenerate
  model gracefully reduces to the random baseline instead of stalling.

## 6. `make_dataset.py` — logs → SFT + DPO data

Turns the per-worker `moves.jsonl.*` logs into training files.

**`load_records(pattern)`** — Glob + read all matching JSONL logs into a list of
dicts, skipping any unparseable lines.

**`is_holdout(rec, holdout_n, holdout_family)`** — A record goes to the
**validation** split if its `n` is in the held-out set **or** its family matches a
held-out substring (e.g. `"in L"`). This is how generalization across *size* and
across *shape family* is measured — you test on `n` values and a family the
model never trained on.

**`main()`** — The build:
1. Load records; for each, render the prompt from its state and serialize its
   move to compact JSON.
2. **SFT:** every record whose `improved` is true becomes a
   `{prompt, completion}` example (imitate successful moves). Grouped by prompt,
   good and bad moves are tracked per state.
3. **DPO:** for each state that saw *both* a good and a bad move, form
   `(chosen=good, rejected=bad)` pairs — up to `--max-pairs-per-state` (default
   4), shuffled. The pairing from the *same* state is the supervision signal.
4. Write `train/valid.jsonl` (SFT) and `dpo_train/valid.jsonl` (DPO); print
   counts and the improved-rate. (The paper's corpus: 42,304 records, 1.4%
   improving → 540 SFT examples, 546 DPO pairs.)

## 7. `train_sft.sh` — Stage 1 (QLoRA SFT via MLX)

A shell script running `mlx_lm.lora` on `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`.
Key settings encode the hard-won memory lessons (see `EFFICIENCY.md`):
`--fine-tune-type lora`, `--num-layers 8`, `--batch-size 1`, `--grad-checkpoint`,
`--iters 4000`, `--learning-rate 1e-4`, `--max-seq-length 1024`. The 4-bit base
(~4GB) + batch 1 + gradient checkpointing + short sequences keep peak memory
~8–12GB on a 64GB Mac. The header comment warns *never* to raise batch/seq: the
old bf16 7B at batch 4 × seq 2048 peaked >60GB and froze macOS. It also explains
that with a quantized base you point the proposer directly at the adapter dir (no
fuse), or fuse with `mlx_lm.fuse` if your version needs it.

## 8. `train_dpo.py` — Stage 2 (DPO via TRL/PEFT)

Runs DPO on the SFT model using the `(chosen, rejected)` pairs, on Apple Silicon
via PyTorch **MPS** with **bf16 LoRA** (bitsandbytes 4-bit is CUDA-only). `main()`:
loads the base model + tokenizer, loads the DPO JSONL as a HF dataset, builds a
`LoraConfig` (r=32, α=64, dropout 0.05, targeting all attention + MLP projection
modules), sets a `DPOConfig` (`beta=0.1`, `lr=5e-6`, grad-accum 8, eval every 200
steps), and runs `DPOTrainer`. Then convert to MLX for the proposer with
`mlx_lm.convert`. The docstring is candid: this is the **version-sensitive** step
(TRL's API churns) and may need arg tweaks; SFT-only is a valid configuration if
DPO fights you (see RUNBOOK Stage 4). `beta` is the DPO temperature from §2.9.

## 9. `quickstart_tune.sh` — the 1–2 hour end-to-end taste

Uses the small 1.5B coder for speed. Five steps: (1) generate ~30 min of training
data with `sweep.py --plan data`, (2) `make_dataset.py`, (3) a short SFT
(`mlx_lm.lora`, a few hundred iters), (4) fuse for fast inference
(`mlx_lm.fuse`), (5) run the tuned policy on a real target and eyeball the
`[llm_proposer]` invalid-rate. Includes a small inline Python step that splits 5%
off train into valid if the holdout came up empty (so `mlx_lm.lora` has a
non-empty validation set). It's the "prove the whole pipeline works before
committing a night to the 7B" path.

## 10. `ablation.py` — the paper's central experiment

**`run(shape, n, container, proposer, model, seed, attempts, rounds)`** — Shell
out to `packer.py` for one `(instance, proposer, seed)` cell, adding `--model`
for the LLM arm, and return `(s, wall_clock_seconds)` from the output JSON.

**`main()`** — For each eval instance (`shape:n:container:record`, default is 6
tan/L instances) × proposer ∈ {random, llm} × seed ∈ [0, seeds): run and collect
`{shape, n, container, record, proposer, seed, s, gap=s−record, seconds}`. Write
`results.csv`. This is the exact, fair race: identical loop, action space, budget,
and seeds; the only difference is who proposes moves. `figures.py`'s
`fig_ablation`/`fig_seeds` consume this CSV. (Result: ties on saturated
instances, LLM better mean gap on the hardest, two best packings LLM-only, at
~6× wall-clock.)


## 11. `validate.py` — the correctness gate (run this first)

Recovers known/proven optima at tiny budgets; every check must PASS before you
trust any record claim.

**`CHECKS`** — A list of `(shape, n, container, known_s, tol, note)`: 2 squares
in square (=2), 2 circles in square (=2+√2), 2 & 3 circles in tan (proven,
Xu 1996), 5 squares in square (proven, Friedman), 3 tans in tan (record
1.961+). These span every collision type (poly-poly, circle-poly, circle-circle,
non-convex container).

**`run(shape, n, container, attempts)`** — Shell out to `packer.py --quick
--elite-rounds 1`, read the output JSON, return `(s, worst_audit_violation)`.

**`main()`** — For each check, run (8–16 attempts), PASS if `s ≤ known + tol`
**and** the audit violation `< 1e-9`. Print a PASS/FAIL line each and exit
nonzero if any fail — "do not hunt records until fixed."

## 12. `verify.py` — the independent 50-digit re-verifier

**Deliberately imports no engine/JAX code** — it re-implements all geometry from
scratch in `mpmath` at `--dps` digits (default 50). If the engine had a bug, it
couldn't certify itself. Required on every candidate before submission.

**`_reg_ngon(k)`** — Regular k-gon vertices in mpmath (same convention as the
engine's `shape_def`, re-derived independently).

**`shape_parts(name)`** — mpmath shape catalog (parts in shape frame): square,
domino, tan, circle, L (two rectangles), ngon. Mirrors `packer.shape_def` but
written separately.

**`container_geom(name, S)`** — mpmath container catalog: returns
`(kind, boundary_polygon, notches)`. The L is a bounding square **plus a notch
rectangle** — the same modeling as the engine, re-implemented.

**`transform(part, x, y, th)`** — Rotate+translate a part's vertices (the
`T + R·v` transform, mpmath).

**`edge_normals(poly)`** — Outward unit normals (mpmath), CCW convention.

**`sat_overlap(p1, p2)`** — Exact SAT signed penetration between two convex
polygons — the independent twin of `pair_pp`. Early-exits as soon as a separating
axis is found.

**`point_poly_signed(p, poly)`** — Signed distance from a point to a convex
polygon boundary (negative inside): min distance over edges, with an inside test
via the CCW cross-product sign on every edge.

**`circle_poly_overlap(c, r, poly)`** — `r − point_poly_signed(c, poly)`; the
circle-polygon twin.

**`verify(data, dps, tol)`** — The full re-check: place every shape (transform
its parts), compute the **max pairwise overlap** across all part-pairs (circle/
circle, circle/poly, poly/poly), and the **max containment violation** (vertices
outside the boundary; circle centers + radius vs boundary; and everything vs the
L notch). Returns `(max_overlap, max_outside)`.

**`main()`** — Load the JSON, run `verify`, print claimed `s`, both worst
violations, and a **PASS/FAIL** verdict against `--tol` (default 1e-9). On PASS it
prints a **safe value to submit**: `s` rounded up past the worst-case violation
(`pad = worst·10 + 1e-12`) so no digit you quote can be wrong. Exits nonzero on
FAIL — which is what `submit.py` checks.

## 13. `sweep.py` — the overnight driver

Runs `packer.py` across a whole *plan* of `(shape, n, container)` combos in one
command, logging training data and hunting records in one pass.

**Record tables** (`TANINTAN`, `TANINL`, `LINTAN`, `TRIINTAN`) — hard-coded
current registry page values per `n`, used to wire `--record` in and flag beats.
`TRIINTAN` is the triangles-in-tans page (scraped 2026-07-20; page ends at 23,
so `n≥24` are unclaimed new-entry territory).

**`plan_items(name)`** — Expand a plan name into a list of
`(shape, n, container, record)`:
- `tier1`: tans in tans, n=3..27 (soft 2005–2009 records).
- `tier2`: tans in L's, n=3..18 (soft 2012 records).
- `tier3`: L's in tans, n=2..22.
- `tri`: equilateral triangles (`ngon:3`) in tans, n=4..30 (the 2026-rush
  family; n=27..30 requested by the maintainer).
- `data`: a diverse small-n mix across 9 families (training-data generation).
- `smoke`: a two-item sanity plan.

**`main()`** — For each combo (respecting an optional `--minutes` budget):
1. **Resume:** skip combos that already have a `runs/<name>.json` unless `--redo`.
2. **Clobber protection:** if a result exists, move it to `_prev` first; after the
   run, keep whichever `s` is smaller (so a worse re-run never destroys a better
   past result); restore `_prev` on failure.
3. Build and run the `packer.py` command (passing `--record`, `--proposer llm
   --model` if requested, `--log`, `--quick`).
4. If `s < record`, flag a **possible record** loudly and append to `beats`.
5. Append a row to `runs/summary.csv` (`name, shape, n, container, s, record,
   beat, seconds`).
At the end it lists all possible beats with a reminder to run `verify.py` before
submitting. This is the workhorse you actually leave running overnight.

## 14. `submit.py` — package a verified record for the registry

**`main()`** — Turns one verified packing JSON into a submission bundle:
1. **Verification gate:** shell out to `verify.py`; if it returns nonzero,
   refuse to package anything ("Fix first"). Capture its "safe value" line.
2. Copy the `.png` into `submissions/<name>/packing.png`.
3. Write `coordinates.txt`: shape/container size conventions (human-readable),
   `s` at full precision, the audit values (noting independent 50-digit
   re-verification), and each shape's `x y theta` at full `repr` precision.
4. Write `submission.txt`: a ready-to-send email body to Erich Friedman with the
   value, the safe-quote line, and a description of the method.
5. Copy the JSON to `verified.json`. Warn if `s` doesn't actually beat the stated
   record. You then email the bundle to the contact on the registry site.

## 15. `organize.py` — tidy results into a browsable tree

**`RECORDS`** — page values for the beat column (tan/L families).

**`main()`** — Scan directories (`. runs examples submissions` by default) for
every `*.json` packing result, dedupe by `(family, n, rounded s)`, group by
`(family, n)`, sort best-first, and **copy** (originals untouched) into
`results/<family>/`: the best as `n{NN}_s{...}` and the rest into an
`alternates/` subfolder. Write `results/INDEX.md` — a sortable table of best `s`
per instance vs page value, flagging possible beats. Re-runnable anytime; it
rebuilds `results/` from whatever exists.

## 16. `render_web.py` — registry-style pictures

**`render(json_file, px)`** — Redraw a packing in the Packing Center's page style:
lavender fill (`#d9daeb`), thin dark edges, no axes/title/frame, white
background, fixed pixel size — writes `<name>_web.png`. It imports `packer` to
reuse the container/shape catalogs so orientation matches the engine exactly.

**`main()`** — Glob the given JSON paths and render each at `--px` (default 420).

## 17. `figures.py` — the manuscript's figures + data

Every figure is saved as `.eps` + `.pdf` (vector, for Springer submission) and
`.png` (600 dpi preview), sized to Springer column widths, and each writes its
underlying numbers to `figures/data/<name>.csv` for the data-availability
declaration.

**`save(fig, name, data_rows, data_header)`** — The common writer: dumps the
three image formats and the optional CSV.

The figure functions (each named `fig_<name>`, dispatched by `main()`):
- **`fig_gallery`** — Vector redraws of representative best packings across all
  collision types.
- **`fig_landscape`** — Record *year* by family/n from the registry scrape (the
  `REG` table): shows the 2026 rush hit regular polygons while tan/L families sat
  untouched for 14–21 years. This is the "why these families" figure.
- **`fig_restarts`** — Histogram + empirical CDF of per-restart final `s` for one
  instance: quantifies how much work is basin-escape vs local refinement
  (motivates the elite pool). Runs `packer.solve_attempt` live.
- **`fig_convergence`** — Shrink-loop traces: gap to the area lower bound vs
  accepted shrink step, several restarts, on a log axis. Re-implements the
  shrink loop inline to record the trace.
- **`fig_moves`** — Improvement rate by move operator (from `--log` data): the
  operator imbalance is the signal a learned policy exploits.
- **`fig_dataset`** — Three-panel corpus characterization: records per family
  (improving vs not), distribution of relative improvement `Δs/s` (bimodal:
  basin-changing ~10⁻²–10⁻¹ vs refinement ~10⁻⁵), and improvement rate vs `n`.
- **`fig_gradcheck`** — Autodiff vs central-finite-difference gradient agreement
  across four instances (poly-poly, circle-poly, non-convex container): ~10⁻⁹,
  i.e. at the finite-difference truncation floor. This is the figure that proves
  the exact-gradient claim.
- **`fig_ablation`** — Mean gap to page value by instance and proposer, from
  `results.csv` (or a "PENDING" placeholder if the ablation hasn't been run).
- **`fig_training`** — Parses `sft.log`: train/val loss curves, marks the
  deployed iteration-500 checkpoint and the NaN-divergence region. The
  overfitting-and-recovery story, from real logs.
- **`fig_records`** — Page-style gallery of the triangles-in-tans registry
  results (`runs/<n>_ngon3_in_tan.json`).
- **`fig_summary`** — Best `s` vs `n` per family against the area lower bound and
  page values (imports the record tables from `sweep.py`).
- **`fig_seeds`** — Per-seed ablation outcomes (log gap) + wall-clock boxplot
  (the ~6:1 throughput cost of the LLM).
- **`fig_beats`** — Horizontal bars of the margins by which the three 2026
  records were beaten.
- **`fig_density`** — Packing density vs `n` for triangles-in-tans, this work vs
  page.
- **`fig_opsfamily`** — Heatmap of each move operator's improvement rate per
  family (from logs).
- **`fig_slack`** — Boxplots of slack over the area bound `s/s_lower − 1` by
  family.
- **`fig_audit`** — Log-log scatter of every reported packing's max overlap vs
  max containment violation, all far under the verifier's 10⁻⁹ line (the
  feasibility-discipline figure).
- **`fig_contacts`** — Mean contact-graph size vs `n` in the logged states
  (motivates the contact-graph state representation).

**`main()`** — `--all` or `--only <names>`; `--quick` shrinks live-compute
budgets (restart/trace/grad-state counts) for a sandbox smoke run. Dispatches via
`globals()[f"fig_{name}"]`.


## 18. How it all connects (data flow)

```
                      ┌───────────────────────── packer.py ─────────────────────────┐
   random restarts ──►│  init_config → solve_attempt (shrink loop: local_min on Φ,   │
                      │  basin-hop jitter, shrink_mult)  →  elite pool (top-k)        │
                      │        │                                                      │
                      │        ▼  canonical_state (moves.py)                          │
                      │   PROPOSER:  sample_move (random)  OR  LLMProposer.propose    │
                      │        │ discrete move JSON                                   │
                      │        ▼  apply_move → solve_attempt (re-shrink) → keep better│
                      │        │                                                      │
                      │        ├─(--log)─►  logs/moves.jsonl.<pid>  (state,move,Δs)   │
                      │        ▼                                                      │
                      │   endgame (backoff shrink + audit)  →  save_outputs (json+png)│
                      └──────────────────────────────────────────────────────────────┘
                               │                                        │
        make_dataset.py ◄──────┘                                        ▼
        (SFT positives + DPO pairs, held-out splits)          verify.py (independent 50-digit)
                │                                                        │
        train_sft.sh (QLoRA SFT, MLX) → train_dpo.py (DPO, TRL)         ▼
                │                                              submit.py → email Friedman
        llm_proposer.py  ◄── model ──┘                         (registry credit)
                │
        back into packer.py --proposer llm   (and ablation.py measures llm vs random)
```

Two loops share one action space (`moves.py`): the **search loop** (find records)
and the **data loop** (log its own behavior → train the policy → feed it back).
`sweep.py` drives the search overnight; `figures.py` turns runs+logs into the
paper; `validate.py`/`verify.py` are the correctness guardrails at both ends.

## 19. How to run it (condensed from RUNBOOK.md)

```bash
# 0. setup + sanity gate (must all PASS)
pip install jax scipy numpy matplotlib joblib mpmath mlx-lm
python validate.py

# 1. hunt records + generate data overnight (no ML needed)
python sweep.py --plan tier1 --attempts 500 --elite-rounds 6
cat runs/summary.csv                        # any beat=True?
python verify.py runs/<beat>.json           # independent check
python submit.py runs/<beat>.json --record <page value> --name "Akshaj Shandilya"

# 2–4. build data, then train the policy
python make_dataset.py --logs 'logs/moves.jsonl.*' --out data --holdout-n 6 13
bash train_sft.sh                           # QLoRA SFT (~2–4h)
python train_dpo.py --base <HF export of model_sft>   # optional DPO

# 5. the paper's central experiment
python ablation.py --model model_sft --seeds 3 --attempts 200 --rounds 4
python figures.py --all

# 6. LLM-guided hunts
python sweep.py --plan tier1 --proposer llm --model <best model> --elite-rounds 8
```

Golden rule: a "beat" is not a record until `verify.py` passes **and** you quote
`s` conservatively (round up at the last digit you trust).

## 20. Curated reading list + math notes

Grouped by the part of the system each illuminates, with why it matters and where
it lives in the code. (These are the works actually cited in
`paper/sn-bibliography.bib`, verified July 2026.)

### The paradigm — "LLM proposes, evaluator disposes"
- **Romera-Paredes et al., *Mathematical discoveries from program search with
  large language models* (FunSearch), Nature 625:468–475, 2024.** The landmark:
  an LLM proposes candidate *programs*, a classical evaluator scores and selects
  them, and the loop discovers new results in extremal combinatorics. This repo
  is the same paradigm with **moves instead of programs** — keeping the numerical
  work classical. Read it to understand the whole design philosophy behind
  `worker_elite` + the proposer split.
- **Yang et al., *Large Language Models as Optimizers* (OPRO), ICLR 2024.** Uses
  a *prompted* (not fine-tuned) LLM as a zeroth-order optimizer that reads a
  trajectory and proposes the next point. The simpler cousin of this system;
  contrast it to see what fine-tuning on self-generated data buys you.
- **Novikov et al., *AlphaEvolve: A coding agent for scientific and algorithmic
  discovery*, Google DeepMind tech report, 2025 (arXiv:2506.13131).** The
  scaled-up evolutionary-program agent. Useful as the "what heavy neural
  machinery does" contrast — and a reminder (see the Flamethrower note) that
  careful classical engineering still dominates on high-precision geometry.

### The classical optimization core
- **Wales & Doye, *Global Optimization by Basin-Hopping…*, J. Phys. Chem. A
  101(28):5111–5116, 1997.** The outer loop, from computational chemistry:
  kick + re-minimize to escape local minima. This *is* `solve_attempt`'s
  jitter-and-retry logic (§2.6). The math intuition: transform the energy
  landscape into basins of attraction, then hop between them.
- **Liu & Nocedal, *On the limited memory BFGS method for large scale
  optimization*, Math. Programming 45:503–528, 1989.** L-BFGS, the inner
  minimizer (`local_min`, §2.5). The key idea: approximate the inverse Hessian
  from the last few gradient/step pairs (limited memory), getting
  curvature-aware steps without storing an `n×n` matrix.
- **Bradbury et al., *JAX: composable transformations of Python+NumPy programs*,
  2018 (github.com/jax-ml/jax).** The autodiff engine behind exact gradients
  (§2.4). Read the autodiff/`grad` and `jit` docs to understand how `penalty()`
  becomes `value_and_grad` for ~one evaluation's cost.

### The packing literature
- **Friedman, *Packing unit squares in squares: a survey and new results*,
  Electronic J. Combinatorics DS7.** The survey by the registry's maintainer;
  the entry point to the whole field and the size conventions the code matches.
- **The Packing Center — erich-friedman.github.io/packing.** The ledger itself:
  browse the tan/L families in `TARGETS.md` and see the `s = 3.531+` truncation
  notation (which is why "beat the 4th decimal" and the safe-quote logic in
  `verify.py` exist).
- **Specht, *Packomania* — packomania.com.** The circle-packing counterpart,
  decades deep — the "don't compete here" reference for `TARGETS.md`'s Tier-4
  caution.
- **Melissen, *Packing and Covering with Circles*, PhD thesis, Utrecht, 1997**
  and **Xu, *On the minimum distance determined by n ≤ 7 points in an isosceles
  right triangle*, Acta Math. Appl. Sinica 12(2):169–175, 1996.** The proofs of
  small-n optima that `validate.py` reproduces (2 & 3 circles in a tan = 2+2√2,
  4+√2). These are your ground truth: if the engine can't hit these, nothing
  else is trustworthy.
- **Lubachevsky & Graham, *Curved Hexagonal Packings of Equal Disks in a
  Circle*, Discrete & Comput. Geom. 18:179–194, 1997.** Classic event-driven
  ("billiards") compaction — the other main family of packing methods, good
  context for why this project chose penalty+gradient instead.
- **Flamethr0wer (Ignacio Vallejo), *polygon-packer* (GitHub, 2026) and the video
  *I Got 122 World Records To Prove A Point* (2026).** The prior open-source
  solver whose penalty→shrink→restart architecture this repo upgraded (exact
  gradients, non-convex shapes/containers, elite pool). The video carries the
  strategic lesson encoded in `TARGETS.md`: classical engineering beat
  contemporaneous LLM systems on regular-polygon families — so attack the *soft*
  tan/L families instead.

### The fine-tuning
- **Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.**
  Freeze the model, learn low-rank adapter matrices. The math: approximate the
  weight update `ΔW` as `B·A` with tiny inner rank `r`, training ~0.1% of params.
  This is the `LoraConfig(r=32, …)` in `train_dpo.py` and `--num-layers 8` in
  `train_sft.sh`.
- **Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS
  2023.** LoRA on top of a **4-bit** frozen base (~4GB vs ~15GB) — what makes 7B
  fine-tuning fit on a 64GB laptop. This is the `-4bit` MLX base in
  `train_sft.sh`; the memory arithmetic is spelled out in `EFFICIENCY.md`.
- **Rafailov et al., *Direct Preference Optimization: Your Language Model is
  Secretly a Reward Model*, NeurIPS 2023.** The DPO loss the `(chosen, rejected)`
  pairs feed (§2.9). The insight: you can optimize a preference directly with a
  simple classification-style loss, skipping the separate reward model and RL of
  RLHF. `beta` in `train_dpo.py` is its temperature.
- **Hui et al., *Qwen2.5-Coder Technical Report*, 2024 (arXiv:2409.12186)** and
  **Hannun et al., *MLX: Efficient and flexible ML on Apple silicon*, 2023.** The
  base model (a code LLM, good at emitting clean JSON) and the training/inference
  framework used throughout (`train_sft.sh`, `llm_proposer.py`).

**Suggested reading order if you're new:** FunSearch (paradigm) → Wales & Doye +
Liu & Nocedal (the classical loop) → Friedman survey + the Packing Center
(the problem) → LoRA/QLoRA + DPO (the learning) → then re-read §2 of this guide
with the papers in hand.

## 21. Glossary (quick lookup)

| Term / file | One-liner |
|---|---|
| SAT | flashlight test: convex shapes are apart iff some axis separates their shadows (`pair_pp`, `sat_overlap`) |
| Φ (penalty) | total "spring energy" of all overlaps + boundary violations; 0 = legal (`penalty`) |
| ω | signed penetration depth of a pair (>0 overlap, <0 gap) |
| autodiff | exact chain-rule gradients at ~one evaluation's cost (JAX, `value_and_grad`) |
| L-BFGS-B | curvature-aware downhill minimizer (`local_min`) |
| basin hopping | kick (`jitter`) + re-minimize to escape a local valley (`solve_attempt`) |
| shrink loop | shrink S, re-relax, repeat; big steps far out, tiny near the bound (`shrink_mult`) |
| elite pool | keep best k configs, perturb *them* with a move instead of restarting cold |
| area lower bound | `S_lower = √(n·area_shape/area_container)`; unbeatable floor |
| canonical state | symmetry-fixed, rounded, contact-graph view the LLM sees (`canonical_state`) |
| contact graph | which shapes touch which (and the wall); the structure moves manipulate |
| move | one of rotate/swap/relocate/jiggle, coarse params, no precise float (`moves.py`) |
| relabel/unrelabel | map move indices canonical↔original |
| endgame + audit | 10⁻¹⁸-tol polish, then measure & clear worst violation to <10⁻¹¹ (`endgame`) |
| SFT | imitate improving moves (cross-entropy) |
| DPO | prefer chosen over rejected move from the same state (§2.9) |
| QLoRA | tiny trainable adapters over a frozen 4-bit model |
| fallback rate | % invalid LLM generations → random move; a health metric (`llm_proposer._report`) |
| `s` / `s = 3.531+` | container size; the `+` means the registry truncated the true digits |
| packer / moves / llm_proposer | engine / action space / inference wrapper |
| validate / verify | known-optima gate / independent 50-digit re-check |
| sweep / organize / submit | overnight driver / tidy results / registry bundle |
| make_dataset / train_sft / train_dpo | logs → data → SFT → DPO |
| ablation / figures | the fair race / manuscript figures + CSVs |

---

*Generated as a companion to the existing repo docs. Every function in every
`.py` file is covered above; cross-check any entry against the source and the
paper (`paper/paper.pdf`) — they were written to agree.*
