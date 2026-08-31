# The Whole Story: how this project got six names on a math registry

This is the plain-language + math walkthrough of everything in this folder.
Read it top to bottom once and you'll understand every file, every formula,
and every decision. Companion docs: DESIGN.md (architecture), RUNBOOK.md
(operations), EFFICIENCY.md (memory), paper/paper.pdf (the formal writeup).

---

## 1. What happened, in order

1. **The idea.** Use an LLM inside a geometric optimizer to hunt packing
   records on Erich Friedman's Packing Center — a public ledger of
   "smallest container found so far" results maintained since the 1990s.
2. **The research.** We scraped the registry and found the strategic map:
   families made of regular polygons had just been swept by automated
   solvers in 2026, but families involving *tans* (right isosceles
   triangles) and *L-shapes* hadn't been improved since 2005–2012.
3. **The engine.** We built a classical optimizer (packer.py): a physics-
   style penalty function with exact machine-computed gradients, a
   shrink-and-reoptimize loop, and a strict feasibility audit. Validated by
   re-discovering mathematically *proven* optima to 8–9 digits.
4. **The AI layer.** We gave a language model a small menu of discrete
   "moves" (rotate this piece, swap those two, relocate that one) and
   trained it — on data the optimizer generated about itself — to suggest
   which move to try. The solver still does all the precise math.
5. **The training saga.** First 7B training run ate 60+GB and froze the
   Mac (fixed: quantized QLoRA + gradient checkpointing → 6GB). Second run
   overfit and diverged to NaN after iteration 3500 (fixed: use the
   iteration-500 checkpoint, which had the best validation loss).
6. **The experiment.** Head-to-head, LLM-suggested moves vs random moves,
   same everything else. Result: ties on easy instances, better average on
   the hardest ones, and the two best packings of the study were found
   *only* via LLM moves — at ~6× the wall-clock cost.
7. **The records.** The engine (and on two candidates, the LLM) produced
   packings below registry values. Everything was re-verified by an
   independent 50-digit checker, emailed to Erich Friedman, and **six
   results are now on the registry credited to Akshaj Shandilya (July
   2026)**: improved records for 21, 22, 23 equilateral triangles in a
   tan, and brand-new entries for 24, 25, 26. Three more candidates
   (7, 10, 12 tans in a tan) await comparison against unpublished digits.
8. **The paper.** A Springer-format manuscript (paper/) documents all of
   it, with every figure regenerable from the code.

---

## 2. The problem (and why it's hard)

**The task:** fit n copies of a shape inside the smallest possible
container of a given form. Like packing n identical ice cream sandwiches
into the smallest possible cooler — you choose where each one goes and how
it's rotated, and you want the cooler as small as possible.

Formally, each shape i has a pose (x_i, y_i, θ_i) — position and rotation
— and the container has size s. We solve:

    minimize s
    subject to:  no two shapes overlap
                 every shape fits inside the container of size s

There's an easy lower bound: the shapes' total area can't exceed the
container's area. For n tans (area ½ each) in a tan container (area s²/2):

    s ≥ √n

You can never beat this; good packings get close to it. The gap between a
packing and this bound is "wasted space."

**Why it's hard:** the problem is two problems tangled together.

- A *continuous* problem: given the arrangement's rough layout, nudge every
  coordinate to squeeze out slack. Calculus is great at this.
- A *combinatorial* problem: WHICH layout? Which shape goes in which
  corner, which pair is flipped head-to-toe? There are astronomically many
  layouts, and calculus can't hop between them — like rearranging furniture:
  sliding the couch an inch is easy, but realizing the couch should swap
  places with the bookshelf is a different kind of decision.

Every method in this project maps onto one of those two halves.

---

## 3. The engine: turning geometry into calculus

### 3.1 Overlap as a number (the separating axis theorem)

To optimize, we need "how badly do these two shapes overlap?" as a smooth
number. For convex shapes there's a beautiful tool, the **separating axis
theorem (SAT)**: two convex shapes don't overlap if and only if there's
some direction along which their shadows don't overlap.

Relatable version: hold a flashlight at different angles and look at the
two objects' shadows on the wall. If from *some* angle the shadows
separate, the objects are apart. If the shadows overlap from *every*
angle, the objects genuinely interpenetrate. For polygons you only need to
check a few flashlight angles — the directions perpendicular to each edge.

The math: project both shapes onto axis a, get intervals; the overlap on
that axis is

    ov(a) = min(max_1, max_2) − max(min_1, min_2)

and the penetration depth is ω = min over axes of ov(a). Positive ω means
overlap; ω ≤ 0 means there's a separating axis. Circles use exact
distances instead (two circles overlap iff the center distance is less
than the sum of radii — no flashlight needed).

### 3.2 The penalty function (springs everywhere)

We sum up every violation, squared:

    Φ(x, s) = Σ max(0, ω_pair)² + Σ max(0, containment violation)²

Think of it as installing springs: any two shapes that overlap get a
spring pushing them apart, any shape poking out of the container gets a
spring pushing it back in. Φ is the total energy stored in the springs.
Φ = 0 exactly when the packing is legal. Squaring makes the energy smooth
near zero, which the optimizer needs.

### 3.3 Exact gradients (knowing the slope vs poking around)

To minimize Φ we need its gradient — which way to nudge each of the 3n
coordinates to reduce the energy. Two ways to get it:

- **Finite differences** (what the prior open-source solver used): wiggle
  each coordinate a tiny bit, re-measure Φ, estimate the slope. That's
  6n extra evaluations *per step* — like finding the steepest way down a
  hill blindfolded by taking a test step in every direction first.
- **Automatic differentiation** (what we use, via JAX): the computer
  applies the chain rule through the exact formula of Φ and returns the
  exact slope in roughly the cost of ONE evaluation. You simply know the
  slope. This single change is the biggest speed upgrade in the project.

We verified it: autodiff gradients agree with careful finite differences
to ~10⁻⁹ relative error (fig_gradcheck).

### 3.4 L-BFGS: rolling downhill with a memory

The minimizer, L-BFGS, is like a ball rolling downhill that also
remembers the shape of the terrain it recently crossed, letting it take
smart, curvature-aware steps instead of timid straight-downhill ones. It
crushes the spring energy to ~10⁻¹⁰ in a few hundred steps.

### 3.5 The shrink loop (tightening the belt)

Now the outer game: start with a comfortably large container. Find a
legal packing (springs relaxed, Φ ≈ 0). Then shrink the container by ~1%
and re-relax. Keep shrinking. When shrinking fails — the springs can't
relax anymore — give the configuration a random shake and try again
("basin hopping": kick the ball out of its valley and see if it rolls
into a deeper one). When even shakes fail, that restart is done. Hundreds
of restarts run in parallel from random layouts.

### 3.6 The elite pool (remix your best suitcases)

Independent restarts throw away everything they learned. Instead we keep
the best k configurations ("elites") and spend compute perturbing *them*:
rotate one piece 45°, swap two pieces, relocate one to a corner, jiggle a
few. Each perturbed variant gets the full shrink treatment. It's the
difference between repacking your suitcase from scratch every time and
taking your best attempt and trying one smart change to it.

That "one smart change" is exactly the slot where the AI goes.

### 3.7 The endgame and the audit (the inspector with a micrometer)

Search-phase tolerances allow overlaps around 10⁻⁵ — invisible, but a
record claim with any overlap is worthless. So the endgame re-polishes
with tolerance 10⁻¹⁸, shrinking with steps that decay geometrically from
10⁻³ down to 10⁻¹¹, and then an audit measures the true worst violation
and inflates s just enough to clear it. Reported packings have violations
below 10⁻¹¹ — about a thousandth of a wavelength of light, if the
container were a meter wide.

Then a *separate program* (verify.py), sharing zero code with the engine,
re-implements all the geometry in 50-digit arithmetic and re-measures
everything. Independent inspector, own instruments. This paranoia paid
off: while building the verifier, its own inside/outside test had a sign
bug that the engine's numbers exposed immediately — precisely because the
two implementations were independent.

---

## 4. The AI layer: a strategist who never touches a ruler

### 4.1 Why the model must not emit coordinates

LLMs are bad at the 4th decimal place, and packings die by the 4th
decimal place. So the model is never asked for numbers. It's the
experienced moving-crew foreman who says "swap the couch and the desk,
and rotate the piano a quarter turn" — and the crew (the solver) does all
the actual millimeter work. The model's entire output is one JSON move:

    {"op": "rotate",   "shape": 4, "deg": 90}
    {"op": "swap",     "a": 1, "b": 3}
    {"op": "relocate", "shape": 2, "fx": 0.25, "fy": 0.75}
    {"op": "jiggle",   "shapes": [0, 2], "mag": 0.1}

### 4.2 What the model sees (canonical state)

Two descriptions of the same packing should look identical to the model,
or it wastes capacity learning that they're the same. So before showing a
state we: sort the shapes by a fixed geometric rule (killing the n!
relabeling symmetry — like always describing a room starting from the
door and going clockwise), round coordinates to 3 decimals, and list the
**contact graph**: which shapes touch which, and which touch the walls.
Contacts are the load-bearing structure of a packing — telling the model
directly beats making it infer geometry from coordinates.

### 4.3 Where the training data came from (the optimizer taught it)

No dataset exists for "good packing moves," so the optimizer generated
one about itself. With logging on, every elite perturbation records:
(state, move tried, did it improve?). Overnight sweeps produced ~42,000
such records for free while also hunting records.

- Moves that improved → **SFT data** (supervised fine-tuning): standard
  next-token training, minimize cross-entropy of the good move's JSON
  given the state prompt. "Here's the situation, here's what worked —
  imitate it."
- States where both a good and a bad move were observed → **DPO pairs**
  (direct preference optimization). The DPO loss, for chosen move y⁺ and
  rejected move y⁻:

      L = −log σ( β·[log π(y⁺|x)/π₀(y⁺|x) − log π(y⁻|x)/π₀(y⁻|x)] )

  In words: adjust the model π (relative to its pre-training π₀) so the
  winning move becomes more likely than the losing one — without needing
  to know *how much* better it was. Like coaching by showing pairs of
  plays: "this one worked, that one didn't" is a strictly stronger signal
  than a pile of unlabeled plays.

### 4.4 QLoRA in one paragraph

Fine-tuning all 7.6 billion weights is absurd on a laptop. LoRA freezes
the model and learns tiny low-rank correction matrices on the side —
like leaving an engine intact and machining a small adapter plate —
about 0.1% of the parameters. QLoRA additionally stores the frozen
weights in 4-bit precision (4GB instead of 15GB). Our final recipe used
~6GB of memory total.

### 4.5 Two teachable failures

- **The 60GB freeze:** training memory ≈ weights + activations, and
  activations scale with batch × sequence length × layers. Full-precision
  weights, batch 4, sequence 2048, no gradient checkpointing → the
  activations alone dwarfed the weights and hit the 64GB ceiling.
  Gradient checkpointing (recompute activations during the backward pass
  instead of storing them — trade ~30% compute for ~10× memory) plus
  4-bit weights plus batch 1 fixed it. Details: EFFICIENCY.md.
- **The NaN divergence:** with only 540 positive examples, 4000 training
  iterations ≈ 7 epochs. Validation loss bottomed at iteration ~400 and
  then *rose* while training loss kept falling — the textbook overfitting
  curve, memorizing rather than learning — until the numbers blew up to
  NaN at 3500. The fix is the textbook one too: deploy the checkpoint
  with the best validation loss, not the last one.

---

## 5. The experiment (a fair race, honestly scored)

The question: do learned moves beat random moves? The design makes the
comparison exact — same engine, same move menu, same restart budget, same
seeds; the ONLY difference is who picks the moves. Three regimes emerged:

1. **Saturated instances** (small n): the initial restarts already find
   the best layout; no elite move of either kind improves anything;
   results identical seed-for-seed. Lesson: a proposal policy can't help
   when there's nothing left to propose.
2. **Hard instances**: LLM moves improved the average gap on two of four
   (including a size never seen in training), were mixed on one — and the
   two single best packings of the entire study (n=10 and n=12 tans)
   were reached only through LLM-proposed moves.
3. **The cost**: generating a move from a 7B model takes ~2s vs ~0
   for a random one; runs took ~6× longer. Per *proposal*, learned moves
   win; per *second*, random search can afford 6× more attempts. The
   paper reports both accountings — that's the honest framing.

---

## 6. The records (and what "3.531+" actually means)

Registry pages truncate: "s = 3.531+" means the true record starts with
3.531 — it lies somewhere in [3.531, 3.532). If you find 3.53107, you
don't know whether you beat the hidden digits until the maintainer
compares. That's exactly the situation of our three tans-in-tans
candidates (n=7, 10, 12 vs Morandi's 2007 values) — pending.

No such ambiguity for the triangles-in-tans results: margins of up to
0.025 dwarf any truncation. Those six are **confirmed on the registry**:

| n | old record (holder) | new value | status |
|---|---|---|---|
| 21 | 4.65799+ (Ananthan, 7/2026) | 4.64141+ | **Shandilya, July 2026** |
| 22 | 4.73289+ (Ananthan, 7/2026) | 4.73276+ | **Shandilya, July 2026** |
| 23 | 4.85470+ (Ananthan, 7/2026) | 4.82974+ | **Shandilya, July 2026** |
| 24–26 | (page ended at 23) | 4.91168+, 5.05586+, 5.10960+ | **new entries, Shandilya** |

Submission workflow, forever: sweep flags a beat → verify.py (independent
50-digit check) → submit.py (bundles picture + full-precision coordinates
+ message) → email erichfriedman68@gmail.com → he verifies and updates.

---

## 7. Reading list (what to read and why)

**The paradigm**
- Romera-Paredes et al., *Mathematical discoveries from program search
  with large language models* (FunSearch), Nature 625, 2024 — the
  landmark "LLM proposes, evaluator disposes" result. Our loop is this
  paradigm with moves instead of programs.
- Yang et al., *Large Language Models as Optimizers* (OPRO), ICLR 2024 —
  prompted LLMs as optimizers; simpler cousin of our approach.
- Google DeepMind, *AlphaEvolve* (2025 report) — the scaled-up program-
  evolution agent; useful contrast for what classical engineering beats.

**The classical optimization**
- Wales & Doye, *Global Optimization by Basin-Hopping...*, J. Phys. Chem.
  A 101, 1997 — the outer loop we use, from computational chemistry.
- Liu & Nocedal, *On the limited memory BFGS method*, Math. Programming
  45, 1989 — the inner minimizer.
- JAX (github.com/jax-ml/jax) — the autodiff system behind exact
  gradients.

**The packing literature**
- Friedman, *Packing unit squares in squares: a survey*, Electronic J.
  Combinatorics DS7 — the survey by the registry's maintainer; the entry
  point to the whole field.
- The Packing Center: erich-friedman.github.io/packing — the ledger
  itself; browse the families we hunted.
- Specht, packomania.com — the circle-packing counterpart, decades deep.
- Melissen, *Packing and Covering with Circles*, PhD thesis, Utrecht,
  1997 — proofs of small-n optima, incl. circles in isosceles right
  triangles (the values our validation gate reproduces).
- Flamethr0wer, *polygon-packer* (GitHub) and *I Got 122 World Records To
  Prove A Point* (video, 2026) — the classical solver whose architecture
  we upgraded, and the strategic lesson about soft vs hard families.

**The fine-tuning**
- Hu et al., *LoRA*, ICLR 2022; Dettmers et al., *QLoRA*, NeurIPS 2023 —
  parameter-efficient training on small hardware.
- Rafailov et al., *Direct Preference Optimization*, NeurIPS 2023 — the
  preference loss our pairs feed.

---

## 8. Glossary / file map

| Term / file | One-liner |
|---|---|
| SAT | flashlight test: convex shapes are apart iff some axis separates their shadows |
| Φ (penalty) | total spring energy of all overlaps and boundary violations; 0 = legal |
| autodiff | exact chain-rule gradients at the cost of ~one evaluation (JAX) |
| L-BFGS | downhill minimizer with terrain memory |
| basin hopping | kick + re-minimize to escape local valleys |
| elite pool | keep best k layouts, perturb them instead of restarting cold |
| canonical state | symmetry-fixed, rounded description + contact graph the LLM sees |
| SFT / DPO | imitate good moves / prefer good over bad moves from same state |
| QLoRA | tiny trainable adapters over a frozen 4-bit model |
| endgame + audit | 10⁻¹⁸-tolerance polish, then measure & clear worst violation |
| packer.py | engine: penalty, gradients, shrink, elites, endgame, logging, proposers |
| moves.py | move schema + canonicalizer (the action space) |
| sweep.py / organize.py | batch hunts with records wired in / tidy results + index |
| verify.py / submit.py | independent 50-digit inspector / registry submission bundler |
| make_dataset.py, train_sft.sh, train_dpo.py | logs → data → tuned model |
| ablation.py, figures.py | the fair race; Springer-format figures + CSVs |
| DESIGN / RUNBOOK / TARGETS / EFFICIENCY | architecture / operations / hit list / memory guide |
