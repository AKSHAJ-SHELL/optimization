"""Discrete move schema + canonical state representation.

This is the shared action space of the random proposer, the LLM proposer,
and the dataset builder. The policy (random or LLM) only ever chooses a move
from this schema with coarse parameters; the classical solver (L-BFGS shrink /
polish in packer.py) resolves all precise coordinates afterwards. The model
never emits a precise float.

Move ops (JSON):
  {"op": "rotate",   "shape": i, "deg": 45|-45|90|-90|180}
  {"op": "swap",     "a": i, "b": j}
  {"op": "relocate", "shape": i, "fx": 0.00-1.00, "fy": 0.00-1.00}   # bbox fractions
  {"op": "jiggle",   "shapes": [i, ...], "mag": 0.05|0.1|0.2}

Shape indices in logged/LLM-facing moves are CANONICAL indices (the order used
in canonical_state); use state["order"] to map back to solver indices.
"""

import json
import math

import numpy as np

ROT_CHOICES = [45, -45, 90, -90, 180]
MAG_CHOICES = [0.05, 0.1, 0.2]
OPS = ("rotate", "swap", "relocate", "jiggle")


# ----------------------------------------------------------------------------
# Sampling (the random proposer / behavior policy)
# ----------------------------------------------------------------------------

def sample_move(rng, n):
    """Sample a random move in ORIGINAL solver indices."""
    r = rng.random()
    if r < 0.35 or n < 2:
        k = max(1, int(rng.integers(1, max(2, n // 3 + 1))))
        shapes = sorted(int(i) for i in rng.choice(n, size=min(k, n),
                                                   replace=False))
        return {"op": "jiggle", "shapes": shapes,
                "mag": float(rng.choice(MAG_CHOICES))}
    if r < 0.60:
        return {"op": "rotate", "shape": int(rng.integers(0, n)),
                "deg": int(rng.choice(ROT_CHOICES))}
    if r < 0.75:
        a, b = rng.choice(n, size=2, replace=False)
        return {"op": "swap", "a": int(a), "b": int(b)}
    return {"op": "relocate", "shape": int(rng.integers(0, n)),
            "fx": round(float(rng.random()), 2),
            "fy": round(float(rng.random()), 2)}


def apply_move(x, S, prob, move, rng):
    """Apply a move (ORIGINAL indices) to config x at container size S."""
    x = np.asarray(x, dtype=float).copy()
    n = len(x) // 3
    op = move["op"]
    if op == "rotate":
        i = int(move["shape"]) % n
        x[3 * i + 2] += math.radians(float(move["deg"]))
    elif op == "swap":
        a, b = int(move["a"]) % n, int(move["b"]) % n
        for d in range(3):
            x[3 * a + d], x[3 * b + d] = x[3 * b + d], x[3 * a + d]
    elif op == "relocate":
        i = int(move["shape"]) % n
        x0, x1, y0, y1 = prob.container_bbox(S)
        x[3 * i] = x0 + float(move["fx"]) * (x1 - x0)
        x[3 * i + 1] = y0 + float(move["fy"]) * (y1 - y0)
        x[3 * i + 2] = rng.uniform(0, 2 * math.pi)
    elif op == "jiggle":
        mag = float(move.get("mag", 0.1))
        for i in move["shapes"]:
            i = int(i) % n
            x[3 * i] += rng.normal(0, mag * S)
            x[3 * i + 1] += rng.normal(0, mag * S)
            x[3 * i + 2] += rng.normal(0, 5 * mag)
    else:
        raise ValueError(f"unknown op {op!r}")
    return x


def relabel(move, inv):
    """Map a move's ORIGINAL indices to CANONICAL indices via inv[orig]=canon."""
    m = dict(move)
    if "shape" in m:
        m["shape"] = int(inv[m["shape"]])
    if "a" in m:
        m["a"], m["b"] = int(inv[m["a"]]), int(inv[m["b"]])
    if "shapes" in m:
        m["shapes"] = sorted(int(inv[i]) for i in m["shapes"])
    return m


def unrelabel(move, order):
    """Map CANONICAL indices back to ORIGINAL via order[canon]=orig."""
    m = dict(move)
    if "shape" in m:
        m["shape"] = int(order[int(m["shape"])])
    if "a" in m:
        m["a"], m["b"] = int(order[int(m["a"])]), int(order[int(m["b"])])
    if "shapes" in m:
        m["shapes"] = sorted(int(order[int(i)]) for i in m["shapes"])
    return m


def validate_move(move, n):
    """Schema-check a (possibly LLM-emitted) move. Returns cleaned move or None."""
    try:
        op = move.get("op")
        if op == "rotate":
            return {"op": "rotate", "shape": int(move["shape"]) % n,
                    "deg": int(move["deg"])} if int(move["deg"]) in ROT_CHOICES \
                else None
        if op == "swap":
            a, b = int(move["a"]) % n, int(move["b"]) % n
            return {"op": "swap", "a": a, "b": b} if a != b else None
        if op == "relocate":
            fx, fy = float(move["fx"]), float(move["fy"])
            if 0 <= fx <= 1 and 0 <= fy <= 1:
                return {"op": "relocate", "shape": int(move["shape"]) % n,
                        "fx": round(fx, 2), "fy": round(fy, 2)}
            return None
        if op == "jiggle":
            shapes = [int(i) % n for i in move["shapes"]][:max(1, n)]
            mag = float(move.get("mag", 0.1))
            mag = min(MAG_CHOICES, key=lambda c: abs(c - mag))
            return {"op": "jiggle", "shapes": sorted(set(shapes)), "mag": mag}
        return None
    except (KeyError, TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------
# Canonical state
# ----------------------------------------------------------------------------

def canonical_state(prob, x, S, tol=1e-5):
    """Symmetry-reduced, low-precision view of a packing for the policy.

    Returns dict with:
      family, n, s, lower_bound, shapes (canonical order, 3-decimal coords,
      degrees), contacts (canonical index pairs; 'wall' for boundary),
      order (order[canon] = original index).
    """
    x = np.asarray(x, dtype=float)
    n = prob.N
    pos = x.reshape(n, 3)
    key = [(round(float(np.hypot(px, py)), 2),
            round(float(math.atan2(py, px)), 2), i)
           for i, (px, py, _) in enumerate(pos)]
    order = [i for _, _, i in sorted(key)]
    inv = {orig: c for c, orig in enumerate(order)}

    shapes = []
    for c, orig in enumerate(order):
        px, py, th = pos[orig]
        deg = math.degrees(th) % 360.0
        shapes.append({"i": c, "x": round(float(px), 3),
                       "y": round(float(py), 3), "deg": round(deg, 1)})

    contacts = []
    try:
        pair_gaps, wall_gaps = prob.gaps(x, S)
        for (si, sj), g in pair_gaps.items():
            if g > -tol:
                contacts.append([int(inv[si]), int(inv[sj])])
        for si, g in enumerate(wall_gaps):
            if g > -tol:
                contacts.append([int(inv[si]), "wall"])
    except Exception:
        pass

    return {"family": f"{prob.shape_name} in {prob.container_name}",
            "n": n, "s": round(float(S), 4),
            "lower_bound": round(prob.S_lower, 4),
            "shapes": shapes, "contacts": contacts, "order": order}


PROMPT_TEMPLATE = """You are a geometric packing optimizer. {n} unit {shape}s are packed in a {container} of size s = {s} (theoretical lower bound {lb}). Propose ONE move that could allow the container to shrink after re-optimization.

Shapes as i:(x, y, deg):
{shapes}
Touching pairs (or wall): {contacts}

Allowed moves (respond with exactly one JSON object, nothing else):
{{"op":"rotate","shape":i,"deg":45|-45|90|-90|180}}
{{"op":"swap","a":i,"b":j}}
{{"op":"relocate","shape":i,"fx":0.0-1.0,"fy":0.0-1.0}}
{{"op":"jiggle","shapes":[i,...],"mag":0.05|0.1|0.2}}

JSON:"""


def render_prompt(state):
    sh = "\n".join(f'{s["i"]}:({s["x"]}, {s["y"]}, {s["deg"]})'
                   for s in state["shapes"])
    ct = ", ".join(f'{a}-{b}' for a, b in state["contacts"]) or "none"
    shape, container = state["family"].split(" in ")
    return PROMPT_TEMPLATE.format(n=state["n"], shape=shape,
                                  container=container, s=state["s"],
                                  lb=state["lower_bound"], shapes=sh,
                                  contacts=ct)


def parse_move_json(text, n):
    """Extract the first JSON object from LLM output; validate against schema."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for end in range(start, min(len(text), start + 500)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return validate_move(json.loads(text[start:end + 1]), n)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
