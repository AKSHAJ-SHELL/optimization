# Record hit list — Erich's Packing Center (scraped 2026-07-08)

## The landscape right now

A gold rush is in progress. Flamethrower ("Ignacio Vallejo", 122 records), Jonathan
Viquerat, and Thomas Greenleaf have swept the **regular-polygon-in-regular-polygon**
families in April–July 2026 — exactly what polygon-packer can express. Do NOT
compete there. The families below use **tans and L's**, which polygon-packer cannot
express, and they have not been touched in 14–21 years.

Rule of thumb for softness: a record stated as a bare decimal ("s = 3.307+") was
found by numeric search and is the most likely to be beatable. A record with a
closed form ("s = 2+√2") is a structured packing and is usually harder to beat.

## Tier 1 — Tans in Tans (nothing new since July 2009)

n tans (leg 1) in the smallest tan (leg s). 27 entries, holders Pegg/Cantrell/
Morandi/Friedman 2005–2009. Numeric-valued (softest) entries:

| n | record s | holder | year |
|---|----------|--------|------|
| 3 | 1.961+ | Ed Pegg | 2005 |
| 5 | 2.405+ | David Cantrell | 2005 |
| 7 | 2.824+ | Maurizio Morandi | 2007 |
| 10 | 3.307+ | Morandi | 2007 |
| 12 | 3.531+ | Morandi | 2007 |
| 17 | 4.181+ | Morandi | 2007 |
| 19 | 4.413+ | Cantrell | 2005 |
| 20 | 4.535+ | Morandi | 2007 |
| 23 | 4.886+ | Morandi | 2007 |
| 24 | 4.947+ | Morandi | 2007 |
| 27 | 5.241+ | Morandi | 2009 |

Closed-form but unproven (attack second): n=6 (2.690+), 22 (4.767+), 26 (5.121+).
"Trivial" entries (n=8: 2√2, n=16: 4, n=18: 3√2, …) are unproven too — tilted
packings sometimes beat trivial ones; probe them after the numeric ones.

    python packer.py --shape tan --n 10 --container tan --attempts 3000 --record 3.3075

## Tier 2 — Tans in L's (nothing new since August 2012)

n tans (short side 1) in the smallest L (short side s). Numeric entries:
n=10 (1.400+), 13 (1.499+), 14 (1.607+), 15 (1.680+) — all Morandi 2012.
Closed-form: n=3 (0.911+), 4 (0.965+), 7 (1.138+), 8 (1.276+), 9 (4/3),
16 (1.747+), 17 (1.804+). Page ends at n=17 → n=18+ are unclaimed territory.

    python packer.py --shape tan --n 13 --container L --attempts 3000 --record 1.4995

## Tier 3 — L's in Tans (Morandi 2012 / Jan 2025; mostly "Trivial")

n L's (short side 1) in the smallest tan (short side s). Only ~7 non-trivial
entries; the many Trivial ones (axis-aligned constructions) are the opportunity —
rotated packings may beat them. Page ends at n=22.

    python packer.py --shape L --n 10 --container tan --attempts 3000 --record 8.4853

## Tier 4 — Circles in Tans, n = 11–21 (2005–2009) — CAUTION

n=1–9ish are PROVEN (Xu 1996, Melissen 1997) — do not attack. n=11–21 are
Cantrell/Morandi/Specht 2005–2009. WARNING: Packomania hosts a
circles-in-right-isosceles-triangle page that may already exceed Friedman's stale
values — cross-check packomania.com/crt before claiming anything here.

## Avoid

- Triangles/squares/pentagons/hexagons/octagons in anything regular: 2026 rush.
- Circles in circles/squares/hexagons: Packomania (Specht) — decades of tuning.
- Squares in squares: now maintained on David Ellsworth's page, heavily optimized.

## Workflow per target

1. Run with `--attempts 2000+ --workers -1` (overnight is better), `--record <s>`.
2. If it beats the record: rerun endgame implicitly (automatic) and check the
   audit line — both violations must be < 1e-11.
3. Eyeball the PNG: contacts should look tight and intentional.
4. Re-verify independently: load the JSON, recompute all pairwise clearances in
   mpmath at 50 digits before believing it.
5. Submit picture + coordinates + s value to Erich Friedman (contact on his
   site). He verifies and credits by name — this is the scoreboard.

## Records stated on the pages, for --record flags

Tans in tans: 3:1.961, 5:2.405, 6:2.690, 7:2.824, 10:3.307, 12:3.531, 15:3.914,
17:4.181, 19:4.413, 20:4.535, 21:4.681, 22:4.767, 23:4.886, 24:4.947, 26:5.121,
27:5.241. (Quote as upper bounds: beat the 4th decimal to be safe.)
Tans in L's: 3:0.9114, 4:0.9659, 7:1.1381, 8:1.2761, 9:1.3333, 10:1.400,
13:1.499, 14:1.607, 15:1.680, 16:1.7474, 17:1.8047.
