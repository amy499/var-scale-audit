# Comparison logic — P3 sign-off needed

Owner: P4. Status: **waiting on P3**. Deadline: reply within 48 h; the pilot cannot be submitted
until this is answered, and changing the bands afterwards means re-running the GPU jobs.

## What P3 needs to decide

Three yes/no answers. Everything below this section is the evidence for them.

1. **Are the band lists in §1 right?** ← the one that matters. Bands define the arm stage sets the
   pilot runs, so changing them later invalidates the runs and every lane's figures.
2. **Is it OK that the two axes have different jobs** (§2) — bands cut on the index axis, the
   "how much is committed" value reported beside it — rather than one axis doing both?
3. **Is the four-arm construction in §3 sound**, including that all arms cut the same number of stages?

If P3 does not reply within 48 h, P4 proceeds on these bands and records the non-response here. The
independence rule requires it — P4's protocol uses predeclared stages and must not block on another lane.

| Record | |
|---|---|
| Sent to P3 | *(fill when sent)* |
| Outcome | *(fill: signed off / objected / no reply)* |

---

## 1. The bands

Cut at **1/3** and **2/3** of progress, half-open `[lo, hi)`, last band closed at 1.0. A stage landing
exactly on a cut goes to the upper band. The cuts come from the schedule alone — no P1–P3 finding feeds
in, which is what keeps P4's pilot independent.

| Model | early | middle | late |
|---|---|---|---|
| VAR-d20 (10 scales) | `si` 0–2 | `si` 3–5 | `si` 6–9 |
| DiT-XL/2 (250 steps) | 83 steps, `t` 999…670 | 83 steps, `t` 666…337 | 84 steps, `t` 333…0 |

VAR's list holds for d16–d30 (all share the same 10 scales / 680 tokens). Full DiT list:
`python -m lanes.p4.bands --full`.

**Worth knowing before you approve:** VAR's middle band holds **77 of 680 tokens (11.3%)**. It is a
third of the *scales*, not a third of the image content. That is why every result also reports how much
is actually committed (§2).

## 2. Why the index axis, and not the others

Equal thirds under each candidate. All recomputed by `lanes/p4/check_p4_shared.py`, which fails if any
number moves:

| Model | Basis | early / middle / late | Verdict |
|---|---|---|---|
| VAR | **scale index** | 3 / 3 / 4 scales | **chosen** |
| VAR | token mass | 7 / 2 / 1 scales | rejected — 7 scales against 1 |
| DiT | **step index** | 83 / 83 / 84 steps | **chosen** |
| DiT | normalized timestep | 83 / 83 / 84 steps | identical to step index |
| DiT | signal `sqrt(ᾱ)` | 134 / 46 / 70 steps | rejected |
| DiT | noise removed `1 − σ` | 190 / 34 / 26 steps | rejected — most lopsided |

The index rows are the only ones giving **both** models a usable three-way split. A one-scale "late"
band makes a protect/damage contrast impossible: there is no budget you can spend both inside and
outside a band of one.

So each stage carries two numbers:

| | What it is | Used for |
|---|---|---|
| `p_place` | VAR `si/(SN-1)`, DiT step index | bands, arm sets, figure axes |
| `p_func` | VAR `cum_tokens/total_tokens`, DiT `sqrt(alpha_bar_in)` | reported beside it, so "middle" is never read as "a third of the content" |

Both travel with every result, so if P3 prefers the other one *reported*, that is a re-plot, not a re-run.

**VAR scales and DiT timesteps are not mechanically equivalent, and nothing here claims they are.**
They are matched by intervention intent and budget only. Every result also carries the native stage.

## 3. The four arms

For a candidate band `B` of `k` stages out of `n`, at budget `m`:

| Arm | Cuts | Tests |
|---|---|---|
| baseline | nothing | the reference every row is paired against |
| protect | `m` stages **outside** `B` | spend the cut away from the candidate band |
| damage | `m` stages **inside** `B` | spend it on the band |
| control | `m` stages drawn across all `n` | is `B` special, or do any `m` stages behave alike? |

All three cut the **same number of stages at the same strength**, so the only difference is *where*.
Cross-model matching is on the fraction `m/n`, never on the operation.

Three consequences:

- **`m` cannot exceed `k`** (damage must fit inside `B`), so the swept reduction is capped at `k/n`:
  **0.3** for VAR, **0.332** for DiT. Figure 4 is a bounded curve, not a general reduction curve.
  DiT's early band caps one lower — its first step can never be skipped.
- **On DiT, equal `m` is not automatically equal severity**, because consecutive skipped steps merge
  into one long transition. Damage is the constrained arm, so its window structure is computed first
  and protect and control reproduce the same window lengths. The code fails the build if they don't.
- **For VAR's middle band at small `m`, protect lands entirely in late** (`si` 6–9). So the contrast is
  protect-in-late vs damage-in-middle. The control arm is what separates "this band is special" from
  "late is special" — state this beside any VAR middle-band result.

Operations: VAR blends `f_hat` back toward its pre-stage value at `after(si)` with **λ = 0** (the
scale's contribution is removed — the repo's tested `restore_all`). DiT uses `--skip-timesteps`.
**VAR reports no compute saving** — its intervention runs after the transformer pass, so it saves
nothing. Only DiT's savings are measured.

## 4. For P1, not P3

- **The protect arm is narrowed.** The plan §4 says "protect important scales while degrading the
  others", which taken literally gives protect `n − k` stages against damage's `k` — different budgets,
  so neither result could be attributed to placement. P4 uses one shared budget instead. **P1 should
  confirm before GPU time is spent.**
- **Two doc corrections.** `docs/hook_interface.md` open question 2 calls DiT's step and noise bases
  "nearly equal"; at the index midpoint they read **0.498** and **0.039**. The thing at 0.4975 is
  *normalized timestep*. And the `docs/HANDOVER.md` glossary calls `p` "comparable across models"
  unqualified — it should say matched intent and budget, not mechanical equivalence.

## 5. Where this lives

`progress.py` (both axes) · `bands.py` (the cuts and frozen lists) · `arms.py` (arm → stage sets) ·
`hooks.py` (the VAR operation) · `plotting.py` + `PLOTTING.md` (the shared template) ·
`check_p4_shared.py` (recomputes every number above).
