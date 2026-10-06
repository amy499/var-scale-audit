# Comparison logic: native stage, normalized progress, and the frozen bands

Owner: P4. Status: **proposed, awaiting P3 sign-off on the comparison logic** (see §8).
Scope: the shared stage-alignment deliverable `docs/HANDOVER.md` §1 lists as owed by P4, plus the
arm construction P4's own Gate B pilot is built on.

Everything here is derived from the `stages` table `runner/generate.py` already writes into every
`run.json`. No change to `runner/`, `configs/`, `scripts/` or `third_party/`; no GPU; no dependency on
another lane's output. Every number below is recomputed from the frozen configs by
`python lanes/p4/check_p4_shared.py`, which fails if any of them moves.

---

## 1. The one claim this document does not make

**VAR scales and DiT timesteps are not mechanically equivalent, and nothing here says they are.**
A VAR scale is a resolution step that commits a fixed number of tokens and never revisits them. A DiT
timestep is one denoising move along a noise schedule over a latent of fixed size. They are matched by
*intervention intent and budget*, never by operation (`VAR Audit Integrated Research and Execution
Plan.md` §4).

What the common axis gives is a way to say "the middle of generation" and have it mean a defined, stated
thing for each model. Every reported result carries the native stage as well — VAR `si`, DiT
`timestep_in` — so a reader can always go back to the model's own units.

## 2. Two axes per stage, with different jobs

| Axis | Job | VAR | DiT |
|---|---|---|---|
| `p_place` | **Placement.** Bands are cut on this. Every figure axis and every arm's stage set derives from it. | `si / (SN - 1)` | `j / (S - 1)`, `j` counting steps in the **baseline** schedule |
| `p_func` | **Reporting convention.** Travels beside every stage so a reader can see how much was actually committed at a stage the placement axis calls "middle". Nothing is matched or cut on it. | `cum_tokens / total_tokens` | `sqrt(alpha_bar_in)`, the signal committed |

Both are in `[0, 1]` and non-decreasing in sampling order. `p_place` reaches exactly 0 at the first
stage and exactly 1 at the last. `p_func` does not start at 0 — VAR's first scale already commits 1 of
680 tokens (0.0015), and DiT's first step already carries `sqrt(ᾱ) = 0.006` — which is a property of the
models, not of the mapping.

`p_place` is the `p` the runner already passes to every hook and records per stage, so adopting it
changes no existing code. For DiT it is read from that recorded field rather than recomputed from the
table's length, because a `--skip-timesteps` run has fewer rows: recomputing would shift the
denominator and the remaining steps would stop lining up with their baseline.

**Why `sqrt(ᾱ)` and not `1 − σ` for DiT's functional axis.** `sqrt(ᾱ)` is the bounded image of
log-SNR, the standard parameterization of the forward noising geometry (`ᾱ = sigmoid(λ)`), which makes
it defensible in the report. `1 − σ` is more lopsided than the token axis it is supposed to balance
against: see §4.

## 3. The frozen bands

Half-open `[lo, hi)` on `p_place`, cut at exactly **1/3** and **2/3** of the progress value, with the
final band closed at 1.0. A stage whose progress lands exactly on a cut belongs to the **upper** band;
the last stage (`p_place = 1.0`) is always late. The cuts come from the schedule alone — the number of
scales or steps and nothing else. No P1–P3 finding feeds into them, which is what keeps P4's pilot
independent of the other lanes.

| Model | early | middle | late |
|---|---|---|---|
| VAR-d20 (10 scales) | `si` 0, 1, 2 | `si` 3, 4, 5 | `si` 6, 7, 8, 9 |
| DiT-XL/2 (250 steps) | 83 steps, `t` 999…670 (`j` 0–82) | 83 steps, `t` 666…337 (`j` 83–165) | 84 steps, `t` 333…0 (`j` 166–249) |

VAR's lists hold for every 256px depth d16–d30: they all share `patch_nums`
`[1, 2, 3, 4, 5, 6, 8, 10, 13, 16]`, so all have the same 10 scales and 680 tokens. DiT's full
250-timestep lists print from `python -m lanes.p4.bands --full`. `lanes/p4/bands.py` keeps the band
function general, so a tiny test model or another schedule still bands correctly.

**What `p_func` says about these bands, which is why it is reported.** VAR's middle band (`si` 3–5) holds
**77 of 680 tokens — 11.3%**. Calling it "the middle of generation" without that number beside it would
misdescribe it. DiT's middle band spans `sqrt(ᾱ)` from 0.105 to 0.564.

## 4. The alternatives, and the numbers that rejected them

Equal thirds under each candidate basis, recomputed by `check_p4_shared.py`:

| Model | Basis | early | middle | late | Verdict |
|---|---|---|---|---|---|
| VAR | **scale index** | `si` 0–2 | `si` 3–5 | `si` 6–9 | **chosen for placement** (3/3/4) |
| VAR | token mass | `si` 0–6 | `si` 7–8 | `si` 9 | rejected: 7 scales against 1 |
| DiT | **step index** | 83, `t` 999–670 | 83, `t` 666–337 | 84, `t` 333–0 | **chosen for placement** |
| DiT | normalized timestep | 83, `t` 999–670 | 83, `t` 666–337 | 84, `t` 333–0 | identical to step index (see §7) |
| DiT | signal `sqrt(ᾱ)` | 134, `t` 999–465 | 46, `t` 461–281 | 70, `t` 277–0 | rejected as a placement basis |
| DiT | noise removed `1 − σ` | 190, `t` 999–241 | 34, `t` 237–104 | 26, `t` 100–0 | rejected: most lopsided of all |

The index rows are the only ones that give **both** models a usable three-way split. Placement must be
the index axis because the alternative is degenerate: equal thirds by VAR token mass put one scale in
late against seven in early, which makes a protect/damage contrast meaningless — there is no budget you
can spend inside a one-scale band and also outside it at the same size.

The functional values are not discarded by that choice. They travel with every stage, so no claim rests
on the index axis alone, and switching which axis a figure *reports* is a re-plot, not a re-run.

## 5. Arm construction and the fixed reduction budget

For a candidate band `B` of size `k` out of `n` native stages, at budget `m`:

| Arm | Stages intervened | Count | Role |
|---|---|---|---|
| baseline | none | 0 | the reference every row is paired against |
| protect | `m` drawn from **outside** `B` | `m` | the exploitation condition: spend the cut away from `B` |
| damage | `m` drawn from **inside** `B` | `m` | the reverse condition: spend the cut on `B` |
| control | `m` drawn **uniformly across all `n`** | `m` | tests whether `B` is special or any `m` stages behave alike |

All three intervened arms use **one budget and one strength**, so the contrast is *where* the cut lands,
not how much or how hard. Cross-model matching is on the fraction `m / n` and on the band, never on the
operation.

Three consequences stated rather than discovered:

- `m` cannot exceed `B`'s **eligible**-stage count, because damage must fit inside `B`. That is `k`,
  except for DiT's early band, where the unskippable first step leaves `k − 1 = 82`.
- The swept reduction therefore cannot exceed `k / n`: **0.3** for VAR-d20 and **0.332** for the
  250-step DiT schedule. Figure 4 describes a bounded curve, not a general reduction curve.
- On DiT, equal `m` alone is not equal severity: consecutive skipped timesteps merge into one long
  transition, so a packed selection and a spread one differ sharply at the same budget. Every arm
  therefore picks its `m` stages at a **fixed stride within its own eligible set**, and records the
  resulting merged-window count and lengths, so the arms are shown to match in structure as well as in
  count.

**This narrows the frozen contract's wording.** `VAR Audit Integrated Research and Execution Plan.md`
§4 reads "protect candidate important scales while pruning/degrading the others". Taken literally,
protect would intervene on `n − k` stages against damage's `k`, so the two arms would differ in budget
as well as in placement and neither result could be attributed to placement alone. The fixed budget
keeps the intended contrast and is what makes a quality–compute curve possible at all. **By this
plan's own authority rule, that departure is P1's call, not P4's — it is listed in §8 as a question for
P1, to be answered before any GPU time is spent on the protect arms.**

### The operations, per model

| Model | Operation | Why this one |
|---|---|---|
| VAR | at `after(si)`, `f_hat = λ·f_hat + (1 − λ)·f_hat_before`, with **λ = 0 at Gate B** | λ = 0 is the only severity with a tested precedent in the repo (`scripts/phase3/hooks_lib.py:restore_all`) and the only one whose intent — the scale's contribution is gone — maps onto a removed DiT step. Rejected zeroing `h_BChw` at `before(si)`: `docs/hook_interface.md` §10a shows it leaves a `r · bias_k` residue, so it means "contributed the zero embedding", not "contributed less". |
| DiT | `--skip-timesteps` | The single supported lever that actually removes model evaluations while keeping pairing, via burned draws. The first step cannot be skipped (`runner/dit_model.py` rejects it), so every DiT arm's selection excludes `t = 999`. |

**VAR reports no compute saving.** The VAR intervention runs *after* the transformer pass, so removing a
scale's contribution saves nothing. Figure 4's shared x-axis is therefore the intervened fraction
`m / n`, with measured savings reported for DiT only and their absence stated on the figure for VAR.

## 6. What every result carries

Per `docs/HANDOVER.md` §3 and R1/R15: the native stage **and** `p_place` **and** `p_func`; the arm label;
the config, manifest and batch size (16, one per experiment); and each row's `generator_state_sha256`
checked equal to its baseline's. A run that fails that pairing check is reported as invalid, not
repaired.

## 7. Two corrections this sign-off should carry to P1

Both files are outside P4's write scope, so they travel to P1 rather than being edited here.

1. **`docs/hook_interface.md` "Open questions" 2** calls DiT's step-count and noise-level bases "nearly
   equal" for the 250-step schedule. They are not. At the index midpoint (`j = 124`, `t = 502`) the step
   basis reads **0.498** while `1 − σ_in` reads **0.039**. The quantity that sits at 0.4975 there is
   *normalized timestep*, not noise level — which is also why the step-index and normalized-timestep
   rows of §4 are identical. The parenthetical should name normalized timestep.
2. **`docs/HANDOVER.md` glossary** describes `p` as "comparable across models" without qualification.
   Once the basis above is signed off, that line should say what the comparison is: matched *intervention
   intent and budget* on a shared placement axis, not mechanical equivalence of a scale and a timestep.

## 8. Sign-off

**P3 signs off on the comparison logic** (`VAR Audit…Plan.md` §3). **P1 answers the §5 narrowing** of the
protect arm, which is a departure from the frozen contract's wording.

What specifically needs P3's answer:

1. Splitting the axis by role — placement on the index axis, the functional value as a reporting
   convention carried beside it (§2) — rather than picking one basis for everything.
2. The band lists in §3, **specifically**. This is the expensive reversal. Bands are cut on the
   placement axis, so they define the arm stage sets the pilot runs: a change invalidates those runs and
   the GPU time is spent again.
3. The arm construction and fixed budget in §5, including the `m / n ≤ k / n` ceiling.

| Item | Record |
|---|---|
| Sent to P3 | *(fill when sent — owed before the pilot job is submitted, not merely before the 12–18 Oct sweep)* |
| Response window | 48 hours from sending, given Gate B on 11 Oct |
| Outcome | *(fill: signed off / objected / no response by the deadline)* |

**If P3 does not respond within the window**, that non-response is recorded here beside the band lists
and P4 proceeds on the predeclared bands. The independence rule requires it: P4's pilot and primary
protocol use predeclared candidate stages and must not block on another lane.

**If P3 objects after other lanes have rendered figures**, the response depends on what is objected to:

- **The reported axis** (§2) — cheap. Both axes are computed for every stage and carried in every tidy
  table, so changing which one a figure reports is a re-plot, not a re-run.
- **The band lists** (§3) — expensive. Every lane's early/middle/late figure and every P4 arm set is
  invalidated, and P4's pilot GPU time is spent again. It stops being a lane edit and becomes a
  group-level decision: P4 raises it with P1 and P3 together, the new lists are frozen once, and every
  lane re-renders from its existing tidy tables before any new runs are submitted.

## 9. Where this is implemented

| File | What |
|---|---|
| `lanes/p4/progress.py` | both axes from a `stages` table; no model, no GPU |
| `lanes/p4/bands.py` | the cuts, the frozen lists, `assign_bands` |
| `lanes/p4/fixtures.py` | the reference stage tables the band lists are derived and checked from |
| `lanes/p4/arms.py` | the arm → stage-set logic of §5, with the merged-window record |
| `lanes/p4/hooks.py` | the VAR severity-scaled blend of §5 |
| `lanes/p4/plotting.py`, `lanes/p4/PLOTTING.md` | the shared template and its contract for P1–P3 |
| `lanes/p4/check_p4_shared.py` | recomputes every number in §3 and §4 and fails if one moves |
