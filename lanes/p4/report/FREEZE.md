# P4 experiment freeze

**Frozen: Thursday 12 November 2026.** No P4 job is submitted after this date. Plan unit U9
(R16, R18).

The plan set the freeze for 5–6 Nov. It moved because:
- the pooled VAR control (3 Nov) needed two extra 111-run jobs, one of which had to be re-run twice;
- the shortlist's confirmation runs could only start once P3's tables were read from P3's branch
  (10 Nov).

The departures from the plan are listed below.

## Code

- **Branch:** `kmaniar/p4-phase-c-e`.
- **Submodules** (checked by `scripts/check_third_party.py` in every job log): `third_party/VAR` @
  78b95394fc58, `third_party/DiT` @ ed81ce222909.

Every job's `job.log` records the commit it ran:

| Jobs | Commit |
|---|---|
| The five full sweeps | 300a7a5 `feat(p4): predeclared shortlist and explicit-stage confirmation runs` |
| The two control pools | c4b7176 `feat(p4): pooled VAR control over every placement` |
| The two confirmation runs | 82cd939 `fix(p4): snapshot reads P3 from its branch, t-based agreement p, footnote fixes` |

The changes after 300a7a5 only add code paths (pool, shortlist, export). `pilot.py plan-full`
produces a byte-identical full-sweep plan at 300a7a5 and at the final head, for both models.

**Where everything ran:**
- All GPU jobs: NSCC ASPIRE2A, A100-SXM4-40GB, PBS, conda env `var-dit`.
- Export, compare, snapshot, shortlist and Figure 4: CPU, on the NSCC login node.
- Figure 5: rendered on a laptop (env `dev`, commit 28a8086) from the same snapshot and summary
  files, which were copied byte for byte with `rsync`.

## Run roots

All roots are under `~/var-scale-audit/` on NSCC. Every one ends `P4 SWEEP COMPLETE, PAIRED AND
SCORED`.

| Role | PBS job | Root | Commit |
|---|---|---|---|
| VAR d20, seed 1 (Phase C) | 26096765 | `outputs/p4/full/var/seed20261006/26096765` | 300a7a5 |
| DiT, seed 1 (Phase C) | 26096768 | `outputs/p4/full/dit/seed20261006/26096768` | 300a7a5 |
| VAR d20, seed 2 (Phase E) | 26096770 | `outputs/p4/full/var/seed20261070/26096770` | 300a7a5 |
| DiT, seed 2 (Phase E) | 26096771 | `outputs/p4/full/dit/seed20261070/26096771` | 300a7a5 |
| VAR d24 (Phase E) | 26099485 | `outputs/p4/full/var/d24/26099485` | 300a7a5 |
| VAR d20 control pool | 26110495 | `outputs/p4/pool/var/var_d20/26110495` | c4b7176 |
| VAR d24 control pool | 26124035 | `outputs/p4/pool/var/var_d24/26124035` (on `/scratch/users/ntu/kushhema/p4/pool_var_d24` through a symlink; move to home after the freeze) | c4b7176 |
| VAR {7,8,9} confirmation, d20 | 26186463 | `outputs/p4/confirm/var/s7+8+9/26186463` | 82cd939 |
| VAR {7,8,9} confirmation, d24 | 26186465 | `outputs/p4/confirm/var/s7+8+9_d24/26186465` | 82cd939 |

**Failed or discarded jobs.** None of these was used for any result.

| Job | What happened |
|---|---|
| 26096741, 26096748 | Duplicate VAR seed-1 submissions. Killed (exit 143); roots deleted. |
| 26099090 | First VAR d24 sweep. `checkpoints/var/var_d24.pth` had not been downloaded. Re-run as 26099485. |
| 26110497 | First d24 pool. Home disk quota ran out at 106/111 runs while both pools held raw tensors. Root deleted. |
| 26120431 | Second d24 pool. The scratch symlink pointed to a path without `ntu/`, so the job could not create its root and ran nothing. |

Gate B (22 Oct, before this plan): smoke 25915819, pilots 25915820 and 25915821, results in
`lanes/p4/results/gate_b/`.

## Snapshot of the other lanes

The snapshot was taken on 10 Nov at `outputs/p4/handoff/2026-11-10/`. Its record is
`lanes/p4/results/handoff_2026-11-10_SOURCES.json`, and all nine files were found:

| Lane | Ref | Commit | Files |
|---|---|---|---|
| P1 | `origin/main` | 14994c7 | `lanes/p1/results/results_sweep.csv`, `ranking.csv` |
| P2 | `origin/main` | 14994c7 | `lanes/p2/results/phasec/p2_summary.csv`, `recovery_ranking.csv`, `p2_curves.csv` |
| P3 | `origin/p3/dit-and-cross-lane` | c03182e | `lanes/p3/results/{var,dit}_{curves,semantic}.csv` |

## Result files

In git under `lanes/p4/results/` (sha256). They are byte-identical to the NSCC copies in
`outputs/p4/summary/2026-11-03/`.

```
a761c53318eabb56851c0962a4e66f86917e636af62a2d5233486d7883ac03c3  dit_seed1_summary.csv
6fdeb1a4c6d545443b740a7ed83b85d6469161069213b220b6a23d7991d65d3b  dit_seed2_summary.csv
9845b09d26e5afc566d8f65035d19107de202130c21855b92ba6369aee2c3500  var_seed1_summary.csv
a1ad8caf086f025343f4d65450e19b35e4385729931d0efcdd5269766a1ea4fd  var_seed2_summary.csv
9e556f77043d2730f2a635eddd804d5dfc00c5f21fa35b64d88d5ff46d37a5d6  var_d24_summary.csv
294b745c7ad622475b4b3051ba39f7a62e3f259cc639ada88f9e21c319d4e2e5  var_seed1_pooled_summary.csv
294b745c7ad622475b4b3051ba39f7a62e3f259cc639ada88f9e21c319d4e2e5  var_seed2_pooled_summary.csv
d4657f66c40b58bf070d7620fe0ec5a9327a2c928c78b7500cf6287173da06cf  var_d24_pooled_summary.csv
87732ec73ed79cd527af69eaf4a0745141b2f1f7a8a5885aebaa65495ad5246e  var_s789_pooled_summary.csv
7daeb56f5e762e049a17647e89792cbb919962a8bdb675234b058e6ed772bc0b  var_s789_d24_pooled_summary.csv
38f71fe335f123e92a2a0dabf1320bc523711976528f697a84f5a0a1d8d78ae9  robustness_dit.csv
28d920855c24250a3a3bd147dbedd7fd7f392b14cacd79577721ddf7242fa1a6  robustness_var.csv
c5bf4a67f0af40442b83cf6e615a3c244d176d113ab57f7956cbb627482f0dcb  robustness_var_seeds_pooled.csv
5b99ad1c42fc3cf56cd93dd27004dfb591870cedc8b950c3609eb03b84ce171c  robustness_var_pooled.csv
db9bf3689bd70828b0487f0862abafe863ad09d151c799a4dc9075997c2a3884  robustness_var_s789_pooled.csv
5ecc5155d80a5bbb509b313a571f80f5941483ddd312ec436862224ecd82b5a7  shortlist_var.json
a28b0a75377abf6132db6ab5d6e831271e8f238420c1cf58cc02443b553f76a1  shortlist_dit.json
245e96da621db58194be4b7c6d2380227934fcb0fea3df24e4622aa927ded256  handoff_2026-11-10_SOURCES.json
46fdd38c1a51cbfb7403d8364294a005dd2b64fcbde8620628bc3f9f9dfac873  figure4_incep_cos_pooled.png
36add4c533dcfb0dce1be48dd17a073a990c364762ae5d3b741219695b96f3ec  figure4_cls_prob_pooled.png
b38ef2bfa16eb9d67022453739e6bcfa29d183c1a43a4e68def57f46daaf1ccb  figure5_var.png
916a69213bf40504d2ced9dc505d759f288f4000fccf30389ada4f46616d6c51  figure5_dit.png
```

The two pooled VAR seed summaries are byte-identical. That is expected: protect and damage do not
depend on the seed, and the pooled control has none.

Not in git (too large), in `outputs/p4/summary/2026-11-03/` (sha256):

```
2c7d239a88f4bb30b19a4f34dfe8511c5b69cd7c9154d65dea0e2feeac0f940c  var_seed1_tidy.csv
d3e70379dc7b7f5a5b5c4b1c927cf36903447e77f1389102e0c34225735066b9  var_seed2_tidy.csv
efe6395fa20e0aabb710c999a74ef769db53e9d283d68baaf22aa70d8b3ccd3a  var_d24_tidy.csv
3508a90f0c99a6eaa75e16167c58c08a8a7d80adfe84ced756ff2b13a804aad1  dit_seed1_tidy.csv
ee014e461daf9d544769a08dee712b1408aae4e3c832b4b768173ed7b125d54d  dit_seed2_tidy.csv
0a7077cf909e125400f66921b63f2c2dd5fc6617494d400874144ebc62e44d10  var_seed1_pooled_tidy.csv
8ffede362785ae5e9c75f1858383faab915583980c3004f4984d4373a7f7e0b7  var_d24_pooled_tidy.csv
62982d36ff006ea533b3fc76efa8e88ae40cd0e22fe75df63c7d91fc8ae73fbc  var_s789_pooled_tidy.csv
8eb45cfda2f2a81bed960b9615416b8266345ccd09e72524306cb317c8ad09d3  var_s789_d24_pooled_tidy.csv
```

Superseded and not used in the report: the single-random-control Figure 4 drafts of 3 Nov,
`figure4_incep_cos.png` (ec717a93…eb55) and `figure4_cls_prob.png` (876b0eee…49e).

## Departures from the plan

1. **Freeze date.** R16 (5 Nov) and R18 (6 Nov) moved to 12 Nov, for the reasons at the top.
2. **Pooled VAR control (3 Nov, c4b7176).** The seed-2 repeat showed that one random VAR control
   placement flipped most contrasts (21 of 36 signs agreed). The VAR control became the per-image
   mean over every placement. This was decided before the d24 pool and the confirmation runs were
   read. DiT keeps its seeded control. See `robustness.md`, section 1.
3. **P3 read from P3's branch.** P3's tables were never merged to main, so the snapshot reads them
   from `origin/p3/dit-and-cross-lane` (c03182e, delivered 2 Nov, before P3's 3 Nov deadline). P1
   is read from `origin/main` after its PR #7 merged; the files are identical to P1's branch. The
   shortlist rule itself is unchanged.
4. **Agreement p-values (82cd939).** P3's `spearman()` falls back to a normal approximation without
   scipy, and NSCC has no scipy. P4 now computes p from the t distribution, as scipy does. The
   p-values never select anything, so the shortlist verdicts did not change.
5. **R17 seed-2 repeat of the new candidate set** was not run for VAR. It would change only the
   control, and the pooled control has no seed. The d24 repeat was run.
6. **Confirmation runs after the planned freeze.** They were submitted on 10 Nov and completed on
   11 Nov.
7. **R7 (P1's position on the narrowed protect arm).** The question was not sent before the freeze.
   It is left open for P1 at PR review.

## After the freeze

- Move the d24 pool root from scratch to home. Scratch purges old files.
- Open the P4 PR from `kmaniar/p4-phase-c-e`. It replaces PR #10, and its diff contains only
  `lanes/p4/`. P1 merges it.
- Deferred by the plan: delete `lanes/p4/p3_fidelity.py` once P3 fixes the `stage=""` mismatch.
