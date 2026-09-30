# Hook interface

Status: implemented in Phase 3 (`runner/hooks.py`, `runner/var_model.py`, `runner/dit_model.py`,
`runner/generate.py --hook / --skip-timesteps`); tests in `scripts/phase3/check_hooks.py`, GPU job
`jobs/phase3_check.pbs`, how-to in `docs/hooks_quickstart.md`. Line numbers refer to the pinned
submodules (`third_party/VAR` @ `78b9539`, `third_party/DiT` @ `ed81ce2`) and to `runner/` as of Phase 2
(the proposal's baseline); where the implementation differs from the proposal, it is noted inline.

A hook is a function the runner calls at a fixed point of every sampling stage, per batch, so an
experiment can read or change the model's intermediate state. Every hook is registered for one side
of a stage: `before` or `after`. Two rules drive this design:

- With zero hooks registered, the output must be bit-identical to what the runner produces today.
- A hook must never change how many random draws an image takes from its own generator. A baseline
  run and an intervened run then stay paired: they share the same random numbers at every stage.

All verification described here runs on CPU. Claims that also need a GPU run are marked
**[GPU-unverified]** where they appear and collected in §12.

---

## 1. Where hooks fire

A **stage** is one scale for VAR and one timestep for DiT. Each stage has two hook points:
`before` and `after`.

### VAR: around `get_next_autoregressive_input`, once per scale

One iteration of the scale loop in `VAR.autoregressive_infer_cfg` (`third_party/VAR/models/var.py:160-187`):

| Line | What happens |
|---|---|
| 164-168 | Transformer forward on `next_token_map` (KV cache on) |
| 168-173 | `logits_BlV` from `get_logits`, then CFG mixing |
| 175 | `idx_Bl = sample_with_top_k_top_p_(...)`: **random draws** (runner's per-row wrapper) |
| 176-180 | `h_BChw`: codebook embedding of `idx_Bl`, or, if `more_smooth`, a Gumbel-softmax mix (**more random draws**) |
| 182 | `h_BChw` reshaped to `(B, Cvae, pn, pn)` |
| **183** | **`f_hat, next_token_map = vae_quant_proxy[0].get_next_autoregressive_input(si, SN, f_hat, h_BChw)`** |
| 184-187 | `next_token_map` → `word_embed` + level/position embedding, doubled for CFG |

Both hook points sit inside one runner wrapper around the call on line 183:

- **`before`:** just before the upstream call. This scale's tokens are sampled and embedded but not
  yet added to the accumulator (`f_hat.add_(h)` at `models/quant.py:191` / `:195`).
- **`after`:** just after the upstream call returns. `f_hat` now includes this scale, and
  `next_token_map` (pre-embedding) for the next scale has been computed from it.

For VAR, the stage boundary is therefore the **accumulation** of scale si, not the transformer
forward. The transformer forward and all random draws for scale si happen before `before(si)`. This is
the only per-scale call in the loop that receives both this scale's contribution and the running
accumulator, so the runner can intercept it without copying the loop. Each hook fires exactly
`SN = len(patch_nums)` times per batch (10 for VAR-d20), including the last scale, where upstream
returns `(f_hat, f_hat)`. A hook point before the transformer forward is not proposed; it would only
add something new at scale 0 (see Open questions).

### DiT: around one iteration of the runner's step loop

The DiT step loop is already runner code (`runner/dit_model.py:52-71`, `DiTModel._p_sample_loop`):

| Line | What happens |
|---|---|
| 65 | `t` for this step |
| 66 | `out = diffusion.p_mean_variance(sample_fn, img, t, ...)`: model forward, CFG, posterior mean/variance |
| 67-68 | per-row noise: **random draws** |
| 69-70 | `img = out["mean"] + nonzero_mask * exp(0.5 * log_variance) * noise` (x_t → x_{t-1}) |

- **`before`:** at the top of the iteration, before line 66. The state is `img` = x_t, the latent this
  step starts from.
- **`after`:** right after line 70. The state is `img` = x_{t-1}. The pre-step latent x_t is also
  available (read-only), so an `after` hook can put it back (§11, "hold").

Each hook fires exactly `num_sampling_steps` times per batch (250 for the FID config). At the last step
(t = 0), the `after` state is the final latent, just before VAE decoding.

Note the asymmetry: DiT's `before` point comes before that step's model call and random draws, while
VAR's comes after them. This does not affect pairing (§7), because no hook can change a draw's size and
draws never read the state.

---

## 2. What "state" is at each hook point, and tradeoffs

### VAR

Candidates available at, or reachable from, the hook points:

| Candidate | Shape | Pros | Cons |
|---|---|---|---|
| `logits_BlV` (post-CFG) | `(B, pn², V)` | Changes the distribution but the draw stays random | Only visible inside the sampler wrapper, before both hook points; a different kind of intervention |
| `idx_Bl` token ids | `(B, pn²)` long | Discrete, interpretable, cheap to log | "Zero" has no meaning (token 0 is a real codebook entry). With `more_smooth=True` it is sampled but **not** used for `h_BChw` |
| `h_BChw` this scale's embedding | `(B, Cvae, pn, pn)` | Exactly what this scale contributes; continuous | Passes through `Phi` (a conv with bias), so zeroing it does not remove the contribution (§10) |
| `f_hat` accumulator | `(B, Cvae, 16, 16)` for d20 | What the decoder finally decodes; carries all earlier scales | Changing it affects every later scale and the image |
| `next_token_map` | `(2B, pn_{k+1}², C)` | Exact transformer input for the next scale | CFG-doubled and embedded; derived from `f_hat` anyway |
| KV cache `blocks[i].attn.cached_k/v` | `(2B, heads, L_so_far, head_dim)` | Transformer memory of earlier scales' inputs | 2 tensors × depth blocks; changing its length is a structural change |

**Proposal (`VARScaleState`):**

| Field | `before(si)` | `after(si)` |
|---|---|---|
| `h_BChw` | modifiable | read-only (what was added) |
| `f_hat` | modifiable: accumulator before this scale is added | modifiable: accumulator after this scale is added |
| `f_hat_before` | — | read-only: `f_hat` as passed to the upstream call (after any `before` hooks) |
| `idx_Bl` | read-only (recorded by the sampler wrapper for this scale) | read-only |
| metadata | `rows` (`(class_id, seed)` per batch row) + stage fields (§3) | same |

If an `after` hook changes `f_hat`, the runner recomputes the next scale's pre-embedding input as
`F.interpolate(f_hat, size=(pn_next, pn_next), mode="area")`, replicating `quant.py:192`
(`(f_hat, f_hat)` at the last scale). Upstream then embeds it as usual (lines 185-187). Logits and the
KV cache are out of scope for now (Open questions).

### DiT

| Candidate | Pros | Cons |
|---|---|---|
| `img` = the latent | The only tensor carried between steps; changing it changes everything that follows | Noisy at early steps, so hard to interpret directly |
| `out["pred_xstart"]` = predicted x_0 | Interpretable ("what the model thinks the image is") | Derived; changing it would require recomputing the posterior mean (`q_posterior_mean_variance`) |
| model output eps / learned variance | Direct model output | Not returned by `p_mean_variance`; would need a wrapper around `sample_fn` |
| `noise` | — | Changing it breaks the pairing on purpose; not a state |

**Proposal (`DiTStepState`):**

| Field | `before(j)` | `after(j)` |
|---|---|---|
| `x` | modifiable: x_t (conditional half `img[:n]`) | modifiable: x_{t-1} (conditional half) |
| `x_before` | — | read-only: x_t as this step's model call received it (after any `before` hooks) |
| `pred_xstart` | — | read-only: `out["pred_xstart"][:n]` |
| metadata | `rows` + stage fields (§3) | same |

With CFG, `img` has `2n` rows. Rows `n..2n-1` are the unconditional copies. `forward_with_cfg`
(`third_party/DiT/models.py`) only ever reads `x[:len(x)//2]`, so those copies never influence the
output. Hooks see and replace only `img[:n]`, and the runner leaves `img[n:]` as upstream computed it.

---

## 3. Hook signature and per-stage fields

```python
Hook = Callable[[str, int, float, str, State], Optional[State]]

def hook(model: str, stage: int, p: float, when: str, state: State) -> State | None: ...
```

| Argument | VAR | DiT |
|---|---|---|
| `model` | `"var"` | `"dit"` |
| `stage` (native) | `si`, the scale index `0 .. SN-1` | `timestep_in`, the original diffusion timestep this step's model call receives (`999 … 0`) |
| `p` in [0, 1] | `si / (SN - 1)`. This is upstream's own `ratio` (`var.py:161`), which also drives the CFG ramp and `Phi` choice | `j / (S - 1)`, where `j = 0 .. S-1` counts steps in the **baseline** schedule (first step p = 0, last step p = 1) |
| `when` | `"before"` or `"after"`, the side of the stage this call is for | same |
| `state` | `VARScaleState` | `DiTStepState` |

- For a single-stage run (`SN == 1` or `S == 1`), `p = 1.0`.
- `stage` and `p` are the same for the `before` and `after` calls of one stage.
- `when` is passed as well as registered, so one function can serve both sides.
- Registration: `Observe(fn, when="after")` / `Modify(fn, when="before")`. `when` is required (no default).
  Optional (added in implementation): `stages=[...]` restricts the hook to those native stage ids
  (`si` for VAR, `timestep_in` for DiT; default `None` = every stage) and `name=` labels it (default:
  the function's name). Before any sampling, the runner raises if a registered stage does not exist in
  the run, or (DiT) is a skipped timestep (§11b), so a hook can never silently not fire.
- Hooks are passed per call, `model.sample(class_ids, seeds, hooks=())`. The runner does not keep
  them as global state. From the command line: `runner.generate --hook FILE.py:NAME` (repeatable).

**Per-stage fields.** These are read-only and identical for every row. They are included in every
`state` and written once per run to `run.json` as a `stages` table (independent of hooks; §5 explains
why this does not affect outputs).

| Field | Model | Value |
|---|---|---|
| `pn` | VAR | `patch_nums[si]` |
| `n_tokens` | VAR | `pn²`, the tokens sampled at this scale |
| `cum_tokens` | VAR | `sum(pn_i² for i ≤ si)`, tokens sampled through this scale |
| `total_tokens` | VAR | `L = sum(pn²)` (680 for every 256px VAR depth, d16 to d30, which all share `patch_nums` and so have 10 scales; 30 for the tiny test model) |
| `timestep_in` | DiT | original timestep of the step's input, `diffusion.timestep_map[i]` |
| `timestep_out` | DiT | original timestep of the step's output (next step's `timestep_in`); `None` for the last step, whose output is x_0 |
| `alpha_bar_in` | DiT | ᾱ of the input latent: `diffusion.alphas_cumprod[i]` (`gaussian_diffusion.py:175`) |
| `alpha_bar_out` | DiT | ᾱ of the output latent: `diffusion.alphas_cumprod_prev[i]` (`:176`); 1.0 at the last step |
| `sigma_in`, `sigma_out` | DiT | noise level, `sqrt(1 - alpha_bar)`: the std of the noise term in `x = sqrt(ᾱ)·x_0 + sqrt(1-ᾱ)·ε` |

Here `i` is the respaced index the runner loop uses (`S-1 … 0`). On `SpacedDiffusion`, these arrays
are already the respaced schedule, so `alphas_cumprod[i]` is the base schedule's ᾱ at `timestep_map[i]`.

`p` is monotonic, starts at 0 and ends at 1, so "intervene at 30% of generation" means the same thing
to code written for either model. It is not the only sensible axis:
- VAR's `p` is uniform in scales, not tokens (the last scale alone is 256 of 680 tokens at 256px);
  `cum_tokens / total_tokens` is the token axis.
- DiT's `p` is uniform in steps, not noise; `sigma_in` is the noise axis.

Both alternatives are available through the fields above (Open questions 1-2).

## 4. Observe hooks vs modify hooks

A hook is registered as either kind, on either side: `Observe(fn, when=...)` or `Modify(fn, when=...)`.

- **Observe:** gets clones of every state tensor and must return `None`; returning anything else is an
  error. Cloning makes read-only structural, not a convention. Upstream mutates `f_hat` in place
  (`quant.py:191`), so a stored reference would otherwise change after the hook returns.
- **Modify:** gets clones and must return a state. The runner checks that every modifiable tensor
  has the same shape, dtype and device as the one it replaces, and raises otherwise. Changes to
  read-only fields are rejected (the runner compares them against its own copy). Returning the input
  unchanged is allowed and must be a bit-exact no-op (§6, test 1).
- **Order at one hook point** (one `when` of one stage):
  1. Modify hooks run in registration order, each receiving the previous one's output.
  2. The runner writes the final state back into the loop.
  3. Observe hooks see that final state, i.e. what the model actually continues from.
- **Order within a stage:** all `before` hooks → the stage's computation → all `after` hooks.
  - DiT: `after(j).x_before` is the latent after the `before(j)` modifications.
  - VAR: `after(si).f_hat_before` is the accumulator after the `before(si)` modifications.

  So "restore the pre-step state" always restores what the step actually started from.
- Neither kind is given the per-row generators; see §7.

## 5. Zero hooks give output identical to today

1. **No hooks, no new code path.**
   - VAR: the quantizer wrapper (§8) is installed only when `hooks` is non-empty. Otherwise the
     swapped objects are exactly today's two sampler wrappers.
   - DiT: with no hooks and no `skip_timesteps`, `sample()` runs the unchanged `_p_sample_loop`.
     Hooks and skips use a separate loop, `_p_sample_loop_hooked`, with the same per-step operations;
     no clone, `cat` or write-back happens unless a hook is registered at that point.
   - Computing the per-stage fields (§3) reads only Python ints and numpy schedule arrays. It never
     touches a tensor in the sampling path, so it cannot change outputs, and it runs with or without hooks.
2. **Default argument.** `hooks=()` is the default everywhere, so existing callers (`runner.generate`,
   `scripts/phase1/timing.py`) are unchanged.
3. **Pinned by the existing test.** `scripts/phase2/check_seeding.py` asserts runner-at-batch-1 ==
   plain upstream at batch 1 and repeat-run equality. Both must keep passing with no hooks, which ties
   zero-hook output to upstream, not just to the previous runner version.
4. **Recorded.** `run.json` (and the single-image `.json`) gets `"hooks"` (name, kind, when, stages of
   each hook; `[]` without hooks), `"skip_timesteps"`, the `stages` table (executed steps only), and each
   row's `generator_state_sha256` after its batch (read after sampling). Image and tensor outputs are unchanged.

## 6. Tests

Implemented in `scripts/phase3/check_hooks.py`. On CPU it uses tiny random-weight models built as in
`scripts/phase2/check_seeding.py` but with the real stage structure (VAR-d20's 10 scales, DiT's 250
steps), all four configs (VAR, VAR `more_smooth`, DiT with CFG, DiT without CFG); on the GPU
(`jobs/phase3_check.pbs`) it uses the real weights and configs. **Every comparison is at the same batch
size**, 16: across batch sizes the outputs differ by ~1e-5 from float rounding (CPU-measured), which
would mask or fake a hook effect. The only batch-1 check is the regression hash of class 207 seed 0,
in the GPU job. (The proposal said batch 1 and 4; changed to 16, the experiment batch size.)

1. **No-op hooks.** Run the manifest with:
   - no hooks;
   - a `Modify` hook that returns its input unchanged, registered at `before`;
   - the same, registered at `after`;
   - one `Observe` hook at each side.

   Assert identical tensor sha256 for every row across all four runs. This covers every hook code path:
   - VAR's quantizer wrapper, including the `after`-path recomputation of `area(f_hat)`;
   - clone and write-back;
   - DiT's `cat` of the modified half with `img[n:]` on both sides.
2. **Firing count and fields.** One observer per side records `(model, stage, p, when, fields, shapes)`.
   For N rows at batch size b, assert:
   - each observer fires exactly `ceil(N/b) × SN` times (VAR) or `ceil(N/b) × S` times (DiT);
   - within a stage, all `before` calls come before all `after` calls, with equal `stage` and `p`;
   - `stage` follows the expected sequence; `p` goes from 0 to 1, strictly increasing;
   - VAR: `n_tokens == pn²`, `cum_tokens` strictly increasing, final `cum_tokens == total_tokens`
     (30 for the tiny model);
   - DiT: `alpha_bar_in` strictly increasing across steps; `alpha_bar_out(j) == alpha_bar_in(j+1)`;
     final `alpha_bar_out == 1.0` and `timestep_out is None`;
   - the fields equal the `stages` table in `run.json`;
   - shapes match, including the last partial batch.
3. **Destructive hook.** A `Modify` hook wipes out row 0 at stage k. For VAR, at `after(k)`:
   `f_hat = f_hat_before`, i.e. drop scale k (§10b). For DiT, at `after(j)`: `x[0] = 0`. Assert:
   - row 0's output differs from baseline;
   - rows 1..b-1 are bit-identical to baseline (the intervention stays inside its row;
     **[GPU-unverified]**, see §12);
   - **pairing:** after the batch, every row's generator state (`Generator.get_state()`) equals the
     baseline run's;
   - VAR: an observer's `idx_Bl` per scale is identical to baseline for scales `0..k`; later scales
     are not required to match.
4. **Before/after round trip.**
   - DiT: `before(j)` zeroes `x[0]`; an `after(j)` observer checks that `x_before[0]` is zero, so
     `x_before` reflects the `before` modifications.
   - VAR: likewise with `f_hat` at `before(k)` / `f_hat_before` at `after(k)`, plus a check that
     `before(k+1).f_hat` equals `after(k).f_hat`.
5. **DiT hold** (§11a): an `after(j)` modify hook returns `x = x_before` for all rows. Assert:
   - generator states equal baseline's;
   - an observer's `before` latents are bit-identical to baseline for steps `≤ j`;
   - the output differs from baseline;
   - the observer fires S times.
6. **DiT skip** (§11b), skipping step k with `0 < k ≤ S-1`. Assert:
   - generator states equal baseline's (burned draw);
   - `before` latents bit-identical to baseline for steps `< k`;
   - observers fire `S-1` times, and no call has `timestep_in == τ_k`;
   - the coefficients each executed step actually uses are bit-identical to baseline's at every step
     except the merged one (`k-1`), which differs. (The proposal asserted this of the merged schedule
     object's own arrays; that is false, see §11b, so the runner uses the baseline object for every
     non-merged step and the test checks what is used.)

   Negative control: a test-only flag disables the burn. Assert that the generator states then
   **differ** from baseline, which demonstrates why the burn is needed.

## 7. Hooks must not change the number of random draws

Each row's generator is advanced only by:
- VAR: the per-row sampler wrapper, one `multinomial` per scale (`helpers.py:19`) plus one
  `exponential_` per scale if `more_smooth` (`helpers.py:26`);
- DiT: one `randn((k, C, H, W))` per step, k = 2 with CFG, else 1 (`runner/dit_model.py:67`) and the
  initial `z` draw.

Each of those draws has a size fixed by `patch_nums`, `V`, `num_sampling_steps` and the latent shape.
None of these depends on the state values, so a value-only intervention keeps every row's draw
sequence, and baseline and intervened runs consume the same random numbers at every later stage. No
hook, `before` or `after`, can reach a draw: DiT draws do not read the state, and VAR draws for a scale
finish before `before(si)`. The §4 shape/dtype check and the generator-state comparisons in tests 3,
5 and 6 enforce this.

Where it could break:

- **Removing tokens** (making a scale's token map shorter) changes `pn²` at the next `multinomial`, so
  that row consumes a different number of draws and every later draw shifts. On CUDA, Philox offsets
  advance by an amount that depends on tensor size, so this holds there too **[GPU-unverified]**.
  Tokens must be masked or replaced, never dropped. The proposed VAR hook points cannot shorten the
  next scale anyway: the next input is built at the fixed size `patch_nums[si+1]`, and the shape check
  rejects a resized `f_hat`.
- **Skipping a stage or stopping early** skips that stage's draws. DiT "skip a timestep" (§11b)
  therefore burns the skipped step's draw. Skipping a VAR scale's transformer forward is not proposed.
- **Changing the schedule mid-run**: `num_sampling_steps`, resolution, `patch_nums`. The one
  exception is §11b, which keeps the draw count by burning.
- **Changing batch composition inside a hook** (dropping a row): the shape check forbids it.
- **A hook using a row generator for its own randomness.** Hooks never receive `gens`. A hook that
  needs randomness uses its own `torch.Generator`. The global RNG is safe to use: neither VAR (labels
  are always given) nor the DiT runner loop reads it.
- **Value-dependent draw counts.** If `torch.multinomial` ever consumed a number of draws that
  depended on the probabilities (e.g. after top-k/top-p puts `-inf` in different places), any
  intervention that changes later logits would desync. The count is believed to depend on tensor
  size only. Tests 3/5 check this on CPU; on CUDA it is **[GPU-unverified]**.
- **Changing logits before sampling** (not proposed) would keep the draw count but, by design, change
  which token those same draws select.

## 8. Coexistence with the runner's existing overrides

**VAR.** Today `VARModel._per_row_rng` (`runner/var_model.py:79`) temporarily replaces
`models.var.sample_with_top_k_top_p_` and `models.var.gumbel_softmax_with_rng` (module namespace, in
memory only) with per-row wrappers. Hooks add a third scoped override, installed in the same context
manager only when `hooks` is non-empty: an instance attribute `get_next_autoregressive_input` on the
quantizer object `self.vae.quantize` (the object `var.vae_quant_proxy[0]` points to). It shadows the
class method for that object only and is removed with `del` on exit.

Per scale, the order in the loop is:
1. The sampler wrapper runs (all random draws for this scale). It also stores the returned
   `idx_Bl` in a per-call slot so the hook state can expose it.
2. If `more_smooth`, the Gumbel wrapper runs (more draws).
3. Upstream embeds and reshapes `h_BChw`.
4. The quantizer wrapper runs:
   1. Build `VARScaleState` and run the `before` hooks.
   2. If there are `after` hooks, clone `f_hat` as `f_hat_before`. The clone is needed because
      upstream adds in place.
   3. Call the original `get_next_autoregressive_input` with the (possibly replaced) `f_hat` and `h_BChw`.
   4. Run the `after` hooks.
   5. If an `after` modify hook ran, return `(f_hat, area(f_hat))` recomputed from the final `f_hat`
      (§2). Otherwise return upstream's result unchanged.

So the sampler wrappers own the randomness, the quantizer wrapper owns the state, and nothing a hook
returns can reach a generator. All three overrides are restored in a `finally`. The load-time check
that the swapped helpers are the ones `var.py` imported gains a matching check that
`quantize.get_next_autoregressive_input` is the upstream `VectorQuantizer2` method.

**DiT.** No override is needed: the step loop is runner code.
- The `before` call goes above line 66: build the state from `img[:n]`, run the hooks, write back
  `img = torch.cat([x_new, img[n:]])`.
- The `after` call goes below line 70: `x_before` is the conditional half as it entered line 66, plus
  `pred_xstart[:n]`. Write back the same way.
- The per-row noise draw on line 67 does not read the state, so it is unaffected by either.
- Skip (§11b) is a schedule option of the same loop, not a hook (§11).

## 9. Variables that carry state between stages

**VAR** (`autoregressive_infer_cfg`), carried from scale `si` to `si+1`:

| Variable | Shape | Role |
|---|---|---|
| `f_hat` | `(B, Cvae, P, P)`, P = `patch_nums[-1]` (16) | Running sum of every scale's `Phi(upsample(h_BChw))`. Allocated at `var.py:157`, updated in place at `quant.py:191`/`:195`, returned and reassigned at `var.py:183`. The only input to the decoder (`fhat_to_img`, `var.py:190`) |
| `next_token_map` | `(2B, pn_{si+1}², C)` | Transformer input for the next scale: `area`-downsampled `f_hat` (`quant.py:192`) → `word_embed` + `lvl_pos` slice (`var.py:185-186`) → `.repeat(2,1,1)` for CFG (`var.py:187`). Initialised from `sos` (`var.py:154`) |
| `blocks[i].attn.cached_k`, `cached_v` | `(2B, heads, L_so_far, head_dim)` each, per block | KV cache, enabled at `var.py:159`, extended by `torch.cat` in `basic_var.py:108-109`. Holds keys/values of every earlier scale's **input** tokens (not of its sampled tokens) |
| `cur_L` | int | Token offset into `lvl_pos` (`var.py:163`); equals `cum_tokens` of the current scale |
| per-row `gens` (runner) | — | Generator state, advanced by the sampler wrappers only |

Constant across scales: `label_B`, `sos`/`cond_BD` (`var.py:151`, `2B` rows: conditional + null
class), `lvl_pos` (`var.py:153`). Recomputed each scale and not carried: `ratio`, `cond_BD_or_gss`,
`x`, `logits_BlV`, `t`, `idx_Bl`, `h_BChw`. Upstream's `self.rng` is not used (the runner passes `g_seed=None`).

A subtlety that matters for §10: the scale-k tokens enter the transformer only through
`next_token_map` for scale k+1 onward, which is built from `f_hat`. The KV cache at scale k holds scale
k's *input*, which came from scales `< k`.

**DiT** (`runner/dit_model.py:_p_sample_loop`), carried from step to step:

| Variable | Shape | Role |
|---|---|---|
| `img` | `(2n, 4, H/8, W/8)` with CFG, else `(n, …)` | x_t; the only tensor carried between steps. `img[n:]` (unconditional copies) is never read by the model |
| per-row `gens` | — | Generator state, advanced once per step per row |

Constant across steps: `model_kwargs` (`y`, `cfg_scale`) and the schedule arrays of `self.diffusion`.
Recomputed each step: `t`, `out` (`mean`, `variance`, `log_variance`, `pred_xstart`), `noise`, `nonzero_mask`.

## 10. VAR: two ways to "remove a scale"

Let k be the scale to remove, `h_k = h_BChw` at scale k, and `Phi_k = quant_resi[k/(SN-1)]`.
Upstream does `f_hat += Phi_k(upsample(h_k))`, with no upsample at the last scale.

**(a) Zero the tokens sampled at scale k:** a `before(k)` modify hook returns `h_BChw = zeros_like(h_BChw)`,
and upstream continues unchanged.
- `upsample(0) = 0`, but `Phi_k(h) = (1 - r)·h + r·conv(h)` and the conv has a bias.
- So `f_hat += r · bias_k`, a per-channel constant over the whole map (r = `quant_resi` = 0.5 for d20).
- This is "scale k contributed the zero embedding", not "scale k contributed nothing".
- Replacing token ids with 0 is different again: 0 is a real codebook entry.

**(b) Remove scale k's contribution to the next scale's input:** an `after(k)` modify hook returns
`f_hat = f_hat_before`, restoring the pre-stage accumulator. This is the VAR counterpart of DiT's hold
(§11a).
- The next scale's input (`area(f_hat)` → `next_token_map`) and every later `f_hat` then contain
  nothing from scale k.
- Scale k's tokens reached the transformer only through `next_token_map` (§9), so this removes them
  from all later transformer inputs and from the decoded image.
- What remains of scale k: the transformer still ran at scale k, and its input is in the KV cache
  (that input came from scales < k).
- Its draws were consumed, which keeps the pairing.

**The proposed hook points support both:** (a) at `before`, running pure upstream code afterwards;
(b) at `after`, with the runner recomputing one upstream line (`quant.py:192`). A third variant (also
removing scale k's entries from the KV cache) would change the cached length; see Open questions.

## 11. DiT: "hold latent for one step" vs "skip a timestep"

Notation: the baseline visits timesteps `τ_0 > τ_1 > … > τ_{S-1}` (original scale; `τ_{S-1} = 0`).
Step j evaluates the model at `τ_j` and maps a latent at level `τ_j` to one at level `τ_{j+1}` (x_0 for
the last step).

### (a) Hold latent for one step

An `after(j)` modify hook returns `x = x_before`, so the latent leaving step j is the one that entered it.

- **Mechanism:** a hook. The schedule, the model calls and the coefficients are unchanged.
- **Draw count:** unchanged. Step j's noise is drawn on line 67 and used to compute x_{t-1}, which is
  then discarded. Every row still makes exactly S step draws.
- **Pairing:** exact. Steps `> j` receive the same noise as baseline; steps `≤ j` are bit-identical
  to baseline.
- **What it tests about necessity:** whether step j's update (its denoising move plus its injected
  noise) is necessary when the rest of the sampler runs as planned.
  - The confound: the held latent is at level `τ_j` but step j+1 is told `τ_{j+1}`. From there on the
    model sees a latent noisier than the schedule assumes and may or may not correct for it.
  - So a large effect means "the update at j mattered, *or* the mismatch hurt". A small effect means
    the trajectory absorbed losing it.
  - Holding also removes that step's injected noise, so the run is slightly less stochastic than baseline.

### (b) Skip a timestep

The model is not evaluated at `τ_k` (`0 < k ≤ S-1`). Step k-1 maps level `τ_{k-1}` directly to level
`τ_{k+1}` using the coefficients of a schedule without `τ_k`. Step k does not run.

- **Mechanism:** not a hook, a runner schedule option (`skip_timesteps=[τ_k]`). It changes which model
  calls happen and the transition coefficients, which a state modification cannot express.
  - The runner builds a second `SpacedDiffusion` with `use_timesteps = baseline − {τ_k}` (respacing
    as in `respace.py:73-86`).
  - It iterates the baseline schedule, using the second object's coefficients for the merged step,
    and at step k only draws noise.
  - `respace.py` derives each beta from consecutive kept ᾱ values, so mathematically only the merged
    transition's coefficients change. In float64 they do not all stay the same numbers: `GaussianDiffusion`
    recomputes ᾱ as `np.cumprod(1 - betas)`, and the merged factor changes the rounding of every ᾱ
    above it (1 ULP differences at timesteps above the merged step, for many choices of k; found
    with numpy before implementation). The runner therefore takes only the merged step's
    coefficients from the second object and keeps the baseline object for every other step, so the
    coefficients actually used are bit-identical to baseline's except at the merged step (test 6).
  - Several timesteps may be skipped (`skip_timesteps` is a list). Consecutive skipped timesteps form
    one window, merged into one transition; each window's merged step uses a schedule lacking only
    that window, so windows do not affect each other's coefficients. Every skipped step's draw is burned.
  - Hooks fire only for executed steps, with the baseline `stage`/`p`, so a skip run lines up with
    its baseline, with one gap. The merged step's per-stage fields describe the merged transition
    (`timestep_out = τ_{k+1}`, ᾱ values from the schedule it uses) and carry `merged: true`.
  - From the command line: `runner.generate --skip-timesteps τ_k [...]`.
- **Draw count:** kept at S step draws per row by **burning** step k's draw: it is drawn with the same
  shape and discarded. The merged step uses step k-1's draw, as in baseline. Without the burn a row
  would make S-1 draws and every later step would receive the next step's noise.
- **Pairing:** with the burn, steps `< k-1` and `> k` are paired with baseline (same noise, same
  coefficients). Only the merged transition differs, in its coefficients and in the noise variance it
  injects. Without the burn, everything after the skip is unpaired (test 6's negative control).
- **What it tests about necessity:** whether the model evaluation at `τ_k` is necessary when the
  sampler stays self-consistent (every latent stays at the noise level the next step is told).
  Unlike hold, there is no level mismatch, so an effect is attributable to losing that evaluation and
  the finer step. With 250 steps a single skip is expected to be small; skipping windows is the likely
  experiment (Open questions).
- `k = 0` is excluded: there is no previous step to merge into, and the start latent `z` is at `τ_0`'s level.

**In short:** hold asks whether the *update* at j is needed, with the schedule kept and a level
mismatch introduced. Skip asks whether the *model evaluation* at τ_k is needed, with the schedule
merged and no mismatch. Both keep S draws per row. Hold pairs exactly; skip pairs everywhere except
the merged transition, and only because of the burn.

## 12. GPU-only assumptions

Everything in §6 runs on CPU in fp32 without autocast. The following are assumed for the GPU runs and
are **not verified**. Each needs the listed check on a GPU node before GPU results rely on it.

Since Phase 3, VAR runs in strict fp32 (`configs/var_d20.yaml`: autocast off, TF32 off), so G1 no
longer applies to the configured runs (the dtypes the hooks see are still reported), and G9's fp16
concern is replaced by the fp32 batch 1 vs 16 measurement below.

`jobs/phase3_check.pbs` runs these checks on real weights:
- G2: test 1 at batch 16, and the batch-1 regression with no-op hooks.
- G3: test 3.
- G4 and G6: generator states in tests 3, 5 and 6.
- G7: its baseline run vs the hook suite's baseline.
- G8: DiT via the Phase 1 hash; VAR in fp32 by comparing the runner at batch 1 with plain upstream
  (`scripts/phase2/upstream_ref.py`) on all 16 manifest rows, which must be identical. The VAR fp32
  regression hash is only recorded.
- G9: VAR fp32 at batch 1 vs 16, measured.
- G10 in part: fp32 vs fp16 timing; hook overhead is not timed separately.

Until that job has run, the entries stay unverified.

| # | Assumption | Why CPU says nothing | Check on GPU |
|---|---|---|---|
| G1 | Under VAR's fp16 autocast, `f_hat` and `h_BChw` are float32 (`f_hat` from `sos.new_zeros`, embedding outside autocast), so the dtype check in §4 accepts float32 | CPU runs have no autocast (`autocast_ctx` is a no-op on CPU) | Test 2 with dtype recorded |
| G2 | A no-op hook is bit-exact: `clone`/`cat` write-back does not change which cuBLAS/cuDNN/SDPA kernel runs (same shape and strides; alignment of a fresh tensor vs a slice could in principle matter) | CPU kernel choice differs from CUDA | Test 1 at batch 1 and 16 |
| G3 | Row isolation: changing row 0 leaves other rows bit-identical at a fixed batch shape | Different kernels, possible split reductions | Test 3 |
| G4 | Draw count per row depends on tensor size only, never on values (CUDA `multinomial`, `randn`, `exponential_` Philox offset increments) | CPU generators are a different implementation | Tests 3, 5, 6 (generator states) |
| G5 | Removing tokens would shift later draws on CUDA as on CPU (§7) | Same | Only relevant if token removal is ever proposed |
| G6 | `Generator.get_state()` on a CUDA generator captures seed and offset fully, so equal states mean equal future draws | CPU generator state is a different object | Tests 3, 5, 6 |
| G7 | Repeat runs at the same batch size are bit-identical on GPU with the config's settings (`cudnn_deterministic: true`, fixed SDPA kernel) | Phase 2 repeat check was CPU-only | `check_seeding`-style repeat run on GPU |
| G8 | Runner at batch 1 equals upstream at batch 1 on GPU (per-row generator on CUDA ≡ upstream's seeded generator) | CPU-only so far | Phase 1 job's equivalence step |
| G9 | Cross-batch-size differences stay small enough that "compare only at the same batch size" is the only precaution needed. ~1e-5 is a CPU fp32 figure; fp16 VAR on GPU will be larger and can flip a token | Magnitude is device- and precision-specific | Measured part of `check_seeding` on GPU |
| G10 | Hook overhead (cloning per stage, 250× per DiT batch) is small next to the model | CPU timing is not representative | `timing.py` with an observer on each side |

---

## Open questions

1. **VAR `p`:** keep `si/(SN-1)` (upstream's `ratio`, uniform in scales) or use `cum_tokens / total_tokens`?
   Both are monotonic from 0 to 1 but disagree a lot in the middle for d20.
2. **DiT `p`:** step count `j/(S-1)` (proposed) or noise level (e.g. `1 - sigma_in`)? For the evenly
   respaced 250 steps they are nearly equal; they diverge for other respacings.
3. **VAR pre-transformer hook point:** `before(si)` sits after scale si's sampling. The only state that
   is not otherwise reachable is at scale 0 (the `sos` input). Is a hook there needed?
4. **Logits hooks (VAR):** a hook point inside the sampler wrapper (before `multinomial`) would allow
   steering the distribution while keeping the draws paired. In scope?
5. **KV cache:** should VAR hooks see or edit `cached_k/cached_v`? Masking entries keeps shapes; removing
   them changes the attended length (no draw-count change, but a structural change to the model).
6. **Pre-modification observers:** observers see the state after that hook point's modify hooks. Is a
   way to observe the unmodified state at the same point also needed?
7. **Per-row targeting:** hooks get `rows` (class_id, seed per batch row) to decide which rows to
   change. Is that enough, or should the runner offer a helper that applies a hook to selected rows only?
8. **Skip windows:** implemented: consecutive skipped timesteps merge into one transition (§11b). Open:
   should hold have a multi-step form?
9. **Hooks on skipped steps:** decided: registering a hook for a skipped timestep is an error; hooks
   registered for all stages simply get no call at a skipped step (the `stages` table has no row for it).
10. **Hook cost:** if G10 shows cloning matters, let observers opt out of cloning (and lose the
    read-only guarantee)?
11. **Recording:** besides hook names and the `stages` table in `run.json`, should hooks be able to emit
    per-row data (e.g. an observer's measurements) into the run output, and in what format?
