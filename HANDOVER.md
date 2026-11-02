# Handover: using the shared pipeline (P2, P3, P4)

This is what you need to run your lane's VAR and DiT experiments on the shared runner. It covers what
the pipeline does, how to set it up on NSCC, the rules that keep everyone's results comparable, your
first hook, and running jobs.

- New terms: see the glossary at the end (§8).
- Writing hooks: `docs/hooks_quickstart.md`.

---

## 1. What the pipeline gives you, and what it doesn't

### It gives you

- **One command for both models:** `python -m runner.generate --config … --manifest …`. It generates
  VAR-d20 or DiT-XL/2 (256px, ImageNet) images, runs the official code unmodified, and was verified
  bit-identical to the official code on NSCC's A100s.
- **Reproducible images.** Each image's randomness comes only from its own seed. The same config,
  manifest and batch size give bit-identical images every run.
- **Paired interventions.** Hooks let you read or change the model's state before or after every
  stage (VAR: 10 scales; DiT: 250 steps). Your intervened run uses exactly the same random numbers as
  the baseline, so differences come from your change.
- **Native stage and normalized progress.** Every hook call gets `stage` (VAR scale 0–9, DiT timestep
  999…0) and `p` in [0, 1]. The per-stage numbers (token counts, ᾱ, σ) come with every call and in
  `run.json`.
- **DiT skip-timestep** (`--skip-timesteps T …`), with pairing kept.
- **VAR token capture** (`--capture-tokens`): per scale, the sampled tokens, top-2 probabilities and
  the chosen token's probability. It's read-only, so outputs don't change.
- **A `run.json` per run**: every setting, versions, submodule commits, hooks used, stages table, and
  each image's output hash and random-generator hash.
- **NSCC setup**: one setup script, checkpoint download with hash checks, and job examples.

### It doesn't give you (yet, or ever)

| Missing | Who |
|---|---|
| The final manifest (dog classes, count, seeds) | P3 (in progress) |
| Quality/semantic metrics (FID, LPIPS, CLIP, entropy…) and their calibration | P3, with P1 validating |
| The shared result schema and merged logs | P2 (pending). Until then, `run.json` is the record. |
| The early/mid/late mapping decision and plot template | P4 |
| Sweep drivers, analysis and figures for your lane | You, in `lanes/pN/` |
| Canonical clean baselines on the final manifest | P1, once the manifest exists |
| Hook access to VAR logits, the KV cache, or the point before a scale's transformer pass | Not supported. Ask P1 if your lane needs it (`docs/hook_interface.md`, Open questions). |
| Dropping tokens or rows, changing resolution or step count mid-run | Not allowed: it breaks pairing |
| A built-in way for hooks to write per-image data into `run.json` | Write your own file (§4) |
| Other datasets (DIBaS, OpenCrystalData), training, multi-GPU | Out of scope for the core |

---

## 2. First-time setup on NSCC, from SSH to a passing smoke test

You need an NSCC account, the NTU VPN, and access to the GitHub repo.

**Project ID.** Each person charges jobs to **their own NSCC personal project**. `myprojects` (below)
shows its ID. The job files in `jobs/` contain `#PBS -P personal-gura0001`, which is P1's personal
project, kept for P1's runs. Don't edit that line. Always submit with your own project on the command
line, `qsub -P <your-project-id> <job file>`. Then confirm with `qstat -f <JOBID> | grep -i project`
that your project is shown. A `qsub` option overriding a `#PBS` line is standard PBS behaviour, but we
haven't confirmed it on NSCC yet, so always do the `qstat` check.

### 2.1 Log in

NSCC is reached through NTU's gateway, in three hops:

1. Connect to the **NTU VPN**.
2. `ssh <NTU_USER>@172.21.26.100` (the NTU gateway).
3. From the gateway, `ssh <NSCC_USER>@103.72.192.6` (NSCC).

```bash
ssh <NTU_USER>@172.21.26.100          # on your laptop, with the NTU VPN connected
ssh <NSCC_USER>@103.72.192.6          # on the gateway
myprojects                            # your projects; note your personal project ID
```

**Optional: one-step login, and VS Code.** Put this in `~/.ssh/config` on your laptop
(`C:\Users\<you>\.ssh\config` on Windows):

```
Host ntu-gw
    HostName 172.21.26.100
    User <NTU_USER>

Host nscc
    HostName 103.72.192.6
    User <NSCC_USER>
    ProxyJump ntu-gw
```

Then `ssh nscc` goes through the gateway in one command (VPN still required). The same entry works in
VS Code: install the **Remote - SSH** extension, run "Remote-SSH: Connect to Host…" and pick `nscc`. You
can then open `~/var-scale-audit` and use a terminal on NSCC inside VS Code. The terminal there is still
the login node: GPU work still goes through `qsub`.

**Troubleshooting:** if Windows `ssh` fails with `Corrupted MAC on input`, add the line
`MACs hmac-sha2-256` under **both** `Host` entries (indented like the other lines).

### 2.2 Clone (once)

```bash
cd ~
git clone --recurse-submodules https://github.com/amy499/var-scale-audit.git
cd ~/var-scale-audit
```

If the repo is private, use a GitHub personal access token as the password.

### 2.3 Run the login-node setup (once; safe to re-run)

```bash
cd ~/var-scale-audit
tmux new -s setup                              # optional: survives SSH drops (tmux attach -t setup)
bash scripts/nscc/setup_login.sh <your-project-id>
```

This does five things:

1. Checks out the pinned submodules.
2. Sets up `checkpoints/`. If `/home/project/<your-project-id>` exists, it links there. Otherwise
   (as on P1's personal project) it makes a plain `checkpoints/` folder in the repo, in your home.
3. Builds the `var-dit` conda env (takes a while the first time).
4. **Downloads your own copy of the checkpoints** (VAR-d20 + DiT, about 5.9 GB) and hash-checks every
   file. Files already there are skipped but still hash-checked. Every file, VAR and DiT, is checked
   against the sha256 pinned in its config; a mismatch is reported as `[FAIL]`.
5. Checks `third_party/` is clean.

Space: the env (6–10 GB) and checkpoints (~5.9 GB) both go in your home, which has a 50 GB quota. VAR-d24
adds 4.1 GB, but only if you download it (`--configs configs/var_d24.yaml`).

It must end with **`Setup complete.`** If the login node kills it, run the same command again: downloads
resume. Its last lines repeat the project-ID rule above (`qsub -P <your-project-id>`, then the `qstat`
check).

### 2.4 Every new session (every SSH login, every new tmux window)

```bash
cd ~/var-scale-audit
module load miniforge3
module load git/2.39.2
eval "$(conda shell.bash hook)"
conda activate var-dit
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
```

Your prompt should now start with `(var-dit)`. If `conda activate` says `EnvironmentNameNotFound`, step
2.3 didn't finish; re-run it.

### 2.5 Login-node checks (CPU only, quick)

```bash
python scripts/check_third_party.py                 # two "[ OK ]" lines
python scripts/download_checkpoints.py --dry-run    # every file marked "(exists)"
```

Optional: `python scripts/phase2/check_seeding.py` runs a CPU test with tiny random models, with no
GPU or checkpoints. It should end in PASS. If the login node kills it, skip it; the GPU smoke test
below matters more.

### 2.6 GPU smoke test (your first job)

Create your lane folders:

```bash
mkdir -p lanes/pN/jobs outputs/pN/pbs          # replace N with your number, here and below
```

Copy the example job into your lane: `cp scripts/examples/smoke.pbs lanes/pN/jobs/smoke.pbs`, then replace
`N` with your number. For this first test it's fine to make the copy directly on NSCC. Your later job
files go through your git branch (§6). If you later commit `lanes/pN/jobs/smoke.pbs` from your laptop,
delete the NSCC copy before `git checkout` on NSCC, or git refuses to overwrite it.
This is `scripts/examples/smoke.pbs`:

```bash
#!/bin/bash
#PBS -N pN_smoke
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o outputs/pN/pbs/
# GPU smoke test (docs/HANDOVER.md 2.6). Copy to lanes/pN/jobs/smoke.pbs, replace N with your number,
# and submit from the repo root with your own project:  qsub -P <your-project-id> lanes/pN/jobs/smoke.pbs
# Passes when both regression lines end in PASS.
set -uo pipefail
[[ -n "${PBS_JOBID:-}" ]] || { echo "submit with qsub; no GPU work on the login node" >&2; exit 1; }
cd "$PBS_O_WORKDIR"
[[ -f runner/generate.py ]] || { echo "submit from the repo root" >&2; exit 1; }

VAR_CONFIG="${VAR_CONFIG:-configs/var_d20.yaml}"
DIT_CONFIG="${DIT_CONFIG:-configs/dit_xl2_256.yaml}"
EXPECT_VAR="${EXPECT_VAR:-f16021ea827ab2ea38e70aab22b8e87d64a47f2c9bc8d6500fce72108cdef2c8}"   # VAR d20 fp32
EXPECT_DIT="${EXPECT_DIT:-2a3a0d6fbce9}"
JOB="${PBS_JOBID%%.*}"
OUT="${OUT_ROOT:-outputs/pN}/smoke/$JOB"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/job.log") 2>&1

set +u
module load miniforge3
module load git/2.39.2
eval "$(conda shell.bash hook)"
conda activate var-dit
set -u
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1

nvidia-smi
python scripts/check_third_party.py
python -m runner.generate --config "$VAR_CONFIG" --class-id 207 --seed 0 --out "$OUT/var" --save-raw
python -m runner.generate --config "$DIT_CONFIG" --class-id 207 --seed 0 --out "$OUT/dit" --save-raw
python scripts/phase3/regression.py --out "$OUT/regression.json" \
    var "$OUT/var.pt" - "$EXPECT_VAR" \
    dit "$OUT/dit.pt" - "$EXPECT_DIT"
```

Submit it, check the project, and watch:

```bash
qsub -P <your-project-id> lanes/pN/jobs/smoke.pbs   # your project on the command line; don't edit #PBS -P lines
qstat -f <JOBID> | grep -i project                  # must show your project ID
qstat -u $USER                                      # Q = queued, R = running; gone = finished
tail -f outputs/pN/smoke/<JOBID>/job.log            # once it is running
```

**It passes** when the log ends with:

```
var: plain f16021ea827a expected f16021ea827ab2ea38e70aab22b8e87d64a47f2c9bc8d6500fce72108cdef2c8 MATCH  PASS
dit: plain 2a3a0d6fbce9 expected 2a3a0d6fbce9 MATCH  PASS
```

The log shows a cuDNN warning ('Applied workaround for CuDNN issue, install nvrtc.so'). It's expected
and harmless; the smoke test still passes.

Two PASS lines mean your setup produces bit-identical images to everyone else's. If you get a MISMATCH,
don't work around it: send P1 the whole `outputs/pN/smoke/<JOBID>/` folder listing and `job.log`.

---

## 3. Frozen settings and rules

These rules keep everyone's baselines identical and every comparison honest. Each one exists because a
verification job showed that breaking it changes images.

| Rule | Why |
|---|---|
| **Configs:** `configs/var_d20.yaml` (VAR) and `configs/dit_xl2_256.yaml` (DiT), exactly as committed | They are the frozen settings: cfg 1.5, top-k 900, top-p 0.96, DiT 250 steps, math attention kernel. |
| **Batch size 16** for the baseline **and** every intervention you compare with it | Batch size changes float rounding. Even in fp32, batch 1 vs 16 left no row bit-identical, so a batch mismatch can fake or hide an effect. |
| **Manifest size must be a multiple of 16.** Never use `--allow-partial-batch` for experiments | A smaller last batch is a different batch size. |
| **Same config, manifest and seeds** for the baseline and the intervention | That's what makes them paired. |
| **Never mix VAR depths** (d20 vs d24) in one comparison | They are different models. `run.json` records `depth`. |
| **Strict fp32 VAR.** Never use `configs/var_d20_fp16.yaml` | In fp16, VAR images changed visibly with batch size on 16/16 rows. That config exists only for historical jobs. |
| **Never edit `runner/`, `third_party/`, `configs/`, `jobs/`, `scripts/`** | Shared by everyone and verified by hash. Copy what you need into your lane. Need a runner change? Ask P1. |
| **Your code goes in `lanes/pN/`** | Hooks, job scripts, analysis, pilot manifests. |
| **Make and keep the clean baseline before the intervention** (plan §4) | Every result is "intervention vs baseline". |
| **Check pairing:** each row's `generator_state_sha256` in your hooked `run.json` must equal the baseline's | If they differ, something consumed random numbers differently and the comparison is invalid. |

---

## 4. Your first hook

Read `docs/hooks_quickstart.md` first. It has the function signature, the tables of what you can read
and change at each point, an observe example and a modify example. `scripts/phase3/hooks_lib.py` has
tested examples, including hooks that change only row 0.

The short version (a summary of the API, not a runnable file):

```python
def my_hook(model, stage, p, when, state): ...
Observe(fn, when="after")                  # read only: gets copies, must return None
Modify(fn, when="after", stages=[3])       # change: return dataclasses.replace(state, f_hat=...) etc.
```

### Example: a P2-style noise injection at VAR scale 3

Illustration of the hook API only; the corruption/severity design is P2's choice.

This is `scripts/examples/noise_hook.py`. Copy it into your lane (e.g. `lanes/p2/hooks.py`) and change it there:

```python
# Illustration of the hook API only; the corruption/severity design is P2's choice.
# Adds Gaussian noise to the VAR accumulator after scale 3. Copy into your lane (e.g. lanes/p2/hooks.py)
# and run from the repo root:  --hook lanes/p2/hooks.py:noise_scale_3
import dataclasses

import torch

from runner.hooks import Modify

SEVERITY = 0.5   # illustrative only; P2's severity schedule is P2's decision


def add_noise(model, stage, p, when, state):
    f = state.f_hat
    # Hook randomness comes from our OWN generators (never the images' generators), seeded per image
    # and stage, so each image's noise does not depend on which batch it is in.
    noise = torch.stack([
        torch.randn(f.shape[1:], generator=torch.Generator(device=f.device).manual_seed(10_000 * seed + stage),
                    device=f.device, dtype=f.dtype)
        for (_class_id, seed) in state.rows])
    scale = f.flatten(1).std(dim=1).view(-1, 1, 1, 1)   # per image, so it doesn't depend on the batch
    return dataclasses.replace(state, f_hat=f + SEVERITY * scale * noise)


noise_scale_3 = Modify(add_noise, when="after", stages=[3], name="noise_scale3_sev0.5")
```

Points worth copying:

- Tensors keep their shape, dtype and device. The runner rejects anything else.
- `state.rows` tells you which image each batch row is: `(class_id, seed)`.
- If an observer writes data, give it an absolute path under `outputs/pN/…`, for example from an
  environment variable as `hooks_lib.py` does with `PHASE3_LOG`. Append one line per call. Hooks run
  once per batch per stage (DiT: 250× per side per batch), so keep them cheap.
- For DiT, the state field is `x` (the latent), not `f_hat`, and `stage` is the timestep. Valid
  timesteps are in the `stages` table of any DiT `run.json`.
- Put helpers in your lane (e.g. `lanes/p2/util.py`) and import them as `from lanes.p2 import util`.
  This works when you run from the repo root, as every command here does. Load the hook with
  `--hook lanes/p2/hooks.py:NAME`.
- Every example here and in `docs/hooks_quickstart.md` is a file in `scripts/examples/`;
  `python scripts/examples/run_examples.py` runs them all on CPU with tiny random weights.

---

## 5. Running an experiment job, and where outputs go

A baseline and an intervention in one job: copy `scripts/examples/experiment.pbs` to `lanes/pN/jobs/`,
replace `N`, and set `HOOK` to your lane's hook. This is `scripts/examples/experiment.pbs`:

```bash
#!/bin/bash
#PBS -N pN_experiment
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o outputs/pN/pbs/
# Baseline + intervention job (docs/HANDOVER.md 5). Copy to lanes/pN/jobs/, replace N and the hook, and
# submit from the repo root with your own project:  qsub -P <your-project-id> lanes/pN/jobs/experiment.pbs
# The baseline and every intervention use the same config, manifest and batch size (16).
set -uo pipefail
[[ -n "${PBS_JOBID:-}" ]] || { echo "submit with qsub; no GPU work on the login node" >&2; exit 1; }
cd "$PBS_O_WORKDIR"
[[ -f runner/generate.py ]] || { echo "submit from the repo root" >&2; exit 1; }

VAR_CONFIG="${VAR_CONFIG:-configs/var_d20.yaml}"
DIT_CONFIG="${DIT_CONFIG:-configs/dit_xl2_256.yaml}"
MANIFEST="${MANIFEST:-manifest/provisional_4x4.csv}"   # switch to P3's frozen manifest once it exists
HOOK="${HOOK:-scripts/examples/noise_hook.py:noise_scale_3}"   # your lane's hook, e.g. lanes/p2/hooks.py:NAME
JOB="${PBS_JOBID%%.*}"
OUT="${OUT_ROOT:-outputs/pN}/experiment/$JOB"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/job.log") 2>&1

set +u
module load miniforge3
module load git/2.39.2
eval "$(conda shell.bash hook)"
conda activate var-dit
set -u
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1

python -m runner.generate --config "$VAR_CONFIG" --manifest "$MANIFEST" --batch-size 16 \
    --out-dir "$OUT/var/baseline"
python -m runner.generate --config "$VAR_CONFIG" --manifest "$MANIFEST" --batch-size 16 \
    --out-dir "$OUT/var/intervention" --hook "$HOOK"
python -m runner.generate --config "$DIT_CONFIG" --manifest "$MANIFEST" --batch-size 16 \
    --out-dir "$OUT/dit/baseline"
```

Submit with `qsub -P <your-project-id> lanes/pN/jobs/experiment.pbs` from the repo root, then check
`qstat -f <JOBID> | grep -i project` shows your project.

### Where things end up

| What | Where |
|---|---|
| Your run's images and tensors | `<out-dir>/class0207_seed0.png`, `.pt` (one pair per manifest row) |
| The run record | `<out-dir>/run.json` |
| Your job's log | `outputs/pN/<job>/<JOBID>/job.log` (from `tee` in the script) |
| PBS's own log (useful if the job never started) | `outputs/pN/pbs/` |
| Default out-dir if you don't pass one | `outputs/<config name>/<manifest name>/`. Pass `--out-dir` so runs don't overwrite each other. |

`outputs/` is in your home folder. It's not in git and not shared with teammates.

### Budget and limits

- **Space:** each `.pt` is about 0.8 MB, so 1,000 images ≈ 1 GB plus PNGs, against a 50 GB home quota.
- **Time, at batch 16, plus model loading:**
  - VAR-d20 fp32: about 95 s per 1,000 images.
  - DiT: about **33 min per 1,000 images** (31.9 s per batch of 16). DiT dominates the budget, so size
    DiT sweeps and walltimes from this number.
- **Queue (unverified):** 1 GPU with ≤ 2 h walltime is said to go to the `gdev` queue, with at most 2
  jobs per user.
- **Cost (unverified):** 64 SU per GPU-hour.
- **Copy images to your laptop.** Run this on the laptop, VPN connected. `-J` jumps through the NTU
  gateway:
  `scp -J <NTU_USER>@172.21.26.100 '<NSCC_USER>@103.72.192.6:~/var-scale-audit/outputs/pN/…/*.png' .`
  With the SSH config from §2.1: `scp 'nscc:~/var-scale-audit/outputs/pN/…/*.png' .`

Because runs are deterministic, you can regenerate the clean baseline yourself instead of copying
someone else's. It should match P1's canonical baseline hash for hash (same config, manifest, batch
size, same GPU type). Compare the `tensor_sha256` values in the two `run.json` files.

---

## 6. Git workflow

**Before you can push a branch,** P1 has to add you as a collaborator on the GitHub repo. Send P1 your
GitHub username, then accept the invitation (email or github.com notifications).

**Never push to `main`.** `main` is protected: direct pushes are rejected. All your work reaches `main`
through a pull request that P1 reviews and merges.

### On your laptop (where you edit)

```bash
git clone --recurse-submodules https://github.com/amy499/var-scale-audit.git   # once
cd var-scale-audit
git checkout main && git pull                  # start from the latest main
git checkout -b pN/<short-topic>               # your branch, e.g. p2/noise-pilot
# ... edit files under lanes/pN/ only ...
git add lanes/pN/
git commit -m "p2: noise injection pilot hooks"
git push -u origin pN/<short-topic>            # first push; later just: git push
```

Then open a **pull request** on GitHub from `pN/<short-topic>` into `main`. P1 reviews and merges it.
Keep pull requests small: one topic, only `lanes/pN/`.

### Keep your branch up to date

Pull `main` into your branch regularly, at least before opening a pull request and whenever P1
announces a runner change:

```bash
git checkout pN/<short-topic>
git fetch origin
git merge origin/main
git push
```

### On NSCC (where you run)

Don't edit code on NSCC; run what you pushed:

```bash
cd ~/var-scale-audit
git fetch origin
git checkout pN/<short-topic>                  # first time; afterwards you're already on it
git pull                                       # get your latest push
git submodule update --init                    # harmless; needed only if the pins changed
python scripts/check_third_party.py
```

Then `qsub -P <your-project-id> lanes/pN/jobs/<job>.pbs` as usual. `run.json` records the commits of
both submodules, but not your branch, so note the branch and commit (`git log -1 --oneline`) with your
results.

---

## 7. Who to ask, and what not to do

**Ask P1** about:

- anything in `runner/`, hooks, configs or checkpoints;
- a smoke-test MISMATCH;
- needing a hook point that doesn't exist;
- "does this break pairing?".

When you ask, send the job's `job.log`, your hook file, the command you ran, and your branch and
commit. P2 is the runner backup.

**Don't:**

- run GPU work on the login node (always `qsub`);
- edit `runner/`, `third_party/`, `configs/`, `jobs/` or `scripts/`. Copy into `lanes/pN/` instead;
- commit an edited `#PBS -P` line. Use `qsub -P` on the command line;
- use `configs/var_d20_fp16.yaml`, or compare against the fp16 hash `48db2d9e166a` (historical only);
- compare runs made at different batch sizes, different VAR depths or different manifests;
- use `--allow-partial-batch`, `--device cpu` or `--no-burn-skipped` for results;
- use the images' random generators inside a hook, or change tensor sizes (mask or replace values instead);
- keep references to tensors you return from a hook;
- `pip install` into `var-dit` without telling P1: it silently changes your environment from everyone
  else's;
- put checkpoints on `/scratch` (purged after 30 days unused);
- commit anything under `outputs/` or `checkpoints/`;
- push to `main`, or merge your own pull request.

---

## 8. Glossary (the terms you'll hear most)

| Term | Meaning |
|---|---|
| **Manifest** | CSV listing the images to make, one `class_id,seed` per line. Same manifest = same images. |
| **Seed** | Number that fixes an image's random choices. Same seed + same settings = same image. |
| **Generator** | A random-number source. The runner gives each image its own, seeded with that image's seed. Hooks never get it. |
| **Pairing** | Baseline and intervened runs use identical random numbers at every stage, so differences come from your change. Checked via each row's `generator_state_sha256` in `run.json`. |
| **Baseline** | The run with no hooks, same config, manifest and batch size. Every result is compared to it. |
| **Batch size** | Images per model call. Always 16: a different batch size changes float rounding. |
| **Hash (sha256)** | A fingerprint of data. Any 1-bit change gives a different hash. `tensor_sha256` fingerprints an image's raw output. We often quote the first 12 characters. |
| **Reference hash** | Known hash of class 207, seed 0 at batch 1. Matching it proves your setup is identical (smoke test). |
| **`run.json`** | The record next to every manifest run: settings, versions, hooks, stages table, per-image hashes. |
| **fp32 / TF32 / autocast** | Full 32-bit floats / an NVIDIA fast mode for fp32 matrix maths / automatic lower precision (fp16). VAR runs **strict fp32**: TF32 off, autocast off. |
| **CFG** | Classifier-free guidance: mix the class-conditioned and "no class" predictions to make images more class-typical. Strength 1.5 for both models. |
| **Scale (VAR)** | One VAR stage: a token grid of 1×1, 2×2, … 16×16 (10 scales, 680 tokens). `f_hat` is the running sum of all scales so far. |
| **Timestep (DiT)** | DiT's noise level, 999 (noise) → 0 (clean). We run 250 steps. `x` is the latent being denoised. |
| **Hook** (Observe / Modify, before / after) | Your function, called at every stage. Observe reads a copy and returns `None`. Modify returns a same-shaped replacement. `before`/`after` = which side of the stage. |
| **Stage and p** | `stage` = native id (VAR scale 0–9, DiT timestep). `p` = progress 0 → 1 through generation, comparable across models. |
| **Depth (d20/d24)** | VAR model size. d20 is primary. Never mix depths in one comparison. |
| **Login node / `qsub`** | The machine you SSH into (no GPU work) / the command that sends a job to a GPU node. |
| **Project ID** | The NSCC project a job is charged to. Yours is your personal project, passed as `qsub -P`. |
