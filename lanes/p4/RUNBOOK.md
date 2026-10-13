# P4 runbook: what to run, where, and what you should see

Owner: P4. Two halves, and the split matters:

| Half | Where | Status |
|---|---|---|
| Progress mapping, bands, plot template, Figure 5 frame, arm construction, the VAR hook | **Your laptop, CPU** | Built and verified |
| The Gate B pilot itself (real VAR-d20 and DiT-XL/2 images) | **NSCC GPU** (PBS, §B) or **TC2 GPU** (SLURM, §B′) | Jobs written, **not yet submitted** |

**No result ever comes from a CPU run.** The CPU checks prove the mechanics — that the hook removes
exactly what it claims, that every arm stays paired, that the bands are what the document says. The
numbers in the report come from NSCC.

---

## A. Local, on the laptop (about 5 minutes)

### A0. Once per machine

```bash
cd ~/Documents/Project/var-scale-audit
git submodule update --init          # third_party/VAR and third_party/DiT, pinned
conda activate var-dit
export PYTHONDONTWRITEBYTECODE=1     # every terminal; upstream imports otherwise dirty the submodules
python scripts/check_third_party.py  # expect two [ OK ] lines
```

### A1. The shared deliverable (no GPU, no checkpoints, no torch)

```bash
python lanes/p4/check_p4_shared.py
```

Expect it to end in:

```
ALL P4 SHARED CHECKS PASS  (NOT RUN: real run.json comparison (no --runs))
```

That one line is the contract: it says what passed **and** what was not exercised. It checks the
progress mapping for both models, the frozen band lists against the frozen configs, every number in
`comparison_logic.md` §4, and the plot template — including that the data layer still works with
matplotlib and Pillow both blocked.

Look at the published bands and the mapping yourself:

```bash
python -m lanes.p4.bands                 # the band lists, with the cuts that produced them
python -m lanes.p4.bands --full          # ... every native stage, including all 250 DiT timesteps
python -m lanes.p4.progress <run dir>    # both progress axes for any run.json you have
python -m lanes.p4.arms --model dit --band middle --stages   # the four arms' stage sets
```

### A2. The pilot mechanics, on tiny CPU models

Build tiny random-weight models once (one process per repo — they cannot share an interpreter):

```bash
mkdir -p /tmp/p4tiny      # make_tiny writes into this directory but does not create it
python scripts/phase2/check_seeding.py --make-tiny var /tmp/p4tiny
python scripts/phase2/check_seeding.py --make-tiny dit /tmp/p4tiny
```

Then:

```bash
python lanes/p4/check_p4_pilot.py --tiny-dir /tmp/p4tiny
```

Expect it to end in `ALL P4 PILOT CHECKS PASS`. This is the important one. It proves, on real
sampling rather than in the abstract:

- the VAR hook at λ = 0 reproduces the repo's own tested `restore_all` removal **bit for bit**;
- at λ = 1 the run is **bit-identical** to the no-hook baseline;
- every arm's per-row `generator_state_sha256` equals its baseline's, so the arms are paired;
- a DiT arm's stages table is missing exactly its skipped timesteps and no others;
- the pairing gate **fails** on a tampered generator state, a dropped row, and an arm that changed
  no image (three deliberate negative controls).

Without `--tiny-dir` the script still runs, but says so in its pass line — the bit-level proof is the
part that needs real sampling.

### A3. A whole pilot end to end, on tiny models

This is the same code path NSCC will run, just with 4-scale / 20-step toy models, so it finishes in
seconds and the images are meaningless:

```bash
python -m lanes.p4.pilot run --model var --band middle \
    --config /tmp/p4tiny/var_tiny.yaml --manifest manifest/provisional_4x4.csv \
    --out-root /tmp/p4pilot/var
python -m lanes.p4.pilot run --model dit --band middle \
    --config /tmp/p4tiny/dit_tiny.yaml --manifest manifest/provisional_4x4.csv \
    --out-root /tmp/p4pilot/dit

python lanes/p4/verify_pairing.py /tmp/p4pilot/var      # expect ALL ARMS PAIRED
python lanes/p4/verify_pairing.py /tmp/p4pilot/dit
python -m lanes.p4.metric_placeholder --pilot /tmp/p4pilot/var
python -m lanes.p4.metric_placeholder --pilot /tmp/p4pilot/dit
python -m lanes.p4.figure4_draft /tmp/p4pilot/var /tmp/p4pilot/dit --out /tmp/p4pilot/figure4.png
```

Open `/tmp/p4pilot/figure4.png`. Four arms per model against the intervened fraction. The numbers are
noise (toy weights); the point is that the chain runs.

**See a gate actually bite.** Delete one arm and run the verifier again:

```bash
mv /tmp/p4pilot/var/damage_m1 /tmp/ && python lanes/p4/verify_pairing.py /tmp/p4pilot/var; echo "exit=$?"
mv /tmp/damage_m1 /tmp/p4pilot/var/
```

It should print `FAIL  the pilot planned 4 runs; these never produced a run.json: ['damage_m1']` and
exit 1. If a pilot silently loses an arm on the cluster, that is what catches it.

### Note on figures

`matplotlib` is in **neither** `environment.yml` nor `environment-macos.yml`. The template falls back
to Pillow, which both carry, so figures render today. If P1 adds matplotlib to the shared env, the
same calls use it automatically — nothing to change.

---

## B. NSCC, on the GPU — **not yet run**

Everything below needs an NSCC account and the NTU VPN. `docs/NSCC.md` and `docs/HANDOVER.md` §2 are
the authority on logging in; this section is only what P4 adds.

### B1. Get the code and checkpoints there

```bash
ssh nscc                                       # per docs/HANDOVER.md 2.1
cd ~/var-scale-audit
git fetch && git checkout p4/progress-mapping-gate-b
bash scripts/nscc/setup_login.sh <your-project-id>     # submodules + checkpoints, safe to re-run
myprojects                                     # note YOUR personal project id
```

### B2. Smoke test first (Gate A for this lane)

```bash
qsub -P <your-project-id> lanes/p4/jobs/smoke.pbs
qstat -f <JOBID> | grep -i project     # MUST show your project; qdel the job if it does not
qstat -u $USER                         # Q queued, R running, gone = finished
tail -f outputs/p4/smoke/<JOBID>/job.log
```

Passes on **two `MATCH  PASS` lines**, one VAR and one DiT. The cuDNN nvrtc warning is expected and
harmless. **A `MISMATCH` is never worked around** — send P1 the output folder listing and `job.log`.

Do not run the pilot until the smoke test passes. It is the check that this machine reproduces the
reference hashes at all.

### B3. The Gate B pilot

```bash
qsub -P <your-project-id> lanes/p4/jobs/pilot_var.pbs    # ~20 min of a 45 min walltime
qsub -P <your-project-id> lanes/p4/jobs/pilot_dit.pbs    # ~25 min of a 2 h walltime
qstat -f <JOBID> | grep -i project                       # check both
```

Each job runs 10 runs — the no-hook baseline **first**, then protect / damage / control at `m/n` of
0.1, 0.2 and 0.3 — then verifies pairing itself. Watch for the last line of `job.log`:

```
P4 PILOT COMPLETE AND PAIRED
```

Anything else means do not use those runs. The job prints `P4 PILOT FAILED (pilot exit N, pairing
exit M)` and exits non-zero if a run failed or an arm is missing or unpaired.

DiT dominates the budget: 31.9 s per batch of 16 plus ~2 min model load. Size any later sweep from
that number, never from VAR's.

### B4. Numbers and the figure

```bash
python lanes/p4/verify_pairing.py outputs/p4/pilot_var/<JOBID>    # 16/16 per arm
python -m lanes.p4.metric_placeholder --pilot outputs/p4/pilot_var/<JOBID>
python -m lanes.p4.metric_placeholder --pilot outputs/p4/pilot_dit/<JOBID>
python -m lanes.p4.figure4_draft outputs/p4/pilot_var/<JOBID> outputs/p4/pilot_dit/<JOBID> \
    --out outputs/p4/figure4_draft.png
git log -1 --oneline        # record this beside the results; run.json has the submodules, not the branch
```

The metric is a **placeholder** — distance from each arm's image to its paired baseline image. It
shows the plumbing works. P3's calibrated quality metrics replace it through the same contract, with
no change to the figure code.

---

## B′. CCDS GPU Cluster TC2 (SLURM) — alternative to NSCC

TC2 is an NTU CCDS GPU cluster (NVIDIA A40s) using **SLURM** instead of PBS. The workflow is the same
as section B but with different SSH, modules, job commands, and resource limits. Use this section
**instead of** section B when running on TC2.

### TC2 quick reference

| Item | Value |
|---|---|
| Head node | `10.96.189.12`, port 22 (NTU network or VPN only) |
| Scheduler | SLURM (`sbatch` / `squeue` / `scancel`) |
| Partition | `MGPU-TC2` |
| QoS | `normal` |
| GPUs | NVIDIA A40 (4–8 per node) |
| Max concurrent jobs | 2 per user |
| Max walltime | 6 hours |
| Max resources per user | 10 CPUs, 1 GPU, 30 GB RAM |
| Home directory | `/home/msai/<username>` |
| Disk quota | 100 GB |
| Conda | `module load anaconda` (system-wide Anaconda) |
| No `/scratch` | Everything lives in your home; manage disk carefully |

### B′1. Log in

TC2 is accessible only on the NTU network (or via NTU VPN).

```bash
ssh <username>@10.96.189.12              # NTU network account ID, lowercase
```

Optional `~/.ssh/config` entry:

```
Host tc2
    HostName 10.96.189.12
    User <username>
```

Then `ssh tc2` is enough (VPN still required from off campus).

On first login your home directory is created. Check your quota and QoS:

```bash
mytcinfo                                 # home dir, QoS, resource limits
ncdu                                     # interactive disk usage viewer
```

### B′2. Clone the repo and set up the environment (once)

```bash
cd ~
git clone --recurse-submodules https://github.com/amy499/var-scale-audit.git
cd ~/var-scale-audit
python scripts/check_third_party.py      # expect two [ OK ] lines
```

Set up conda (first time only):

```bash
module load anaconda
conda init bash && source ~/.bashrc
conda env create -f environment.yml      # creates var-dit env
conda activate var-dit
```

Auto-load in every session — add to `~/.bashrc`:

```bash
module load anaconda
conda activate var-dit
```

Download checkpoints:

```bash
export PYTHONDONTWRITEBYTECODE=1
python scripts/download_checkpoints.py   # ~5.9 GB; checks hashes
python scripts/check_third_party.py      # re-check after download
```

TC2 has no `/scratch`. Everything lives in your home directory. The 100 GB quota must hold the
conda env (~6–10 GB), checkpoints (~5.9 GB), and outputs. Run `ncdu` regularly and pull outputs to
your laptop as they complete.

### B′3. Every new session

```bash
cd ~/var-scale-audit
module load anaconda
eval "$(conda shell.bash hook)"
conda activate var-dit
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
```

### B′4. Smoke test first (Gate A)

```bash
mkdir -p outputs/p4/slurm
sbatch lanes/p4/jobs/smoke_tc2.sh
squeue -u $USER                          # PD = pending, R = running, gone = finished
```

Once finished, check:

```bash
cat outputs/p4/smoke/<JOBID>/job.log     # look for two MATCH PASS lines
```

The SLURM stdout/stderr go to `outputs/p4/slurm/output_p4_smoke_<JOBID>.out` and
`error_p4_smoke_<JOBID>.err`. The application log is `outputs/p4/smoke/<JOBID>/job.log`.

**A `MISMATCH` is never worked around.** Send P1 the output folder listing and `job.log`.

Do not run the pilot until the smoke test passes.

### B′5. The Gate B pilot

```bash
sbatch lanes/p4/jobs/pilot_var_tc2.sh    # ~20 min of a 45 min walltime
sbatch lanes/p4/jobs/pilot_dit_tc2.sh    # ~25 min of a 2 h walltime
squeue -u $USER                          # check both
```

TC2 allows at most **2 concurrent jobs** per user, so both can run at once. Watch for the last
line of `job.log`:

```
P4 PILOT COMPLETE AND PAIRED
```

Anything else means do not use those runs.

### B′6. Numbers and the figure

Same as section B4, using the TC2 output paths:

```bash
python lanes/p4/verify_pairing.py outputs/p4/pilot_var/<JOBID>
python -m lanes.p4.metric_placeholder --pilot outputs/p4/pilot_var/<JOBID>
python -m lanes.p4.metric_placeholder --pilot outputs/p4/pilot_dit/<JOBID>
python -m lanes.p4.figure4_draft outputs/p4/pilot_var/<JOBID> outputs/p4/pilot_dit/<JOBID> \
    --out outputs/p4/figure4_draft.png
git log -1 --oneline
```

### B′7. Monitoring and troubleshooting (TC2 / SLURM)

| Action | Command |
|---|---|
| Check job queue | `squeue -u $USER` or `squeue -la` |
| Job details | `scontrol show jobid <JOBID>` |
| Cancel a job | `scancel <JOBID>` |
| Job history | `sacct -u $USER` or `myjobhistory` |
| Completed job efficiency | `seff <JOBID>` |
| Node status | `sinfo -N -l` |
| GPU assigned to running job | `scontrol -d show jobid <JOBID>` (look for `JOB_GRES`) |

Common SLURM failure reasons (visible via `squeue -la`):

| Reason | What to do |
|---|---|
| `QOSMaxJobsPerUserLimit` | 2 jobs max. Wait for one to finish. |
| `QOSMaxCpuPerUserLimit` / `QOSMaxMemoryPerUser` / `QOSMaxGRESPerUser` | Total resources across all running jobs exceed QoS. Reduce or run one job at a time. |
| `Priority` / `Resources` | Normal queuing. Defer submission if many jobs are pending. |

### B′8. Copying results to your laptop

From your laptop (NTU VPN connected):

```bash
scp '<username>@10.96.189.12:~/var-scale-audit/outputs/p4/…/*.png' .
scp 'tc2:~/var-scale-audit/outputs/p4/…/*.png' .    # with the ~/.ssh/config from B′1
```

Or use FileZilla (SFTP, host `10.96.189.12`, port 22).

### PBS ↔ SLURM cheat sheet (for P4 jobs)

| PBS (NSCC) | SLURM (TC2) |
|---|---|
| `qsub -P <project> job.pbs` | `sbatch job.sh` |
| `qstat -u $USER` | `squeue -u $USER` |
| `qstat -f <JOBID>` | `scontrol show jobid <JOBID>` |
| `qdel <JOBID>` | `scancel <JOBID>` |
| `#PBS -N name` | `#SBATCH --job-name=name` |
| `#PBS -q normal` | `#SBATCH --qos=normal` |
| `#PBS -l select=1:ngpus=1` | `#SBATCH --nodes=1` + `#SBATCH --gres=gpu:1` |
| `#PBS -l walltime=HH:MM:SS` | `#SBATCH --time=HH:MM:SS` |
| `#PBS -j oe` | `#SBATCH --output=…` + `#SBATCH --error=…` |
| `$PBS_JOBID` | `$SLURM_JOB_ID` |
| `$PBS_O_WORKDIR` | `$SLURM_SUBMIT_DIR` |
| `module load miniforge3` | `module load anaconda` |

---

## C. If something fails

| Symptom | What it means |
|---|---|
| `checkpoint not downloaded: ...` | Run `scripts/nscc/setup_login.sh` (NSCC) or `python scripts/download_checkpoints.py` (TC2) |
| `MISMATCH` in the smoke log | Stop. Send P1 the folder listing and `job.log`. Never work around it. |
| `PAIRING FAILED` | The comparison is invalid. Re-run it; never adjust anything to make it pass. |
| `P4 PILOT FAILED` | At least one run failed or an arm is missing. The log names which. |
| `no plotting library is installed` | Neither matplotlib nor Pillow is importable. Pillow is in both env files; check the env is active. |
| Check script fails after a `runner/` change | Expected if P1 changed the runner. Re-run both check scripts and tell P4 what broke. |
| `EnvironmentNameNotFound: var-dit` (TC2) | Conda env not created. Run `conda env create -f environment.yml` after `module load anaconda`. |
| `CommandNotFoundError` on `conda activate` (TC2) | Initialize conda first: `conda init bash`, then `source ~/.bashrc`, then `eval "$(conda shell.bash hook)"`. |
| Job stuck in PD with `QOSMaxJobsPerUserLimit` (TC2) | Already have 2 jobs running. Wait or `scancel` one. |
| No output folder created (TC2) | Job didn't start in the repo root. Check `outputs/p4/slurm/error_*.err` for details. |
| Disk quota exceeded (TC2) | TC2 home is 100 GB with no `/scratch`. Run `ncdu`, remove old outputs, pull results to laptop. |
