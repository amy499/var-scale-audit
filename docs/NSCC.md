# Running on NSCC ASPIRE 2A

Everything below is typed by you. `<NSCC_USER>` is your NSCC username, `<NTU_USER>` your NTU username,
`<PROJECT_ID>` **your own** NSCC project ID (`myprojects` shows it), `<JOBID>` the number `qsub`
prints (e.g. `1234567` from `1234567.pbs101`), and `pN` your lane (`p1` … `p4`).

Edit code on your laptop (README, "Setup"), push your branch, and only pull and run on NSCC:

```bash
cd ~/var-scale-audit && git fetch origin && git checkout <your-branch> && git pull
git submodule update --init && python scripts/check_third_party.py
```

## Where things live

| What | Where | Why |
|---|---|---|
| Repo + outputs | `~/var-scale-audit` (your home) | Home: never purged; 50 GB quota |
| Conda env `var-dit` (~6–10 GB) | `~/.conda/envs/var-dit` | Home |
| Conda package cache | `/scratch/users/<org>/<NSCC_USER>/conda-pkgs` (if that folder exists) | Scratch: 30-day purge is fine for a cache |
| Checkpoints (~5.9 GB) | If `/home/project/<PROJECT_ID>` exists: `/home/project/<PROJECT_ID>/var-scale-audit/checkpoints`, symlinked as `checkpoints/`. Otherwise (e.g. most personal projects): a plain `checkpoints/` folder in the repo, in your home | Neither is ever purged |

Never put checkpoints on `/scratch`: files not accessed for 30 days are deleted without notice.

## 1. Log in

You need the NTU VPN (or the NTU network). NSCC is reached through the NTU gateway:

```bash
ssh <NTU_USER>@172.21.26.100          # the NTU gateway, from your laptop
ssh <NSCC_USER>@103.72.192.6          # NSCC, from the gateway
```

Optional: put this in `~/.ssh/config` on your laptop, and `ssh nscc` does both hops (VPN still needed).
The same entry works in VS Code's Remote - SSH extension.

```
Host ntu-gw
    HostName 172.21.26.100
    User <NTU_USER>

Host nscc
    HostName 103.72.192.6
    User <NSCC_USER>
    ProxyJump ntu-gw
```

If your account was set up for the NTU jump host instead, `ssh <NSCC_USER>@aspire2antu.nscc.sg` also
works. First-time jump-host access: email hpcsupport@ntu.edu.sg.

## 2. Check quota

```bash
myprojects                 # your project IDs
myquota                    # home: you need about 16 GB free (env + checkpoints)
myquota -p <PROJECT_ID>    # only if your project has /home/project space
```

## 3. Clone (once)

```bash
cd ~
git clone --recurse-submodules https://github.com/amy499/var-scale-audit.git
cd ~/var-scale-audit
```

If the repo is private, use a GitHub personal access token as the password, or an SSH key.

## 4. Set up on the login node (once; safe to re-run; no GPU work)

```bash
cd ~/var-scale-audit
tmux new -s setup        # optional: survives SSH drops (reattach: tmux attach -t setup)
bash scripts/nscc/setup_login.sh <PROJECT_ID>
```

This checks out the submodules, sets up `checkpoints/` (see the table above), creates the conda env,
downloads and hash-checks the checkpoints, and checks `third_party/`. If NSCC kills it on the login node
or your connection drops, run the same command again: downloads resume and the env is only rebuilt if
`environment.yml` changed. It must end with `Setup complete.`

## 5. Every new session

```bash
cd ~/var-scale-audit
module load miniforge3
module load git/2.39.2           # if NSCC no longer has this version: module avail git, load any
eval "$(conda shell.bash hook)"
conda activate var-dit
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
```

## 6. Project ID: always your own

Every job is charged to the project given on the command line: `qsub -P <PROJECT_ID> <job file>`.
Some committed job files (all of `jobs/`) carry a `#PBS -P` line with P1's project, for P1's runs.
Don't edit or commit `#PBS -P` lines; the command-line `-P` overrides them (standard PBS behaviour).
After every submit, check:

```bash
qstat -f <JOBID> | grep -i project    # must show your project ID
```

If it shows someone else's project, `qdel <JOBID>` and tell P1.

## 7. Your first job: the smoke test

```bash
mkdir -p lanes/pN/jobs outputs/pN/pbs
cp scripts/examples/smoke.pbs lanes/pN/jobs/smoke.pbs   # then, inside the copy, replace every pN with your lane
qsub -P <PROJECT_ID> lanes/pN/jobs/smoke.pbs            # from the repo root
```

It passes when `outputs/pN/smoke/<JOBID>/job.log` ends with two `MATCH  PASS` lines. 1 GPU with
≤ 1 h walltime on `normal` is routed to the `gdev` development queue (max 2 jobs per user).

## 8. Monitor and read results

```bash
qstat -u $USER                                  # Q = queued, R = running, finished jobs disappear
tail -f outputs/pN/<job>/<JOBID>/job.log        # live log once it is running (Ctrl-C to stop)
```

Every job writes into its own `outputs/pN/<job>/<JOBID>/` folder; PBS's own log goes to
`outputs/pN/pbs/`. Copy files to your laptop (run on the laptop, VPN connected):

```bash
scp -J <NTU_USER>@172.21.26.100 '<NSCC_USER>@103.72.192.6:~/var-scale-audit/outputs/pN/…/*.png' .
scp 'nscc:~/var-scale-audit/outputs/pN/…/*.png' .     # same, with the ~/.ssh/config from step 1
```

## P1's verification jobs (`jobs/`)

`jobs/phase1_check.pbs` … `jobs/phase3b_depths.pbs` are the verification jobs behind the runner
(README, "Precision and reference hashes"). You don't need them for experiments. Anyone can re-run one
with `qsub -P <PROJECT_ID> jobs/<file>.pbs`; each writes `outputs/<job name>/<JOBID>/` with a
`summary.txt`. For example, `phase1_check` writes:

| File | Content |
|---|---|
| `job.log`, `summary.txt` | full log; the summary to paste back |
| `nvidia-smi.txt`, `gpu.json` | GPU, driver, torch/CUDA versions, `CUDA_VISIBLE_DEVICES` |
| `runner_{var,dit}.{png,pt,json}` | runner output: image, raw tensor, settings/attention record |
| `ref_{var,dit}_{matched,asis}.{png,pt}` | official-notebook code path outputs |
| `equivalence.json` | tensor hashes and max differences |
| `timing_{var,dit}.json` | timings, peak VRAM, 1,000-image extrapolation |

## If something goes wrong

- **`EnvironmentNameNotFound: var-dit`**: step 4 didn't finish; re-run it.
- **`qsub: Unknown project` / job rejected**: the project after `qsub -P` is wrong (check `myprojects`),
  or you left out `-P` and the job file names a project you can't charge.
- **`qstat -f <JOBID> | grep -i project` shows someone else's project**: `qdel <JOBID>` and tell P1.
- **No `outputs/pN/<job>/<JOBID>/` folder**: the job didn't start in the repo root (always `qsub` from
  `~/var-scale-audit`). PBS's own log in `outputs/pN/pbs/` says why.
- **`FATAL: torch.cuda.device_count() == 0`** (verification jobs): the summary's `DIAGNOSIS` lines and
  `CUDA_VISIBLE_DEVICES` say why. Send the summary to P1; don't work around it in the job.
- **Smoke test `MISMATCH`**: don't work around it; send P1 the folder listing and `job.log`.
- **`module: command not found`** or **`module load git/2.39.2` fails**: you are not on NSCC, or that
  module version was removed; `module avail git` lists the current ones.
