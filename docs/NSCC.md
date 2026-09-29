# Running on NSCC ASPIRE 2A

Everything below is typed by you. `<USER>` is your NSCC username, `<PROJECT_ID>` your NSCC project ID,
`<JOBID>` the number `qsub` prints (e.g. `1234567` from `1234567.pbs101`).

## Where things live

| What | Where | Why |
|---|---|---|
| Repo + outputs | `~/var-scale-audit` (`/home/users/<org>/<USER>`) | Home: never purged; 50 GB quota |
| Conda env `var-dit` (~6–10 GB) | `~/.conda/envs/var-dit` | Home |
| Conda package cache | `/scratch/users/<org>/<USER>/conda-pkgs` | Scratch: 30-day purge is fine for a cache |
| Checkpoints (~5.9 GB) | `/home/project/<PROJECT_ID>/var-scale-audit/checkpoints`, symlinked as `checkpoints/` | Project space: never purged, shared with the team |

Never put checkpoints on `/scratch`: files not accessed for 30 days are deleted without notice.

## 0. On your laptop (once)

Review the diff, then commit and push (including `.gitmodules` and the two submodule entries).

## 1. Log in

NTU users: be on the NTU network (or NTU VPN) and go through the NTU jump host. First-time jump-host
access: email hpcsupport@ntu.edu.sg.

```bash
ssh <USER>@aspire2antu.nscc.sg
```

## 2. Check quota

```bash
myquota
myquota -p <PROJECT_ID>
```

You need about 10 GB free in home and about 6 GB in the project.

## 3. Clone (once)

```bash
cd ~
git clone --recurse-submodules https://github.com/amy499/var-scale-audit.git
cd ~/var-scale-audit
```

(If the repo is private, use a GitHub personal access token as the password, or an SSH key.)

## 4. Set up on the login node (no GPU work)

```bash
cd ~/var-scale-audit
tmux new -s setup        # optional: survives SSH drops (reattach: tmux attach -t setup)
bash scripts/nscc/setup_login.sh <PROJECT_ID>
```

This creates the conda env, downloads and verifies all checkpoints, and checks `third_party/`. It is safe to
re-run: if NSCC kills it on the login node or your connection drops, run the same command again. Downloads
resume and the env is only rebuilt if `environment.yml` changed. It ends with `Setup complete.`

## 5. Fill in the project ID in the job script

```bash
sed -i 's/<PROJECT_ID>/YOUR_REAL_PROJECT_ID/' jobs/phase1_check.pbs
grep '#PBS -P' jobs/phase1_check.pbs
```

(This edits a tracked file; don't commit it, or `git checkout jobs/phase1_check.pbs` afterwards.)

## 6. Submit (from the repo root)

```bash
cd ~/var-scale-audit
qsub jobs/phase1_check.pbs
```

1 GPU with a 1 h walltime on `normal` is routed to the `gdev` development queue (max 2 jobs per user).

## 7. Monitor

```bash
qstat -u $USER                                     # Q = queued, R = running, finished jobs disappear
tail -f outputs/phase1_check/<JOBID>/job.log       # live log once it is running (Ctrl-C to stop)
```

## 8. Read the results

```bash
cat outputs/phase1_check/<JOBID>/summary.txt
```

Paste `summary.txt` back. Everything else is in the same folder:

| File | Content |
|---|---|
| `job.log` | full log of the job |
| `nvidia-smi.txt`, `gpu.json` | GPU, driver, torch/CUDA versions, `CUDA_VISIBLE_DEVICES` |
| `runner_{var,dit}.{png,pt,json}` | runner output: image, raw tensor, settings/attention record |
| `ref_{var,dit}_{matched,asis}.{png,pt}` | official-notebook code path outputs |
| `equivalence.json` | tensor hashes and max differences |
| `timing_{var,dit}.json` | timings, peak VRAM, 1,000-image extrapolation |

Copy the images to your laptop (run on the laptop, via the jump host if needed):

```bash
scp '<USER>@aspire2antu.nscc.sg:~/var-scale-audit/outputs/phase1_check/<JOBID>/*.png' .
```

## If something goes wrong

- **`FATAL: torch.cuda.device_count() == 0`**: the summary's `DIAGNOSIS` lines and `CUDA_VISIBLE_DEVICES`
  say why. Paste the summary back; don't work around it in the job.
- **`qsub: Unknown project` / job rejected**: the `#PBS -P` line still has the placeholder or a wrong ID.
- **No `outputs/phase1_check/<JOBID>/`**: the job didn't start in the repo root. Check
  `outputs/phase1_check/phase1_check.o<JOBID>` for PBS's own log.
- **`EnvironmentNameNotFound: var-dit`**: step 4 didn't finish; re-run it.
