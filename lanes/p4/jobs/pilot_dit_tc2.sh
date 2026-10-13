#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH --time=02:00:00
#SBATCH --job-name=p4_pilot_dit
#SBATCH --output=outputs/p4/slurm/output_%x_%j.out
#SBATCH --error=outputs/p4/slurm/error_%x_%j.err
# P4 Gate B pilot, DiT half, for TC2 (SLURM). Mirrors lanes/p4/jobs/pilot_dit.pbs for NSCC.
# Submit from the repo root:
#     sbatch lanes/p4/jobs/pilot_dit_tc2.sh
# then confirm:  squeue -u $USER
#
# 10 runs: the no-hook baseline first, then protect / damage / control at m = 25, 50, 75 steps
# (m/n = 0.1, 0.2, 0.3 of the 250 steps) via --skip-timesteps, batch 16, one batch per run.
# Walltime: DiT is 31.9 s per batch of 16, so one batch is ~32 s at m = 0 and less for arms that
# skip steps; model load dominates at ~2 min per process. 10 processes -> ~25 min.
# 2 h leaves headroom. DiT dominates the whole lane's budget.
set -uo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "submit with sbatch; no GPU work on the head node" >&2; exit 1; }
cd "$SLURM_SUBMIT_DIR"
[[ -f runner/generate.py ]] || { echo "submit from the repo root" >&2; exit 1; }

CONFIG="${DIT_CONFIG:-configs/dit_xl2_256.yaml}"
MANIFEST="${MANIFEST:-manifest/provisional_4x4.csv}"
BAND="${BAND:-middle}"
JOB="$SLURM_JOB_ID"
OUT="${OUT_ROOT:-outputs/p4}/pilot_dit/$JOB"
mkdir -p "$OUT" outputs/p4/slurm
exec > >(tee -a "$OUT/job.log") 2>&1

set +u
module load anaconda
eval "$(conda shell.bash hook)"
conda activate var-dit
set -u
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1

nvidia-smi
python scripts/check_third_party.py
git log -1 --oneline
python -m lanes.p4.pilot run --model dit --band "$BAND" --config "$CONFIG" \
    --manifest "$MANIFEST" --out-root "$OUT"
PILOT_RC=$?
[[ $PILOT_RC -eq 0 ]] || echo "PILOT INCOMPLETE (exit $PILOT_RC) - verifying what exists, then failing"
python lanes/p4/verify_pairing.py "$OUT"
PAIR_RC=$?

if [[ $PILOT_RC -eq 0 && $PAIR_RC -eq 0 ]]; then
    echo "P4 PILOT COMPLETE AND PAIRED"
    exit 0
fi
echo "P4 PILOT FAILED (pilot exit $PILOT_RC, pairing exit $PAIR_RC) - do not use these runs"
exit 1
