#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH --time=00:45:00
#SBATCH --job-name=p4_pilot_var
#SBATCH --output=outputs/p4/slurm/output_%x_%j.out
#SBATCH --error=outputs/p4/slurm/error_%x_%j.err
# P4 Gate B pilot, VAR half, for TC2 (SLURM). Mirrors lanes/p4/jobs/pilot_var.pbs for NSCC.
# Submit from the repo root:
#     sbatch lanes/p4/jobs/pilot_var_tc2.sh
# then confirm:  squeue -u $USER
#
# 10 runs: the no-hook baseline first, then protect / damage / control at m = 1, 2, 3 scales
# (m/n = 0.1, 0.2, 0.3 of the 10 scales), all at lambda = 0, batch 16, one batch per run.
# Walltime: VAR-d20 fp32 is ~95 s per 1000 images, so 16 images is ~1.5 s; model load dominates at
# ~1-2 min per process. 10 processes -> ~20 min. 45 min leaves headroom for a slow filesystem.
set -uo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "submit with sbatch; no GPU work on the head node" >&2; exit 1; }
cd "$SLURM_SUBMIT_DIR"
[[ -f runner/generate.py ]] || { echo "submit from the repo root" >&2; exit 1; }

CONFIG="${VAR_CONFIG:-configs/var_d20.yaml}"
MANIFEST="${MANIFEST:-manifest/provisional_4x4.csv}"
BAND="${BAND:-middle}"
JOB="$SLURM_JOB_ID"
OUT="${OUT_ROOT:-outputs/p4}/pilot_var/$JOB"
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
python -m lanes.p4.pilot run --model var --band "$BAND" --config "$CONFIG" \
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
