#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=p4_score
#SBATCH --output=outputs/p4/slurm/output_%x_%j.out
#SBATCH --error=outputs/p4/slurm/error_%x_%j.err
# P4 scoring for TC2 (SLURM). Mirrors lanes/p4/jobs/score.pbs for NSCC.
# Submit from the repo root:
#     ROOTS="outputs/p4/full/var/seed1/<JOBID>" sbatch --export=ALL lanes/p4/jobs/score_tc2.sh
# Score only roots that were generated on TC2: one results root never mixes machines.
set -uo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "submit with sbatch; no GPU work on the head node" >&2; exit 1; }
cd "$SLURM_SUBMIT_DIR"
[[ -f runner/generate.py ]] || { echo "submit from the repo root" >&2; exit 1; }
[[ -n "${ROOTS:-}" ]] || { echo "set ROOTS to one or more P4 roots" >&2; exit 1; }
mkdir -p outputs/p4/slurm

set +u
module load anaconda
eval "$(conda shell.bash hook)"
conda activate var-dit
set -u
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 KEEP_RAW="${KEEP_RAW:-0}"

nvidia-smi
git log -1 --oneline
# shellcheck disable=SC2086  # ROOTS is a space-separated list on purpose
python -m lanes.p4.score $ROOTS
exit $?
