#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --job-name=p4_sweep_var
#SBATCH --output=outputs/p4/slurm/output_%x_%j.out
#SBATCH --error=outputs/p4/slurm/error_%x_%j.err
# P4 sweep, VAR half, for TC2 (SLURM): the fallback twin of lanes/p4/jobs/sweep_var.pbs.
# Use it only if NSCC fails or queues past 12 h, and only after lanes/p4/jobs/smoke_tc2.sh passes on TC2.
# Submit from the repo root:
#     sbatch lanes/p4/jobs/sweep_var_tc2.sh
#     SEED=20261070 sbatch --export=ALL lanes/p4/jobs/sweep_var_tc2.sh                      # Phase E seed 2
#     CONFIG=configs/var_d24.yaml OUT_TAG=d24 sbatch --export=ALL lanes/p4/jobs/sweep_var_tc2.sh
# A TC2 root is never mixed with NSCC runs: each job writes its own root, and machine.txt records which.
set -uo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "submit with sbatch; no GPU work on the head node" >&2; exit 1; }
cd "$SLURM_SUBMIT_DIR"
[[ -f runner/generate.py ]] || { echo "submit from the repo root" >&2; exit 1; }
mkdir -p outputs/p4/slurm

set +u
module load anaconda
eval "$(conda shell.bash hook)"
conda activate var-dit
set -u
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1

MODEL=var JOB="$SLURM_JOB_ID" MACHINE=tc2
source lanes/p4/jobs/sweep_body.sh
