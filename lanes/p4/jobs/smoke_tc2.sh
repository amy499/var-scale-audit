#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH --time=00:30:00
#SBATCH --job-name=p4_smoke
#SBATCH --output=outputs/p4/slurm/output_%x_%j.out
#SBATCH --error=outputs/p4/slurm/error_%x_%j.err
# P4 GPU smoke test for TC2 (SLURM). Mirrors lanes/p4/jobs/smoke.pbs for NSCC.
# Submit from the repo root:
#     sbatch lanes/p4/jobs/smoke_tc2.sh
# then confirm it queued:  squeue -u $USER
# A MISMATCH is never worked around: send P1 the output folder listing and job.log.
# Passes when both regression lines end in PASS.
set -uo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "submit with sbatch; no GPU work on the head node" >&2; exit 1; }
cd "$SLURM_SUBMIT_DIR"
[[ -f runner/generate.py ]] || { echo "submit from the repo root" >&2; exit 1; }

VAR_CONFIG="${VAR_CONFIG:-configs/var_d20.yaml}"
DIT_CONFIG="${DIT_CONFIG:-configs/dit_xl2_256.yaml}"
EXPECT_VAR="${EXPECT_VAR:-f16021ea827ab2ea38e70aab22b8e87d64a47f2c9bc8d6500fce72108cdef2c8}"   # VAR d20 fp32
EXPECT_DIT="${EXPECT_DIT:-2a3a0d6fbce9}"
JOB="$SLURM_JOB_ID"
OUT="${OUT_ROOT:-outputs/p4}/smoke/$JOB"
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
python -m runner.generate --config "$VAR_CONFIG" --class-id 207 --seed 0 --out "$OUT/var" --save-raw
python -m runner.generate --config "$DIT_CONFIG" --class-id 207 --seed 0 --out "$OUT/dit" --save-raw
python scripts/phase3/regression.py --out "$OUT/regression.json" \
    var "$OUT/var.pt" - "$EXPECT_VAR" \
    dit "$OUT/dit.pt" - "$EXPECT_DIT"
