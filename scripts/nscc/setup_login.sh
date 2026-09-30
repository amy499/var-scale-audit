#!/bin/bash
# NSCC ASPIRE 2A, LOGIN node only (no GPU work). Safe to re-run.
#
#   bash scripts/nscc/setup_login.sh <PROJECT_ID>
#
# 1. submodules at their pinned commits
# 2. checkpoints/ -> /home/project/<PROJECT_ID>/var-scale-audit/checkpoints
#    (GPFS project space: never purged. /scratch purges files unused for 30 days.)
#    If /home/project/<PROJECT_ID> is missing, falls back to a real checkpoints/ dir in the repo.
# 3. conda env "var-dit" from environment.yml (created, or updated only if the file changed)
# 4. download + verify checkpoints (resumes if the login node kills it; just re-run)
# 5. third_party/ check
set -eo pipefail

PROJECT_ID="${1:?usage: bash scripts/nscc/setup_login.sh <PROJECT_ID>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME=var-dit
PROJECT_DIR="/home/project/${PROJECT_ID}"
CKPT_STORE="${PROJECT_DIR}/var-scale-audit/checkpoints"

if [[ -n "${PBS_JOBID:-}" ]]; then
    echo "ERROR: this is the login-node setup script; don't run it inside a PBS job." >&2
    exit 1
fi
cd "$REPO"

echo "== [1/5] submodules"
git submodule update --init

if [[ -d "$PROJECT_DIR" ]]; then
    echo "== [2/5] checkpoint storage: $CKPT_STORE"
    mkdir -p "$CKPT_STORE"
    if [[ -L checkpoints ]]; then
        if [[ "$(readlink -f checkpoints)" != "$(readlink -f "$CKPT_STORE")" ]]; then
            echo "ERROR: checkpoints -> $(readlink -f checkpoints), expected $CKPT_STORE. Remove the link and re-run." >&2
            exit 1
        fi
    elif [[ -d checkpoints ]]; then
        if [[ -n "$(ls -A checkpoints)" ]]; then
            echo "ERROR: checkpoints/ is a non-empty directory. Move its files to $CKPT_STORE, delete it, re-run." >&2
            exit 1
        fi
        rmdir checkpoints
    fi
    [[ -L checkpoints ]] || ln -s "$CKPT_STORE" checkpoints
    echo "   checkpoints -> $(readlink -f checkpoints)"
else
    # No project space: keep checkpoints in the repo (home quota) instead.
    echo "== [2/5] checkpoint storage: $REPO/checkpoints (fallback)"
    echo "   $PROJECT_DIR not found; using a real checkpoints/ directory in the repo."
    if [[ -L checkpoints ]]; then
        echo "ERROR: checkpoints is a symlink to $(readlink checkpoints), but $PROJECT_DIR is missing. Remove the link and re-run." >&2
        exit 1
    fi
    mkdir -p checkpoints
    echo "   free space where checkpoints/ lives (need about 6 GB):"
    df -h "$REPO/checkpoints"
fi
mkdir -p outputs/phase1_check   # PBS writes its job log here

echo "== [3/5] conda env '$ENV_NAME'"
module load miniforge3
eval "$(conda shell.bash hook)"
# Package cache on scratch (purgeable cache; saves the 50 GB home quota).
SCRATCH_DIR="${HOME/\/home\/users\//\/scratch\/users\/}"
if [[ "$SCRATCH_DIR" != "$HOME" && -d "$SCRATCH_DIR" ]]; then
    export CONDA_PKGS_DIRS="$SCRATCH_DIR/conda-pkgs"
fi
# Login nodes have no NVIDIA driver; tell the solver which CUDA the GPU nodes provide.
export CONDA_OVERRIDE_CUDA=11.8
yml_hash="$(sha256sum environment.yml | cut -d' ' -f1)"
env_prefix="$(conda env list | awk -v n="$ENV_NAME" '$1 == n {print $NF}')"
if [[ -n "$env_prefix" ]]; then
    stamp="$env_prefix/.environment.yml.sha256"
    if [[ -f "$stamp" && "$(cat "$stamp")" == "$yml_hash" ]]; then
        echo "   up to date with environment.yml"
    else
        conda env update -n "$ENV_NAME" -f environment.yml
    fi
else
    conda env create -f environment.yml
fi
conda activate "$ENV_NAME"
echo "$yml_hash" > "$CONDA_PREFIX/.environment.yml.sha256"
# Import check only (CPU): torch must be a CUDA build.
python - <<'EOF'
import torch, torchvision, timm, diffusers, yaml
print(f"   torch {torch.__version__} (CUDA build {torch.version.cuda}), torchvision {torchvision.__version__}, "
      f"timm {timm.__version__}, diffusers {diffusers.__version__}")
if torch.version.cuda is None:
    raise SystemExit("ERROR: torch is a CPU-only build; environment.yml solve went wrong")
EOF

echo "== [4/5] checkpoints"
python scripts/download_checkpoints.py

echo "== [5/5] third_party"
python scripts/check_third_party.py

echo
echo "Setup complete."
echo "Next: job files in jobs/ contain P1's project (#PBS -P personal-gura0001); don't edit it."
echo "Teammates: submit from the repo root with your own project, qsub -P <your-project-id> <job file>,"
echo "then confirm with 'qstat -f <JOBID> | grep -i project' that your project is shown."
echo "(A qsub option overriding a #PBS line is standard PBS behaviour, but untested on NSCC by us.)"
