#!/bin/bash
# Shared body of the P4 sweep jobs (sweep_var.pbs, sweep_dit.pbs and their _tc2.sh twins).
# Not submitted on its own: each wrapper sets MODEL, JOB and MACHINE, activates the env, and sources it
# from the repo root. One sweep = generate every run, check pairing, score, then one status line.
#
# Environment overrides (qsub -v / sbatch --export):
#   CONFIG    model config            default: configs/var_d20.yaml or configs/dit_xl2_256.yaml
#   SEED      control-placement seed  default: 20261006 (lanes/p4/arms.py CONTROL_SEED)
#             Phase E second seed: 20261070. Only the control arm depends on it.
#   OUT_TAG   names the root          default: seed<SEED>; use d24 for the VAR d24 repeat
#   STAGES    comma-separated stage set: a confirmation run (lanes/p4/shortlist.py) instead of the
#             full sweep; the root goes under outputs/p4/confirm/
#   MANIFEST  default: manifest/frozen_v1_1.csv
#   KEEP_RAW  1 keeps the .pt tensors (default 0: dropped once scored, the 50 GB quota rule)
#
# Roots: outputs/p4/full/<model>/<OUT_TAG>/<JOB> or outputs/p4/confirm/<model>/<OUT_TAG>/<JOB>.
# One root never holds runs from two machines (MACHINE is recorded in machine.txt).

case "$MODEL" in
    var) DEFAULT_CONFIG=configs/var_d20.yaml ;;
    dit) DEFAULT_CONFIG=configs/dit_xl2_256.yaml ;;
    *) echo "MODEL must be var or dit, got '$MODEL'" >&2; exit 1 ;;
esac
CONFIG="${CONFIG:-$DEFAULT_CONFIG}"
SEED="${SEED:-20261006}"
OUT_TAG="${OUT_TAG:-seed$SEED}"
MANIFEST="${MANIFEST:-manifest/frozen_v1_1.csv}"
STAGES="${STAGES:-}"
export KEEP_RAW="${KEEP_RAW:-0}"

if [[ -n "$STAGES" ]]; then
    KIND=confirm
    ACTION=(run --stages "$STAGES")
else
    KIND=full
    ACTION=(run-full)
fi
OUT="${OUT_ROOT:-outputs/p4}/$KIND/$MODEL/$OUT_TAG/$JOB"
if [[ -e "$OUT/pilot_plan.json" ]]; then
    echo "$OUT already holds a sweep; refusing to mix two jobs in one root" >&2
    exit 1
fi
mkdir -p "$OUT"
echo "$MACHINE" > "$OUT/machine.txt"
exec > >(tee -a "$OUT/job.log") 2>&1

echo "P4 $KIND sweep: model $MODEL  config $CONFIG  seed $SEED  stages ${STAGES:-<all bands>}"
echo "root $OUT  machine $MACHINE  keep raw $KEEP_RAW"
nvidia-smi
python scripts/check_third_party.py
git log -1 --oneline              # run.json records the submodules, not this repo's branch

# Exit codes are captured rather than ignored, so a sweep that lost arms partway cannot finish green.
python -m lanes.p4.pilot "${ACTION[@]}" --model "$MODEL" --config "$CONFIG" \
    --manifest "$MANIFEST" --seed "$SEED" --out-root "$OUT"
PILOT_RC=$?
[[ $PILOT_RC -eq 0 ]] || echo "SWEEP INCOMPLETE (exit $PILOT_RC) - verifying and scoring what exists, then failing"
python lanes/p4/verify_pairing.py "$OUT"
PAIR_RC=$?
# Scored even when pairing failed: the scores are needed to diagnose it, and the raw tensors would
# otherwise fill the quota. A failed root is still never used for a result.
python -m lanes.p4.score "$OUT"
SCORE_RC=$?

echo "root $OUT"
if [[ $PILOT_RC -eq 0 && $PAIR_RC -eq 0 && $SCORE_RC -eq 0 ]]; then
    echo "P4 SWEEP COMPLETE, PAIRED AND SCORED"
    exit 0
fi
echo "P4 SWEEP FAILED (sweep exit $PILOT_RC, pairing exit $PAIR_RC, scoring exit $SCORE_RC) - do not use these runs"
exit 1
