#!/bin/bash
#SBATCH --account=def-bboulet
#SBATCH --job-name=bat-only-happo
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=34
#SBATCH --mem=40G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/thegadri/links/projects/def-bboulet/thegadri/sustaindc-results/slurm/%x-%j.out
#SBATCH --error=/home/thegadri/links/projects/def-bboulet/thegadri/sustaindc-results/slurm/%x-%j.err

# One-agent battery HAPPO launch (protocol section 5.6, Amendment A3).
#
#   SEED=1 sbatch jobs/04-battery-only.sbatch                  # first launch, YAML defaults
#   SEED=2 sbatch jobs/04-battery-only.sbatch                  # only after seed 1 meets the criteria
#   SEED=1 LR=1e-4 sbatch jobs/04-battery-only.sbatch          # menu: lr and critic_lr move together
#   SEED=1 ENTROPY=0.01 sbatch jobs/04-battery-only.sbatch     # menu: entropy_coef
#
# Budget: at most 6 launches and 600 updates in total. Each launch is 100 updates.
# The seed comes from the launch config. This script never passes --seed, --enabled or --mode.
# W&B stays disabled, and no credential is read or written.

set -euo pipefail
source /home/thegadri/links/projects/def-bboulet/thegadri/jobs/common.sh

SEED="${SEED:?set SEED to 1, 2 or 3}"
LR="${LR:-}"
ENTROPY="${ENTROPY:-}"

case "$SEED" in 1|2|3) ;; *) echo "ERROR: SEED must be 1, 2 or 3, got '$SEED'" >&2; exit 1 ;; esac
case "$LR" in ""|3e-4|1e-4) ;; *) echo "ERROR: LR must be 3e-4 or 1e-4, got '$LR'" >&2; exit 1 ;; esac
case "$ENTROPY" in ""|0.1|0.01) ;; *) echo "ERROR: ENTROPY must be 0.1 or 0.01, got '$ENTROPY'" >&2; exit 1 ;; esac

CONFIG_NAME="battery-only-seed${SEED}.json"
CONFIG_SRC="$DC_RL_ROOT/results/launch-configs/$CONFIG_NAME"
test -f "$CONFIG_SRC" || { echo "ERROR: missing $CONFIG_SRC" >&2; exit 1; }

# Name the run after what actually changed, so launches never share a directory.
EXP_NAME="battery-only-happo-s${SEED}"
[ -n "$LR" ] && EXP_NAME="${EXP_NAME}-lr${LR}"
[ -n "$ENTROPY" ] && EXP_NAME="${EXP_NAME}-ent${ENTROPY}"

WORKDIR="$SLURM_TMPDIR/dc-rl-$SLURM_JOB_ID"
mkdir -p "$WORKDIR"
# The sync leaves out results/, so the launch config is copied in on its own.
rsync -a --exclude='results' --exclude='__pycache__' "$DC_RL_ROOT/" "$WORKDIR/"
mkdir -p "$WORKDIR/results/launch-configs"
cp "$CONFIG_SRC" "$WORKDIR/results/launch-configs/$CONFIG_NAME"
cd "$WORKDIR"

# Refuse to start a launch whose config breaks the protocol's fixed arithmetic.
"$PYTHON" - "$SEED" "results/launch-configs/$CONFIG_NAME" <<'PY'
import json
import sys

seed, path = int(sys.argv[1]), sys.argv[2]
config = json.load(open(path, encoding="utf-8"))
algo, env = config["algo_args"], config["env_args"]
train = algo["train"]
updates = train["num_env_steps"] // train["episode_length"] // train["n_rollout_threads"]
assert algo["seed"]["seed_specify"] is True and algo["seed"]["seed"] == seed, "config seed differs from SEED"
assert env["agents"] == ["agent_bat"], f"agents are {env['agents']}"
assert algo["eval"]["fixed_month_validation"]["enabled"] is True, "fixed-month validation is off"
assert algo["logger"]["wandb"]["mode"] == "disabled", "W&B must stay disabled"
assert updates > 0 and updates % train["eval_interval"] == 0, f"{updates} updates, eval_interval {train['eval_interval']}"
print(f"config ok: seed={seed} updates={updates} eval_interval={train['eval_interval']} "
      f"workers={train['n_rollout_threads']} lr={algo['model']['lr']} entropy_coef={algo['algo']['entropy_coef']}")
PY

EXTRA=()
if [ -n "$LR" ]; then EXTRA+=(--lr "$LR" --critic_lr "$LR"); fi
if [ -n "$ENTROPY" ]; then EXTRA+=(--entropy_coef "$ENTROPY"); fi

echo "exp_name=${EXP_NAME} seed=${SEED} lr=${LR:-default} entropy=${ENTROPY:-default} cpus=${SLURM_CPUS_PER_TASK}"

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  "$PYTHON" -u train_sustaindc.py \
    --load_config "results/launch-configs/$CONFIG_NAME" \
    --exp_name "$EXP_NAME" \
    --log_dir "$RESULTS_ROOT" \
    ${EXTRA[@]+"${EXTRA[@]}"}
