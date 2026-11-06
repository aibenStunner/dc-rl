# SustainDC on Fir: Working Setup Guide

**Repository**: `dc-rl/` (SustainDC multi-agent RL simulator)
**Target**: Digital Research Alliance of Canada Fir cluster
**Python**: 3.11 (CUDA 12.6, H100 nodes)
**Active Experiment**: `team_tariff_targeted_safeguards` (Experiment 1)
**Algorithms**: HAPPO (`state_type: FP`)

This document records all steps that enable end-to-end experiment execution on Fir
without source code changes to SustainDC or HARL. It records debug steps, the
resolution path, and the exact commands that succeeded.

---

## 1. Environment prerequisites

### 1.1 Activate the Fir-specific virtual environment

The working project root on Fir is a symlinked path under the allocation
account's project area:

```text
/home/thegadri/links/projects/def-bboulet/thegadri
```

Activate the venv:

```bash
source /home/thegadri/links/projects/def-bboulet/thegadri/envs/sustaindc-py311-cu126/bin/activate
```

Verify the Python interpreter:

```bash
which python
# Should print: /home/thegadri/links/projects/def-bboulet/thegadri/envs/sustaindc-py311-cu126/bin/python
python --version
# Should print: Python 3.11.5
```

If you are not already in the venv, create it from scratch:

```bash
python -m venv /home/thegadri/links/projects/def-bboulet/thegadri/envs/sustaindc-py311-cu126
source /home/thegadri/links/projects/def-bboulet/thegadri/envs/sustaindc-py311-cu126/bin/activate
```

### 1.2 Set environment variables before any pip work

These must be cleared to avoid the `/localscratch` permission failure that
occurred earlier in this project:

```bash
unset PIP_PREFIX
unset PIP_TARGET
unset PYTHONUSERBASE
export PYTHONNOUSERSITE=1
```

---

## 2. Working dependency stack (Fir-compatible, no source changes)

The original repository `requirements.txt` **cannot** be used directly on Fir
Python 3.11 because it pins legacy packages unavailable in the Alliance
wheelhouse (`pygame==2.1.2`, `pymunk==6.2.1`, `jax==0.4.26`, etc.). The
following minimal, compatible stack has been verified to work:

```bash
# Core HARL/SustainDC stack (exact pins that work on Fir Python 3.11)
"$PYTHON" -m pip install --force-reinstall --no-deps \
    "Gymnasium==0.29.1" \
    "gymnasium-notices==0.0.1" \
    "pettingzoo==1.22.2" \
    "SuperSuit==3.7.0"

# Verify: must report zero broken requirements
"$PYTHON" -m pip check
# Expected: No broken requirements found.

# Optional: confirm each version
"$PYTHON" - <<'PY'
import gymnasium
import pettingzoo
import supersuit

print("Gymnasium:", gymnasium.__version__)
print("PettingZoo:", pettingzoo.__version__)
print("SuperSuit:", supersuit.__version__)

assert gymnasium.__version__ == "0.29.1"
assert pettingzoo.__version__ == "1.22.2"
assert supersuit.__version__ == "3.7.0"
PY
```

### 2.1 Important: Do NOT include these from original requirements.txt

The original full lockfile was the source of repeated install failures on Fir.
The following packages cause problems and must be **excluded** from any Fir
minimal manifest:

| Package | Why it fails |
|---|---|
| `jax==0.4.26` | Requires `ml-dtypes==0.4.0` → `numpy==2.0.0rc1` source build |
| `ray==2.24.0` | Not used by HARL training path |
| `keras==2.12.0` | Not used by active codebase |
| `tensorflow==2.12.0` | Not used |
| `absl-py==2.1.0` | Missing initially; requires separate install after import check |
| `rpds-py==0.18.0` | Killed during backend dependency install |
| `pygame==2.1.2` | Source build requires PNG/JPEG/PORTMIDI |
| `pymunk==6.2.1` | Not available in Fir Python 3.11 wheelhouse (need alternate Python) |
| `scipy==1.13.0` | Unavailable strict pin |

### 2.2 Verified package set

After extensive troubleshooting, this stack enables SustainDC + HARL + HAPPO
imports and a full training run on Fir:

```text
Gymnasium==0.29.1
gymnasium-notices==0.0.1
pettingzoo==1.22.2
SuperSuit==3.7.0
torch==2.6.0+cu126     # CUDA-enabled from official cu126 index
numpy                 # picked by pip from compatible upper bound
pandas
scipy                 # relaxed pin
h5py                  # relaxed pin
pvlib
psychrolib
wandb==0.19.8
absl-py==2.1.0
```

Install each individually; do not reinstall the original full lockfile.

**`pip check` will still report two advisories** for `pygame` and `pymunk`
(the versions SuperSuit 3.7.0 declares are not available on Fir Python
3.11). These are safe to ignore: neither package is imported by the active
HAPPO/SustainDC training path, and the training run completes normally.
See Section 7.5.

### 2.3 The exact upstream stack cannot be reproduced on Fir Python 3.11

This is a hard platform boundary, not a missing install command, and it is
worth recording because it will come up again if anyone tries to "fix" the
install:

- The upstream lockfile requires `pygame==2.1.2` and `pymunk==6.2.1`.
  Fir's wheelhouse offers only `pygame 2.5.2+computecanada` and
  `pymunk 7.0.1+computecanada` for Python 3.11, and the exact old
  versions have no binary wheel there. Building them from source fails on
  missing PNG/JPEG/PORTMIDI system libraries.
- `SuperSuit==3.8.1` is **not** a valid substitute: it imports
  `BaseParallelWrapper` from `pettingzoo.utils`, which does not exist in
  `pettingzoo==1.22.2`, so it fails with
  `ImportError: cannot import name 'BaseParallelWrapper'`.
- `pettingzoo>=1.23` is **not** a valid substitute either: it always
  returns `(observations, infos)` from `reset()`, while the SustainDC
  adapter is written against the pre-1.23 single-value contract.

The workable combination is therefore the legacy API stack
(`pettingzoo==1.22.2` + `SuperSuit==3.7.0` + `Gymnasium==0.29.1`) with the
two unavailable transitive dependencies tolerated. If the exact upstream
pins are ever required, the only source-free route is a **separate Python
3.10 virtual environment** — do not downgrade the working venv in place.

---

## 3. CUDA-enabled PyTorch on Fir

Install PyTorch with CUDA 12.6 support using the Alliance-provided index:

```bash
unset PIP_PREFIX
unset PIP_TARGET
unset PYTHONUSERBASE
export PYTHONNOUSERSITE=1

# Install CUDA PyTorch from the official index
"$PYTHON" -m pip install --no-deps \
    torch==2.6.0+cu126 \
    -f https://download.pytorch.org/whl/cu126

# Verify CUDA availability
"$PYTHON" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY
```

Expected output:

```text
torch: 2.6.0+cu126
cuda build: 12.6
cuda available: True
device count: 1
device name: NVIDIA H100 80GB HBM3
```

### 3.1 About the PIP_PREFIX fix

Early in this project, the job failed with:

```text
Permission denied: '/localscratch/thegadri.64635690.0'
```

Root cause: `PIP_PREFIX` environment variable pointing to an unwritable
scratch path. The fix is:

```bash
unset PIP_PREFIX
unset PIP_TARGET
unset PYTHONUSERBASE
export PYTHONNOUSERSITE=1
```

After clearing these, verify:

```bash
"$PYTHON" -m pip config list -v
# Should NOT show :env:.prefix='/localscratch/...'
```

### 3.2 About TMPDIR / /localscratch

Do not rely on `TMPDIR` changes to fix permission errors. The issue is the
`PIP_PREFIX` environment variable, not the temporary directory. The correct
fix is the `unset` commands above.

---

## 4. W&B online configuration

### 4.1 Initial interactive login

Once per user/account on Fir, log in interactively (inside an interactive
allocation or the login node):

```bash
unset PIP_PREFIX
unset PIP_TARGET
unset PYTHONUSERBASE
export PYTHONNOUSERSITE=1

python -m wandb login --relogin
```

This stores credentials in `~/.netrc`. Do NOT put W&B API keys in Slurm
scripts, YAML configs, or saved experiment directories.

### 4.2 Set W&B mode in Slurm scripts

The active jobs run W&B in **offline** mode (`WANDB_MODE=offline` in
`common.sh`, `--wandb-mode offline` on the command line) so that cluster
runs never block on network I/O and never risk losing a run to a
transient API failure. The runs are synced to the online project
afterwards.

Pass `--wandb-mode` and `--wandb-project` on the training command. These
are handled by `apply_wandb_overrides`, which runs *after* the generic
config merge, so they correctly win over the YAML defaults.

To switch a job to online tracking, change two things:

```bash
export WANDB_MODE=online
```

and on the command line:

```bash
    --wandb-mode online \
    --wandb-project sustaindc-thesis \
    --wandb-group <group> \
    --wandb-tags <tag1> <tag2> \
```

**Never pass bare `--mode` or `--enabled`.** Those leaf names exist in
both the algo and env configs and would also overwrite the Experiment 1
reward mode and the PV flag respectively (see Section 5.7).

Full example:

```bash
srun "$PYTHON" train_sustaindc.py \
  --algo happo \
  --exp_name experiment1-smoke \
  --cuda True \
  --n_rollout_threads 1 \
  --n_eval_rollout_threads 1 \
  --num_env_steps 2048 \
  --episode_length 128 \
  --use_eval False \
  --log_dir "$RESULTS_ROOT" \
  --wandb-mode offline \
  --wandb-project sustaindc-thesis \
  --wandb-group experiment1 \
  --wandb-tags fir smoke h100 experiment1
```

### 4.3 Where offline runs are stored, and how to sync them

Offline runs are written under the `--log_dir` you pass, following HARL's
run layout:

```text
$RESULTS_ROOT/sustaindc/<location>/<algo>/<exp_name>/seed-00001-<timestamp>/wandb/offline-run-<id>/
```

Find them with:

```bash
find "$RESULTS_ROOT" -type d -name 'offline-run-*'
```

Sync them to the online project from a machine with network access and
valid W&B credentials:

```bash
"$PYTHON" -m wandb sync /path/to/offline-run-<id>
# or every run under a directory:
"$PYTHON" -m wandb sync --sync-all /path/to/run-directory
```

Keep the offline run directory until the sync completes; it holds the
data needed for the upload.


### 4.4 W&B cache directories

To avoid the `Unable to find cache directory` warning, set these before the
training command:

```bash
export HOME="${HOME:-/home/thegadri}"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export WANDB_CACHE_DIR="$PROJECT_ROOT/.cache/wandb"
export WANDB_CONFIG_DIR="$PROJECT_ROOT/.config/wandb"
export WANDB_DATA_DIR="$PROJECT_ROOT/.local/share/wandb"

mkdir -p "$XDG_CACHE_HOME" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"
```

### 4.5 W&B credentials handling

Credentials are stored in `~/.netrc` after the initial `wandb login`.
Do NOT add API keys to Slurm scripts, saved configs, logs, or artifacts.
The `--wandb` flag enables online mode; `--wandb-mode disabled` turns it off.

---

## 5. Slurm job scripts

### 5.1 Shared environment script (`jobs/common.sh`)

Every job sources this file. It is the single source of truth for paths,
module loads, and the environment sanitisation that the Fir install requires.

```bash
#!/bin/bash
set -euo pipefail

module --force purge
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

export PROJECT_ROOT="$HOME/links/projects/def-bboulet/thegadri"
export THESIS_HOME="/home/thegadri/links/projects/def-bboulet/thegadri"
export REPO_ROOT="$THESIS_HOME/grad-thesis"
export DC_RL_ROOT="$REPO_ROOT/dc-rl"
export VENV="$THESIS_HOME/envs/sustaindc-py311-cu126"
export RESULTS_ROOT="$THESIS_HOME/sustaindc-results"
export PYTHON="$VENV/bin/python"

unset PYTHONHOME
unset PYTHONPATH
unset PIP_PREFIX
unset PIP_TARGET
unset PYTHONUSERBASE
export PYTHONNOUSERSITE=1

export WANDB_MODE=offline
export MPLBACKEND=Agg

export HOME="${HOME:-/home/thegadri}"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export WANDB_CACHE_DIR="$PROJECT_ROOT/.cache/wandb"
export WANDB_CONFIG_DIR="$PROJECT_ROOT/.config/wandb"
export WANDB_DATA_DIR="$PROJECT_ROOT/.local/share/wandb"

mkdir -p "$XDG_CACHE_HOME" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"

# Guard against SCRATCH being unset under `set -u`.
export PIP_CACHE_DIR="${SCRATCH:-/tmp}/pip-cache"

# One process is used per rollout environment. Prevent each process from
# creating an additional pool of BLAS/OpenMP threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

source "$VENV/bin/activate"

test -f "$DC_RL_ROOT/train_sustaindc.py"
test -d "$RESULTS_ROOT"

echo "Python executable:"
ls -l "$PYTHON"

echo "Python version:"
"$PYTHON" --version

echo "Python location from interpreter:"
"$PYTHON" - <<'PY'
import sys
print(sys.executable)
print(sys.version)
PY

cd "$DC_RL_ROOT"

echo "CUDA check:"
srun "$PYTHON" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY

echo "SustainDC import/config check:"
srun "$PYTHON" - <<'PY'
import torch
import numpy
import pandas
import scipy
import gymnasium
import pettingzoo
import supersuit
import pvlib
import psychrolib
import wandb
import absl

import sustaindc_env
from harl.utils.configs_tools import get_defaults_yaml_args

algo_args, env_args = get_defaults_yaml_args("happo", "sustaindc")

print("All required imports succeeded.")
print("Torch version:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("Configured state type:", env_args["state_type"])
print("Configured reward mode:", env_args["reward"]["mode"])
print("Configured location:", env_args["location"])
PY
```

Two details in this file are load-bearing and were both found by debugging:

- `PIP_CACHE_DIR="${SCRATCH:-/tmp}/pip-cache"` — under `set -u` an unset
  `SCRATCH` aborts the job before Python starts.
- `OMP_NUM_THREADS=1` — the rollout workers are separate OS processes, each
  of which would otherwise spawn its own BLAS thread pool and oversubscribe
  the node.

### 5.2 Smoke test job (`jobs/00-smoke.sbatch`)

```bash
#!/bin/bash
#SBATCH --account=def-bboulet
#SBATCH --job-name=sustaindc-smoke
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/thegadri/links/projects/def-bboulet/thegadri/sustaindc-results/slurm/%x-%j.out
#SBATCH --error=/home/thegadri/links/projects/def-bboulet/thegadri/sustaindc-results/slurm/%x-%j.err

set -euo pipefail
source /home/thegadri/links/projects/def-bboulet/thegadri/jobs/common.sh

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  "$PYTHON" -u train_sustaindc.py \
    --algo happo \
    --env sustaindc \
    --exp_name fir-smoke \
    --cuda True \
    --n_rollout_threads 1 \
    --n_eval_rollout_threads 1 \
    --num_env_steps 2048 \
    --episode_length 128 \
    --use_eval False \
    --use_render False \
    --log_dir "$RESULTS_ROOT" \
    --wandb-mode offline \
    --wandb-project sustaindc-thesis \
    --wandb-group smoke \
    --wandb-tags h100 smoke experiment1
```

**Always invoke Python by absolute path** (`"$PYTHON"`), never as bare
`python`. Inside a Slurm step, `python` is not on `PATH` and the job dies
with:

```text
error: execve(): python: No such file or directory
```

### 5.3 Throughput benchmark (`jobs/01-throughput.sbatch`)

Each configuration runs an identical amount of *per-worker* work (10 PPO
updates of 1024 steps), so the training-window wall time is a direct
contention measurement: if workers do not contend, every worker count
finishes in the same number of seconds.

```bash
#!/bin/bash
#SBATCH --account=def-bboulet
#SBATCH --job-name=sustaindc-bench
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/thegadri/links/projects/def-bboulet/thegadri/sustaindc-results/slurm/%x-%j.out
#SBATCH --error=/home/thegadri/links/projects/def-bboulet/thegadri/sustaindc-results/slurm/%x-%j.err

# cpus-per-task and mem above are PLACEHOLDERS. The real values are set by
# 00-submit-throughput.sh, which computes them per worker count and passes
# them via --cpus-per-task / --mem on the sbatch command line. Do not raise
# the header values above what the target node actually provides, or Slurm
# rejects the job with:
#   sbatch: error: CPU count per node can not be satisfied

set -euo pipefail
source /home/thegadri/links/projects/def-bboulet/thegadri/jobs/common.sh

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------
TOTAL_ENV_STEPS=163840
EPISODE_LENGTH=1024

# Passed in by 00-submit-throughput.sh.
N_ROLLOUT_THREADS="${THROUGHPUT_WORKERS:-8}"
CPUS="${THROUGHPUT_CPUS:-16}"
MEM_MB="${THROUGHPUT_MEM_MB:-16384}"

WORKER_UPDATES=$(( TOTAL_ENV_STEPS / EPISODE_LENGTH / N_ROLLOUT_THREADS ))
if (( WORKER_UPDATES < 1 )); then
  echo "ERROR: TOTAL_ENV_STEPS too small for ${N_ROLLOUT_THREADS} workers" >&2
  exit 1
fi
NUM_ENV_STEPS=$(( N_ROLLOUT_THREADS * WORKER_UPDATES * EPISODE_LENGTH ))

RUN_NAME="throughput2-w${N_ROLLOUT_THREADS}"
BENCHMARK_DIR="$RESULTS_ROOT/benchmarks"
mkdir -p "$BENCHMARK_DIR"

echo "workers=${N_ROLLOUT_THREADS} cpus=${CPUS} mem=${MEM_MB}MB"
echo "per-worker updates=${WORKER_UPDATES} total steps=${NUM_ENV_STEPS}"
echo "visible CPUs: $(nproc)"

# ---------------------------------------------------------------------------
# Work on a job-local copy so the repo is never modified.
# ---------------------------------------------------------------------------
WORKDIR="$SLURM_TMPDIR/dc-rl-$SLURM_JOB_ID"
mkdir -p "$WORKDIR"
rsync -a --exclude='results' --exclude='__pycache__' \
  "$DC_RL_ROOT/" "$WORKDIR/"
cd "$WORKDIR"

# ---------------------------------------------------------------------------
# The generic CLI override reaches nested config values by LEAF NAME, so
# --seed, --torch_threads and --failure_policy all work. Verify what
# actually took effect by echoing the resolved config.
# ---------------------------------------------------------------------------
python3 - <<'PY'
from harl.utils.configs_tools import get_defaults_yaml_args

algo_args, env_args = get_defaults_yaml_args("happo", "sustaindc")
print("seed.seed            =", algo_args["seed"]["seed"])
print("device.torch_threads =", algo_args["device"]["torch_threads"])
print("device.cuda          =", algo_args["device"]["cuda"])
print("train.n_rollout      =", algo_args["train"]["n_rollout_threads"])
print("train.num_env_steps  =", algo_args["train"]["num_env_steps"])
print("train.episode_length =", algo_args["train"]["episode_length"])
print("eval.use_eval        =", algo_args["eval"]["use_eval"])
print("logger.wandb.mode    =", algo_args["logger"]["wandb"]["mode"])
print("logger.wandb.project =", algo_args["logger"]["wandb"]["project"])
print("reward.mode          =", env_args["reward"]["mode"])
PY

# ---------------------------------------------------------------------------
# GPU monitoring
# ---------------------------------------------------------------------------
GPU_LOG="$BENCHMARK_DIR/${RUN_NAME}-${SLURM_JOB_ID}.gpu.csv"
nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
  --format=csv,noheader,nounits \
  -l 2 > "$GPU_LOG" &
GPU_MONITOR_PID=$!
sleep 1

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
TRAIN_LOG="$BENCHMARK_DIR/${RUN_NAME}-${SLURM_JOB_ID}.train.log"

START_TIME=$(date +%s)

set +e
srun --ntasks=1 --cpus-per-task="$CPUS" --cpu-bind=cores \
  "$PYTHON" -u train_sustaindc.py \
    --algo happo \
    --env sustaindc \
    --exp_name "$RUN_NAME" \
    --seed_specify True \
    --seed 101 \
    --cuda True \
    --torch_threads 1 \
    --n_rollout_threads "$N_ROLLOUT_THREADS" \
    --n_eval_rollout_threads 1 \
    --episode_length "$EPISODE_LENGTH" \
    --num_env_steps "$NUM_ENV_STEPS" \
    --use_eval False \
    --use_render False \
    --log_dir "$RESULTS_ROOT" \
    --wandb-mode offline \
    --wandb-project sustaindc-thesis \
    --wandb-group throughput \
    --wandb-tags h100 benchmark "workers-${N_ROLLOUT_THREADS}" \
    --failure_policy error \
  2>&1 | tee "$TRAIN_LOG"
TRAIN_RC=${PIPESTATUS[0]}
set -e

END_TIME=$(date +%s)
kill "$GPU_MONITOR_PID" 2>/dev/null || true

PEAK_RSS_KB=$(ps -o rss= -p $$ 2>/dev/null | awk '{print $1+0}')

printf "job_id=%s\tworkers=%s\tsteps=%s\telapsed_seconds=%s\tcpus=%s\tmem_mb=%s\tpeak_rss_kb=%s\trc=%s\n" \
  "$SLURM_JOB_ID" \
  "$N_ROLLOUT_THREADS" \
  "$NUM_ENV_STEPS" \
  "$(( END_TIME - START_TIME ))" \
  "$CPUS" \
  "$MEM_MB" \
  "$PEAK_RSS_KB" \
  "$TRAIN_RC" \
  > "$BENCHMARK_DIR/${RUN_NAME}-${SLURM_JOB_ID}.summary.tsv"

echo "=== SUMMARY ==="
cat "$BENCHMARK_DIR/${RUN_NAME}-${SLURM_JOB_ID}.summary.tsv"
exit "$TRAIN_RC"
```

### 5.4 Throughput submitter (`jobs/00-submit-throughput.sh`)

`--cpus-per-task` must scale with the worker count, so each configuration
needs its own Slurm task. This script queries the partition, computes what
actually fits, skips worker counts that do not, and reports what it did.

```bash
#!/bin/bash
# 00-submit-throughput.sh
# Submits one throughput task per worker count, each with its own
# --cpus-per-task and --mem. Run from the login node.
set -euo pipefail

PARTITION="${PARTITION:-compute}"
WORKER_COUNTS=(1 2 4 8 16 32)

# Cores per rollout worker, plus a small allowance for the parent process
# (HARL creates one subprocess per rollout env).
CORES_PER_WORKER="${CORES_PER_WORKER:-2}"
PARENT_CORES="${PARENT_CORES:-2}"

# Memory per worker in MiB, plus a fixed parent allowance.
MEM_PER_WORKER_MB="${MEM_PER_WORKER_MB:-2048}"
PARENT_MEM_MB="${PARENT_MEM_MB:-4096}"

# ---------------------------------------------------------------------------
# Discover the smallest node in the partition, so every request fits.
# ---------------------------------------------------------------------------
NODE_INFO=$(sinfo -N -p "$PARTITION" -h -o "%N %c %m" | sort -k2,2n | head -1)
if [[ -z "$NODE_INFO" ]]; then
  echo "ERROR: no nodes found in partition '${PARTITION}'" >&2
  exit 1
fi

read -r NODE_NAME NODE_CORES NODE_MEM_MB <<< "$NODE_INFO"

echo "Reference node: ${NODE_NAME}"
echo "  cores: ${NODE_CORES}"
echo "  memory: ${NODE_MEM_MB} MB"
echo

# Never request more than ~90% of the node's RealMemory.
MEM_CAP_MB=$(( NODE_MEM_MB * 9 / 10 ))

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH="$JOB_DIR/01-throughput.sbatch"

SUBMITTED=0
SKIPPED=()

for w in "${WORKER_COUNTS[@]}"; do
  cpus=$(( w * CORES_PER_WORKER + PARENT_CORES ))
  mem_mb=$(( w * MEM_PER_WORKER_MB + PARENT_MEM_MB ))

  if (( cpus > NODE_CORES )); then
    echo "SKIP workers=${w}: needs ${cpus} cores, node has ${NODE_CORES}"
    SKIPPED+=("$w")
    continue
  fi

  if (( mem_mb > MEM_CAP_MB )); then
    mem_mb="$MEM_CAP_MB"
    echo "NOTE workers=${w}: memory request capped to ${mem_mb} MB"
  fi

  echo "Submitting workers=${w} cpus=${cpus} mem=${mem_mb}MB"
  sbatch \
    --export=ALL,THROUGHPUT_WORKERS="$w",THROUGHPUT_CPUS="$cpus",THROUGHPUT_MEM_MB="$mem_mb" \
    --cpus-per-task="$cpus" \
    --mem="${mem_mb}M" \
    --job-name="bench-w${w}" \
    "$SBATCH"
  SUBMITTED=$(( SUBMITTED + 1 ))
done

echo
echo "Submitted ${SUBMITTED} task(s)."
if (( ${#SKIPPED[@]} > 0 )); then
  echo "Skipped worker counts: ${SKIPPED[*]}"
  echo "To fit them, lower CORES_PER_WORKER (currently ${CORES_PER_WORKER})"
  echo "or lower MEM_PER_WORKER_MB, or use a larger node class."
fi
```

Run it:

```bash
cd /home/thegadri/links/projects/def-bboulet/thegadri/jobs
bash 00-submit-throughput.sh
```

### 5.5 HAPPO baseline (`jobs/02-happo-baseline.sbatch`)

The worker count is derived from the cores actually allocated, so the same
script works on any node size. The `eval_interval` default is raised
because evaluation is the single largest serial cost in a run (see
Section 6.8).

```bash
#!/bin/bash
#SBATCH --account=def-bboulet
#SBATCH --job-name=sustaindc-happo
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=34
#SBATCH --mem=40G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/thegadri/links/projects/def-bboulet/thegadri/sustaindc-results/slurm/%x-%j.out
#SBATCH --error=/home/thegadri/links/projects/def-bboulet/thegadri/sustaindc-results/slurm/%x-%j.err

set -euo pipefail
source /home/thegadri/links/projects/def-bboulet/thegadri/jobs/common.sh

# --- Configuration -------------------------------------------------------
WORKERS="${WORKERS:-16}"
CPUS_PER_WORKER="${CPUS_PER_WORKER:-2}"
PARENT_CORES="${PARENT_CORES:-2}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-10000000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
SEED="${SEED:-101}"
# Facility capacity in MW. The scaling mode sizes the modelled hardware to
# this rating (see Section 6.6). Use 10 for the 10 MW scenario.
CAPACITY_MW="${CAPACITY_MW:-1}"

# Fall back to a safe worker count if the node cannot supply 2 cores/worker.
REQUESTED_CPUS=$(( WORKERS * CPUS_PER_WORKER + PARENT_CORES ))
if (( REQUESTED_CPUS > SLURM_CPUS_PER_TASK )); then
  echo "Reducing workers: need ${REQUESTED_CPUS} cores, have ${SLURM_CPUS_PER_TASK}"
  WORKERS=$(( (SLURM_CPUS_PER_TASK - PARENT_CORES) / CPUS_PER_WORKER ))
  if (( WORKERS < 1 )); then
    echo "ERROR: not enough cores for even 1 worker" >&2
    exit 1
  fi
  echo "Falling back to ${WORKERS} workers"
fi

WORKDIR="$SLURM_TMPDIR/dc-rl-$SLURM_JOB_ID"
mkdir -p "$WORKDIR"
rsync -a --exclude='results' --exclude='__pycache__' \
  "$DC_RL_ROOT/" "$WORKDIR/"
cd "$WORKDIR"

echo "workers=${WORKERS} cpus=${SLURM_CPUS_PER_TASK} steps=${NUM_ENV_STEPS} eval_interval=${EVAL_INTERVAL} capacity_mw=${CAPACITY_MW}"

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  "$PYTHON" -u train_sustaindc.py \
    --algo happo \
    --env sustaindc \
    --exp_name "happo-${CAPACITY_MW}mw-seed${SEED}" \
    --seed_specify True \
    --seed "$SEED" \
    --cuda True \
    --torch_threads 1 \
    --datacenter_capacity_mw "$CAPACITY_MW" \
    --n_rollout_threads "$WORKERS" \
    --n_eval_rollout_threads 1 \
    --episode_length 1024 \
    --num_env_steps "$NUM_ENV_STEPS" \
    --eval_interval "$EVAL_INTERVAL" \
    --use_eval True \
    --use_render False \
    --log_dir "$RESULTS_ROOT" \
    --wandb-mode offline \
    --wandb-project sustaindc-thesis \
    --wandb-group baseline \
    --wandb-tags h100 happo experiment1 "${CAPACITY_MW}mw" "seed-${SEED}"
```

**Capacity note.** The 10 MW configuration is roughly **3× slower per step**
than 1 MW (measured: 62 s versus 24 s for ten 30-day episodes), because it
simulates 15,800 servers instead of 1,701. Re-measure throughput before
committing to a sweep, and expect a `num_env_steps=1e7` run to take
noticeably longer at 10 MW.

To run the 10 MW baseline:

```bash
CAPACITY_MW=10 sbatch jobs/02-happo-baseline.sbatch
```

### 5.6 Multi-seed array (`jobs/03-seeds.sbatch`)

One job per seed, one H100 each, all seeds independent. This is the
recommended experiment layout: it avoids the intra-job contention
documented in Section 7 and finishes in the wall time of the longest
single run.

```bash
#!/bin/bash
#SBATCH --account=def-bboulet
#SBATCH --job-name=sustaindc-seeds
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=34
#SBATCH --mem=40G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-4
#SBATCH --output=/home/thegadri/links/projects/def-bboulet/thegadri/sustaindc-results/slurm/%x-%A_%a.out
#SBATCH --error=/home/thegadri/links/projects/def-bboulet/thegadri/sustaindc-results/slurm/%x-%A_%a.err

set -euo pipefail
source /home/thegadri/links/projects/def-bboulet/thegadri/jobs/common.sh

# --- Configuration -------------------------------------------------------
SEEDS=(101 102 103 104 105)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
ALGO="${ALGO:-happo}"

WORKERS="${WORKERS:-16}"
CPUS_PER_WORKER="${CPUS_PER_WORKER:-2}"
PARENT_CORES="${PARENT_CORES:-2}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-10000000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
# Facility capacity in MW; the scaling mode sizes the hardware to this rating.
CAPACITY_MW="${CAPACITY_MW:-1}"

# Fall back to a safe worker count if the node cannot supply 2 cores/worker.
REQUESTED_CPUS=$(( WORKERS * CPUS_PER_WORKER + PARENT_CORES ))
if (( REQUESTED_CPUS > SLURM_CPUS_PER_TASK )); then
  echo "Reducing workers: need ${REQUESTED_CPUS} cores, have ${SLURM_CPUS_PER_TASK}"
  WORKERS=$(( (SLURM_CPUS_PER_TASK - PARENT_CORES) / CPUS_PER_WORKER ))
  if (( WORKERS < 1 )); then
    echo "ERROR: not enough cores for even 1 worker" >&2
    exit 1
  fi
  echo "Falling back to ${WORKERS} workers"
fi

WORKDIR="$SLURM_TMPDIR/dc-rl-$SLURM_JOB_ID"
mkdir -p "$WORKDIR"
rsync -a --exclude='results' --exclude='__pycache__' \
  "$DC_RL_ROOT/" "$WORKDIR/"
cd "$WORKDIR"

echo "algo=${ALGO} seed=${SEED} workers=${WORKERS} cpus=${SLURM_CPUS_PER_TASK} steps=${NUM_ENV_STEPS} capacity_mw=${CAPACITY_MW}"

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  "$PYTHON" -u train_sustaindc.py \
    --algo "$ALGO" \
    --env sustaindc \
    --exp_name "${ALGO}-${CAPACITY_MW}mw-seed${SEED}" \
    --seed_specify True \
    --seed "$SEED" \
    --cuda True \
    --torch_threads 1 \
    --datacenter_capacity_mw "$CAPACITY_MW" \
    --n_rollout_threads "$WORKERS" \
    --n_eval_rollout_threads 1 \
    --episode_length 1024 \
    --num_env_steps "$NUM_ENV_STEPS" \
    --eval_interval "$EVAL_INTERVAL" \
    --use_eval True \
    --use_render False \
    --log_dir "$RESULTS_ROOT" \
    --wandb-mode offline \
    --wandb-project sustaindc-thesis \
    --wandb-group "$ALGO" \
    --wandb-tags h100 "$ALGO" experiment1 "seed-${SEED}"
```

### 5.7 How CLI config overrides actually resolve

`train_sustaindc.py` accepts ad-hoc `--key value` overrides. The merge
walker in [harl/utils/configs_tools.py](harl/utils/configs_tools.py)
recurses into nested dicts and matches a CLI key against a config leaf
**by name, anywhere in the tree**:

```python
def update_dict(dict1, dict2):
    for k in dict2:
        if type(dict2[k]) is dict:
            update_dict(dict1, dict2[k])      # recurses into nested dicts
        else:
            if k in dict1:
                dict2[k] = dict1[k]           # only if the LEAF NAME exists
```

Verified by running the resolver directly against the real config:

| Override | Resolves to | Effect |
|---|---|---|
| `--seed 101` | `algo.seed.seed` | works |
| `--seed_specify True` | `algo.seed.seed_specify` | works |
| `--torch_threads 1` | `algo.device.torch_threads` | works |
| `--cuda True` | `algo.device.cuda` | works |
| `--n_rollout_threads 16` | `algo.train.n_rollout_threads` | works |
| `--num_env_steps 12345` | `algo.train.num_env_steps` | works |
| `--episode_length 512` | `algo.train.episode_length` | works |
| `--use_eval False` | `algo.eval.use_eval` | works |
| `--use_render False` | `algo.render.use_render` | works |
| `--failure_policy error` | `algo.logger.wandb.failure_policy` | works |
| `--datacenter_capacity_mw 10` | `env.datacenter_capacity_mw` | works (top-level env key) |

Two rules follow:

**Dotted names do not work.** `--seed.seed 101` produces the key
`seed.seed`, which matches no leaf, so it is silently ignored. Always use
the bare leaf name.

**Two leaf names collide between the algo and env configs.** A bare
`--mode` or `--enabled` sets *both* matches:

| Leaf name | Matches | Danger |
|---|---|---|
| `mode` | `logger.wandb.mode` **and** `reward.mode` | `--mode online` would also overwrite the Experiment 1 reward mode |
| `enabled` | `logger.wandb.enabled` **and** `pv.enabled` | `--enabled True` would also switch PV on |

Never pass bare `--mode` or `--enabled`. Use `--wandb-mode` and
`--wandb-project` instead, which are handled by `apply_wandb_overrides`
([train_sustaindc.py:149-150](train_sustaindc.py#L149)) and therefore run
*after* the generic merge and win over the YAML.

Because the resolver matches by name, an override that "does nothing" is
almost always a typo or a colliding name — not a nesting problem. To
confirm what actually took effect, run the resolver the same way the
script does (see the verification block in Section 5.3).

---

## 6. Experiment types and how to run them

### 6.1 Smoke test (quick verification)

```bash
sbatch jobs/00-smoke.sbatch
```

Validates:

- H100 allocation
- CUDA visibility
- CUDA-enabled PyTorch
- SustainDC imports and config load
- Experiment 1 reward mode
- W&B offline tracking
- HAPPO GPU startup

Expected output at training start:

```text
choose to use gpu...
WARNING, using default values for chiller sizing...
share_observation_space:  [Box(-2.0, 2.0, (46,), float32), ...]
observation_space:  [Box(-2.0, 2.0, (40,), float32), ...]
action_space:  [Discrete(3), Discrete(3), Discrete(3)]
Start Training
```

### 6.2 Throughput benchmark

```bash
cd /home/thegadri/links/projects/def-bboulet/thegadri/jobs
bash 00-submit-throughput.sh
```

Each configuration runs an identical amount of *per-worker* work (10 PPO
updates of 1024 steps), so the training-window wall time measures
contention directly: if workers do not fight for cores, every worker count
finishes in the same number of seconds.

### 6.3 Measured throughput on Fir (H100, 16 cores)

Measured 2026-10-19 on the `compute` partition, `--cpus-per-task=16` for
every configuration.

| Workers | Train window | Per-worker rate | Aggregate | Per-worker step cost | GPU util (mean) | Peak GPU mem |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 52 s | 196.9 /s | 197 /s | 5.08 ms | 11.7% | 823 MiB |
| 2 | 52 s | 196.9 /s | 394 /s | 5.08 ms | 13.2% | 829 MiB |
| 4 | 52 s | 196.9 /s | 788 /s | 5.08 ms | 17.2% | 837 MiB |
| 8 | 62 s | 165.1 /s | 1,321 /s | 6.06 ms | 10.0% | 867 MiB |
| 16 | 92 s | 111.3 /s | 1,781 /s | 8.99 ms | 6.3% | 907 MiB |
| 32 | 172 s | 59.5 /s | 1,905 /s | 16.80 ms | 4.7% | 993 MiB |

**What this means:**

- **w1–w4 is contention-free.** Per-worker step cost is exactly 5.08 ms at
  every worker count up to 4, on 16 cores.
- **Contention starts at w8** and gets progressively worse: the train
  window grows 52 → 62 → 92 → 172 s.
- **The GPU is not the bottleneck.** Mean utilization *falls* from 11.7% to
  4.7% as workers increase, and peak memory never exceeds 993 MiB of
  81,559 MiB. The parent process's batched actor/critic forward passes are
  a fixed GPU cost per step; the workers' physics is CPU cost that scales
  with worker count, so the fixed GPU work is stretched out.
- **The limit is CPU oversubscription.** At w32 there are 33 processes
  (32 workers + parent) on 16 cores.

**Therefore: one H100 per job, never two. A second GPU buys nothing.**

**The fix is to scale `--cpus-per-task` with the worker count.** The
submitter in Section 5.4 does this (2 cores/worker + 2 for the parent).
If the node has 64 cores, w16 at 34 cores and w32 at 66 cores should both
land back near a 52 s train window, roughly doubling to tripling
aggregate throughput. If w16's train window stays near 92 s after scaling
cores, the ceiling is memory-bandwidth or L3 contention rather than core
count, and ~1,900 steps/s is the node limit.

### 6.4 HAPPO baseline

```bash
sbatch jobs/02-happo-baseline.sbatch
```

Use this as the reference run for comparing against ablations.

### 6.5 Multi-seed runs (five seeds)

```bash
sbatch --array=0-4 jobs/03-seeds.sbatch
```

Each seed is an independent run with its own H100. This layout is
**strongly preferred** over multiple seeds inside one job:

- Workers within a job share one policy, so a single job cannot produce
  independent seed replicates.
- Contention inside a job begins at w8 on 16 cores, so a job running many
  seeds sequentially pays that penalty repeatedly.
- Independent jobs are limited only by cluster concurrency, and the whole
  sweep finishes in the wall time of the longest single run.

Each seed produces an independent run saved under:

```text
sustaindc-results/sustaindc/montreal/happo/happo-seed<SEED>/seed-000<SEED>-<timestamp>/
```

Compare tariff cost, demand peaks, grid import, solar use, thermal
performance, battery throughput, and carbon across seeds.

### 6.6 Facility capacity and BESS sizing contract

`datacenter_capacity_mw` now has two modes, selected in
`harl/configs/envs_cfgs/sustaindc.yaml`:

```yaml
facility_sizing:
  mode: scaling     # or "legacy" to reproduce earlier runs
```

**`scaling` (active).** The value is a **design rating**: the modeled
facility's total electrical demand, IT plus modeled cooling, at full
utilization under a declared design envelope. The envelope is 100%
utilization, the CRAC setpoint range 15.0–21.6 °C, and the location's
design ambient. Calibration is a deterministic search over rack count and
rack occupancy, and the layout is rejected if it cannot land within **±2%**
of the target.

`dc_env.sizing_metadata` records the resolved design:

```text
mode, target_mw, achieved_mw, relative_error, rack_count, servers_per_rack,
server_count, utilization, crac_setpoint_c, min_crac_setpoint_c,
ambient_temperature_c, it_mw, tower_fan_mw, compressor_mw, hvac_mw,
excluded_crac_fan_mw, excluded_cw_pump_mw, excluded_ct_pump_mw, caveat
```

Measured calibration for the Montréal scenario:

| Target | Achieved | Error | Racks | Servers/rack | Total servers | BESS |
|---:|---:|---:|---:|---:|---:|---|
| 1 MW | 0.9985 MW | −0.149% | 9 | 189 | 1,701 | 1 MW / 2 MWh |
| 10 MW | 10.0384 MW | +0.384% | 79 | 200 | 15,800 | 10 MW / 20 MWh |

### 6.6.1 Verified scaling behaviour (10 paired 30-day episodes)

Ten episodes per capacity, matched seeds and start days, fixed actions,
built from the real active config. Reproduce with:

```bash
$PY tests/_scaling_paired_check.py
```

**Scaling ratios (10 MW / 1 MW):**

| Quantity | Ratio | 1 MW | 10 MW |
|---|---:|---:|---:|
| Facility energy | 9.87× | 415.7 MWh | 4,105.2 MWh |
| IT energy | 9.57× | 306.6 MWh | 2,934.9 MWh |
| HVAC energy | 10.73× | 109.1 MWh | 1,170.3 MWh |
| Grid import | 9.87× | 409.8 MWh | 4,046.0 MWh |
| PV generation | 10.00× | 5.9 MWh | 59.2 MWh |
| Peak demand | 9.89× | 791.8 kW | 7,834.4 kW |
| Tariff cost | 6.37× | 3.70M c | 23.55M c |

Energy scales essentially linearly. HVAC scales slightly *faster* than 10×
because cooling is non-linear in load. **Cost scales 6.37×, not 10×** — the
tariff structure absorbs the difference, and the 10 MW case is on Rate L.

**Battery dispatch** (charge in the morning, discharge in the evening):

| | 1 MW | 10 MW |
|---|---:|---:|
| Max charge power | 1,000.00 kW | 10,000.00 kW |
| Max discharge power | 787.46 kW | 7,792.74 kW |
| Charge energy | 55,095 kWh | 550,951 kWh |
| Discharge energy | 50,534 kWh | 505,343 kWh |
| SoC range | 0.00 → 0.91 | 0.00 → 0.91 |
| PV curtailed | 493.5 kWh | 4,950.7 kWh |

The rated power is hit exactly, confirming the rating is reachable and
enforced. Discharge is lower because it is limited by available facility
load, which is correct. The 8.3% charge-to-discharge gap is consistent with
90% round-trip efficiency plus the curtailment that simultaneous PV causes.

### 6.6.2 Three things to keep in mind

- **It is a design rating, not a cap.** The worst-case episode peak across
  six start days was 0.817 MW at 1 MW (81.9% of rating) and 8.171 MW at
  10 MW (81.4%), on the hottest day of the trace (32.9 °C ambient, day
  211). The gap is the workload trace averaging ~46% utilization, so the
  facility never approaches its nameplate. The 10 MW scenario therefore
  draws roughly 8 MW in practice.
- **The cooling model excludes CRAC fan and pump loads** from the meter.
  The excluded values are reported in the metadata. This is a documented
  limitation of the upstream model, not a validated site-power figure.
- **The design ambient must match the weather file.** Montréal now uses
  33.0 °C, the maximum dry bulb in
  `data/Weather/CAN_QC_Montreal-Trudeau.Intl.AP.716270_TMYx.2011-2025.epw`.
  Before this was fixed, Montréal fell through to the 50 °C default (a
  Phoenix value), which under-sized the cooling plant and made the 1 MW
  target unreachable.

One trap when writing your own probe scripts: build the environment from the
active config rather than hand-writing it. `EnvConfig.DEFAULT_CONFIG` has
`weather.randomize: True`, while the active YAML sets it to `false`. A
hand-written config therefore enables weather randomization and produces a
warmer, non-reproducible trace.

**BESS sizing.** The battery is a dispatchable grid-services BESS, not UPS
backup. Its bus-side rated power and nominal cell energy are explicit:

```yaml
battery:
  power_fraction: 1.0     # bus MW = power_fraction * facility MW
  duration_hours: 2.0     # cell MWh = rated MW * duration
  # or override directly:
  # rated_power_mw: 3.0
  # capacity_mwh: 6.0
```

So 1 MW facility → 1 MW / 2 MWh; 10 MW facility → 10 MW / 20 MWh. Because
discharge efficiency is below one, two hours of *nominal cell energy* does
not guarantee two full hours of bus output.

**Workload.** The trace stays a normalized utilization fraction. Identical
seeds and actions produce identical utilization and queue trajectories at
any capacity; only the installed compute behind that fraction changes. One
queue task is one percentage point of modeled compute capacity for one
15-minute interval — it is a model accounting unit, not a real job.

**`legacy` mode** reproduces the earlier behaviour exactly: fixed 20 racks
of 200 servers regardless of capacity, and the battery sized from the
estimated facility peak. Use it to reproduce saved checkpoints and
published runs.

### 6.7 Experiment 1 reward configuration

The active YAML config (`harl/configs/envs_cfgs/sustaindc.yaml`) sets:

```yaml
state_type: FP
reward:
  mode: team_tariff_targeted_safeguards
  scale: 0.01
  degradation_cost_weight: 0.0
  ls_overdue_penalty_weight: 0.50
  ls_dropped_penalty_weight: 0.25
  ls_backlog_penalty_weight: 0.20
  dc_thermal_penalty_weight: 0.75
  dc_constraint_penalty_weight: 0.40
battery:
  round_trip_efficiency: 0.90
  degradation_cost_c_per_kwh: 5.78
  initial_soc: 0.5
```

All experiments use `days_per_episode: 30`, `sample_whole_year: True`,
`episode_start_policy: valid_midnight`.

### 6.7 Sizing the training budget

The YAML default `num_env_steps: 2000000000` is unreachable in practice.
HARL computes the number of PPO updates as
`num_env_steps // episode_length // n_rollout_threads`
([on_policy_base_runner.py:237](harl/runners/on_policy_base_runner.py#L237)),
and wall-clock time follows from the measured aggregate throughput.

At the measured rates (w8 → 1,321 steps/s, w16 → 1,781 steps/s,
w32 → 1,905 steps/s):

| `num_env_steps` | w8 (1,321/s) | w16 (1,781/s) | PPO updates @ `episode_length=1024`, w8 |
|---:|---:|---:|---:|
| 1e6 | 12.6 min | 9.4 min | 122 |
| 1e7 | 2.1 h | 1.6 h | 1,220 |
| 2e7 | 4.2 h | 3.1 h | 2,441 |
| 1e8 | 21 h | 15.6 h | 12,207 |
| 2e9 (default) | 26 days | 16 days | 244,141 |

**Set `num_env_steps` explicitly in every job.** `1e7` is the recommended
unit for a baseline run (~2 h at w8, ~1,220 PPO updates); use `2e7` if
convergence needs it.

Planning number: use **1,300 steps/s** as the conservative throughput
estimate for scheduling, and **1,900 steps/s** as the optimistic ceiling
for a single node.

### 6.8 Evaluation is the largest serial cost in a run

This is the most easily missed cost in a HARL run on this simulator.

**Each evaluation event runs a full 30-day physical episode** — 2,880
environment steps at 15-minute resolution — and it runs *serially*,
blocking the training loop
([on_policy_base_runner.py:319-323](harl/runners/on_policy_base_runner.py#L319)).
The GPU is idle for the whole evaluation.

The eval loop terminates when
`eval_episode >= eval_episodes` (default 1), so the default
`n_eval_rollout_threads: 2` spawns a second evaluation environment that
is never needed.

The number of evaluation events is
`episodes // eval_interval`, where
`episodes = num_env_steps // episode_length // n_rollout_threads`. The
YAML default `eval_interval: 5` is aggressive.

Measured cost at the contention-free per-worker rate of 5.08 ms/step:

| `num_env_steps` | Workers | `eval_interval` | Eval events | Train | Eval | **Total** | Eval share |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e7 | 8 | 5 (default) | 244 | 2.10 h | 0.99 h | **3.09 h** | **32%** |
| 1e7 | 8 | 10 | 122 | 2.10 h | 0.50 h | 2.60 h | 19% |
| 1e7 | 8 | 20 | 61 | 2.10 h | 0.25 h | 2.35 h | 10% |
| 1e7 | 8 | 50 | 24 | 2.10 h | 0.10 h | 2.20 h | 4% |
| 1e7 | 16 | 5 (default) | 122 | 1.56 h | 0.50 h | **2.06 h** | **24%** |
| 1e7 | 16 | 20 | 30 | 1.56 h | 0.12 h | 1.68 h | 7% |
| 2e7 | 16 | 5 (default) | 244 | 3.12 h | 0.99 h | **4.11 h** | **24%** |
| 2e7 | 16 | 20 | 61 | 3.12 h | 0.25 h | 3.37 h | 7% |

**Two things follow:**

1. **Raise `eval_interval` to 20.** This recovers roughly 17–24
   percentage points of wall-clock — more than any other single change
   available. Over 1e7 steps you still get 61 evaluation points.

2. **Set `n_eval_rollout_threads` to 1.** The default of 2 spawns a
   redundant evaluation environment that exists for the whole run and
   adds contention. The eval loop only needs enough threads to satisfy
   `eval_episodes`.

**Checkpoint caveat.** The runner saves a model only when an evaluation
ties or beats the current best evaluation reward
([on_policy_base_runner.py:324-325](harl/runners/on_policy_base_runner.py#L324)).
Fewer evaluation events therefore means fewer checkpoint opportunities.
`eval_interval=20` still gives 61 chances over 1e7 steps, which is
ample; `eval_interval=50` would reduce them to 24 and is not recommended
for a baseline whose policy trajectory you want to preserve.

Note also that raising the worker count *reduces* the number of eval
events for a fixed `num_env_steps` (fewer PPO episodes), which is why
w16 costs proportionally less evaluation overhead than w8.

### 6.9 Reading the training log

HARL prints one pair of lines per logged episode
([sustaindc_logger.py:228-229](harl/envs/sustaindc/sustaindc_logger.py#L228)):

```text
Episode 105: Avg Net Energy=1234.567, Avg CO2=0.123, Water Usage=4.567
Tasks in Queue=1.234, Tasks Dropped=0.000
```

- `Avg Net Energy` — **cumulative grid import in kWh**, not a reward and
  not a per-step value. The logger accumulates
  `bat_total_energy_with_battery_KWh` across the episode and divides by
  the step count, so the printed number is the episode's total grid
  import divided by the number of logged steps
  ([sustaindc_logger.py:126,186](harl/envs/sustaindc/sustaindc_logger.py#L126)).
  It is **positive**, and **lower is better** — less grid import. Note
  that the underlying field is also exposed as `bat_grid_import_kwh`
  ([bat_env_fwd_view.py:88-89](../envs/bat_env_fwd_view.py#L88)),
  confirming it is grid import, not a net/reduced quantity.
- `Avg CO2` — average grid carbon intensity over the episode.
- `Water Usage` — cumulative cooling water use.
- `Tasks in Queue` / `Tasks Dropped` — workload service quality. These
  are the load-shifting safeguards; a run that lowers energy cost while
  dropping work is not a valid result.

These lines are emitted every `log_interval` episodes (default 5), and
they are the fastest way to confirm a run is healthy without opening
W&B. They are also a *proxy* for the reward, not the reward itself: the
Experiment 1 shared reward is `-scale * tariff_cost` with
`scale = 0.01` and `tariff_cost` in cents
([reward_creator.py:178-179](../utils/reward_creator.py#L178)), so
it also depends on the tariff structure, not only on energy. For the
headline economic result, read the tariff cost and demand-peak fields
from the evaluation exports rather than this console line.

---

## 7. Debugging checklist (what we learned)

### 7.1 Python executable mismatch

```text
which python -> /home/thegadri/projects/thegadri/envs/thesis/bin/python
which pip -> /cvmfs/.../tensorboard/2.20.0/bin/pip
```

**Fix**: Always use `python -m pip`, never bare `pip`. Activate the correct
venv first.

### 7.2 PIP_PREFIX environment variable

```text
:env:.prefix='/localscratch/thegadri.64635690.0'
```

**Fix**: Before any pip operations:

```bash
unset PIP_PREFIX
unset PIP_TARGET
unset PYTHONUSERBASE
export PYTHONNOUSERSITE=1
```

### 7.3 Missing `absl-py`

After `pip check` succeeds, import validation may fail:

```text
ModuleNotFoundError: No module named 'absl'
```

**Fix**: Install:

```bash
python -m pip install absl-py==2.1.0
```

### 7.4 PettingZoo/PySuperSuit API mismatch on Fir Python 3.11

```text
ValueError: too many values to unpack (expected 2)
```

at `pettingzoo/utils/conversions.py`, line 306.

**Fix**: The SustainDC source deliberately uses the pre-1.23 PettingZoo
contract (`obs = env.reset(...)` single return). The compatible stack is:

```text
Gymnasium==0.29.1
pettingzoo==1.22.2
SuperSuit==3.7.0
```

Do not upgrade PettingZoo to `>=1.23` or SuperSuit to newer than `3.7.0`
without also modifying SustainDC source code.

### 7.5 Exact old pygame/pymunk unavailable

`SuperSuit==3.7.0` requires `pygame==2.1.2` and `pymunk==6.2.1`, which are
not available as Python 3.11 binary wheels on Fir's wheelhouse. The
compatible approach is to keep `SuperSuit==3.7.0` but **accept** the
`pip check` warning about those two packages — they are not imported by the
active HAPPO/SustainDC training path, and the `pip check` advisory does not
block execution.

See Section 2.3 for the full boundary analysis, including why
`SuperSuit==3.8.1` and `pettingzoo>=1.23` are not valid substitutes.

### 7.6 W&B `wandb-core` permission error

```text
RuntimeError: W&B initialization failed: PermissionError: [Errno 13]
Permission denied: '.../site-packages/wandb/bin/wandb-core'
```

The W&B package cannot execute its bundled binary from inside the venv.
Fix the executable bit:

```bash
chmod +x "$VENV"/lib/python3.11/site-packages/wandb/bin/wandb-core
```

If it still fails, reinstall W&B in the venv (with `PIP_PREFIX` cleared):

```bash
unset PIP_PREFIX; unset PIP_TARGET; unset PYTHONUSERBASE
export PYTHONNOUSERSITE=1
"$PYTHON" -m pip install --force-reinstall --no-deps wandb==0.19.8
chmod +x "$VENV"/lib/python3.11/site-packages/wandb/bin/wandb-core
```

Note the path in the error may be under
`/home/thegadri/links/projects/def-bboulet/...` rather than
`/home/thegadri/projects/...` — these are the same directory reached
through a symlink.

### 7.7 Slurm rejects the job: "CPU count per node can not be satisfied"

```text
sbatch: error: CPU count per node can not be satisfied
sbatch: error: Batch job submission failed: Requested node configuration is not available
```

The requested `--cpus-per-task` (or `--mem`) exceeds what the node offers.
Find the real limits:

```bash
sinfo -N -p compute -h -o "%N %c %m %G" | head -20
NODE=$(sinfo -N -p compute -h -o "%N" | head -1)
scontrol show node "$NODE" | grep -E "CoreCount|CPUAlloc|RealMemory|AllocMem"
```

`%c` is cores per node, `%m` is memory per node in MB, and `RealMemory` is
usually slightly below the nominal `%m` value — request about 90% of it.

Keep the `#SBATCH --cpus-per-task` header small in job scripts and let the
submitter override it on the command line. `00-submit-throughput.sh` does
this and also skips worker counts that cannot fit.

### 7.8 Throughput is CPU-bound, not GPU-bound

The most important operational finding. See Section 6.3 for the measured
table. Summary of symptoms and fixes:

| Symptom | Cause | Fix |
|---|---|---|
| Training window grows with worker count | CPU oversubscription (one process per rollout env) | Scale `--cpus-per-task` with worker count |
| GPU utilization falls as workers increase | Fixed GPU work stretched over longer CPU time | Same as above; the GPU is not the limit |
| GPU memory barely used (~1 GB of 80 GB) | Single small policy network | Expected; one H100 per job is correct |
| `num_env_steps: 2000000000` never finishes | YAML default is ~26 days at measured rates | Set `num_env_steps` explicitly (see Section 6.7) |

### 7.9 Import validation after dependency setup

Run the following after installing the dependency stack and before submitting
any Slurm job:

```bash
cd "$DC_RL_ROOT"

"$PYTHON" - <<'PY'
import torch
import numpy
import pandas
import scipy
import gymnasium
import pettingzoo
import supersuit
import pvlib
import psychrolib
import wandb
import absl

import sustaindc_env
from harl.utils.configs_tools import get_defaults_yaml_args

algo_args, env_args = get_defaults_yaml_args("happo", "sustaindc")

print("All required imports succeeded.")
print("Torch version:", torch.__version__)
print("Configured state type:", env_args["state_type"])
print("Configured reward mode:", env_args["reward"]["mode"])
print("Configured location:", env_args["location"])
PY
```

Expected output:

```text
All required imports succeeded.
Torch version: 2.6.0+cu126
Configured state type: FP
Configured reward mode: team_tariff_targeted_safeguards
Configured location: montreal
```

### 7.10 CUDA check inside Slurm job

The H100 must show `CUDA_VISIBLE_DEVICES=0` and `nvidia-smi` must list the
GPU. The smoke test includes both checks:

```bash
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
```

If `nvidia-smi` cannot run inside the job, check `--gpus-per-node=h100:1`
and `module load cuda/12.6`.

---

## 8. Storage locations

| Path | Purpose |
|---|---|
| `$PROJECT` / `~/projects/thegadri/` | Repo, venv, scripts, long-lived results if quota permits |
| `$SCRATCH` | pip cache/temp or large disposable intermediates |
| `$SLURM_TMPDIR` | Job-local copied source tree/temp runtime files |

---

## 9. What must NOT be done on Fir

| Anti-pattern | Why |
|---|---|
| `pip install -r requirements.txt` | Original lockfile causes repeated failures; incompatible pins |
| Downgrading Python to 3.10 in the active venv | Use a separate venv if Python 3.10 is needed |
| `SuperSuit>=3.8` with `pettingzoo==1.22.2` | Import-incompatible (`BaseParallelWrapper`) |
| `pettingzoo>=1.23` without source changes | SustainDC adapter expects pre-1.23 reset contract |
| Installing `pygame==2.1.2` from source | Fails on Fir Python 3.11; not imported by active training path |
| Putting W&B API keys in Slurm scripts or YAML configs | Violates security policy; credentials in `~/.netrc` only |
| `TMPDIR` / `PIP_PREFIX` workarounds that aren't the explicit unset | The PIP_PREFIX issue is fixed by the three `unset` commands |
| Using the full original `requirements.txt` as-is | Causes grpcio, h5py, numpy, pandas, pettingzoo, scipy, ray failures |
| Claiming contiguous utility-bill reconstruction | The simulator explicitly states episodes are independent accounting periods |
| Requesting two GPUs for one training job | GPU is 5–18% utilized; the second GPU buys nothing |
| Raising `--cpus-per-task` in the script header above node capacity | Slurm rejects the job before it starts; let the submitter set it |
| Running many seeds inside a single job | Workers share one policy, so a job cannot produce independent seed replicates |
| Leaving `num_env_steps` at the YAML default | 2e9 steps is ~26 days at measured throughput |
| Passing bare `--mode` or `--enabled` | These leaf names collide; they also set the reward mode and the PV flag |
| Leaving `eval_interval` at the default of 5 | Evaluation is serial and runs a full 30-day episode; it costs ~24–32% of wall-clock (Section 6.8) |
| Leaving `n_eval_rollout_threads` at the default of 2 | The eval loop only needs `eval_episodes` threads; the second one is contention for the whole run |
| Reading `Avg Net Energy` as a reward | It is cumulative grid import in kWh, positive, lower is better (Section 6.9) |

---

## 10. Verification before first submit

After installing dependencies and before the first Slurm submit, run the
full validation block (Section 7.9). If that passes, the smoke script
(Section 5.2) should succeed and proceed to training startup rather than
failing at environment reset.

If the smoke script fails at the `reset()` step, the error will most likely
be the PettingZoo API mismatch (Section 7.4) — verify the exact dependency
stack is correct before investigating source changes.

If a job dies immediately with
`execve(): python: No such file or directory`, the script is calling bare
`python` instead of `"$PYTHON"` (Section 7.1).

If a job dies before producing any output, check for an unset `SCRATCH`
variable aborting the script under `set -u` (Section 5.1).

---

*Document last updated: 2026-10-19*