#!/usr/bin/env bash
#SBATCH --job-name=aam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%A_%a.out
#SBATCH --error=runs/slurm/%A_%a.err

set -euo pipefail

MANIFEST=${1:?usage: sbatch --array=... aam_array.sbatch CONFIG_MANIFEST}
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [[ -z "$CONFIG" ]]; then
  echo "No config at array index $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi

mkdir -p runs/slurm
: "${AAM_VENV_ACTIVATE:=.venv/bin/activate}"
# shellcheck disable=SC1090
source "$AAM_VENV_ACTIVATE"

export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HF_HOME=${HF_HOME:-$PWD/.cache/huggingface}

python -m aamemory.cli run "$CONFIG"
