#!/bin/bash

_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

function prepare_conda_env() {
  # the python version to use
  local python_version=${1:-3.9}
  shift
  # the conda env name
  local env_name=${1:-mole}
  shift

  echo ">>> Preparing conda environment \"${env_name}\", python_version=${python_version}"

  # Preparation
  set -e
  eval "$(conda shell.bash hook)"
  conda env remove -y --name $env_name || true
  conda create --name $env_name python=$python_version pip -y
  conda activate $env_name
  pip install --upgrade pip

  # Install libraries
  # TODO: (optional) install PyTorch if you use it, preferably using conda; check https://pytorch.org/get-started/locally/ for the latest command
  pip install -r requirements.txt
  # pip install -e .[dev]
}

function prepare_local_env() {

  echo "About to install the necessary Python libraries in the current pip environment."
  read -p "Enter \"yes\" to confirm: " confirm

  if [ "$confirm" = "yes" ]; then
    pip install -r requirements.txt
    exit 0
  fi

  echo "Aborting installation."
}

function main() {
  local option=${1:-local}
  shift

  case $option in
  conda)
    prepare_conda_env $@
    ;;
  local)
    prepare_local_env
    ;;
  *)
    echo "Usage: $(basename $0): Installs the necessary Python libraries..."
    echo "local(default): in the current python environment, it's recommended to create and activate a Python virtualenv first."
    echo "conda: in a conda environment called "mole"."
    echo "nvcc is required to install and use flash-attn."
    exit 1
    ;;
  esac
}

main $@
