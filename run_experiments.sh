#!/usr/bin/env bash
set -euo pipefail

GRAPHS=(BA SBlow SBhigh lattice)
RECURRENT_COMPONENTS=(dcrnn gcnconv)
# PS=(0.05 0.1 0.2 1)
PS=(0.2 0.8)

for g in "${GRAPHS[@]}"; do
  for p in "${PS[@]}"; do
    for rc in "${RECURRENT_COMPONENTS[@]}"; do
      echo ">>> Running graph=${g}, recurrent_component=${rc}, probability_of_selecting_nodes=${p}"
      # Optional: name your W&B run per combo
      export WANDB_NAME="g=${g} rc=${rc} p=${p}"

      # Your training/runner script here:
      python main.py logger=wandb_oxmlgh\
        "dataset.graph=${g}" \
        "model.model.recurrent_component=${rc}" \
        "model.model.probability_of_selecting_nodes=${p}"
      done
  done
done