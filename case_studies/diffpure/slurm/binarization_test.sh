#!/bin/bash

SLURMDIR="/home/bethge/rzimmermann/diffpure-slurm-logs/"
function execute_job {
  local arguments=$1

  echo "Using arguments: $arguments"
  #read -r -p "Are you sure? [enter N to cancel] " response
  #if [ "$response" == "N" ]; then
  #  exit 1
  #fi

  sbatch case_studies/diffpure/slurm/binarization_test_single.sh "${arguments}" || echo "error in $SLURMDIR/slog--$SLURM_JOB_ID.err on $SLURM_JOB_NODELIST" > $SLURMDIR/slog-common.err
}

mkdir -p "$SLURMDIR"


for startidx in {1..500}; do
  endidx=$(expr $startidx + 1)
  execute_job "1 1 $startidx $endidx"
done