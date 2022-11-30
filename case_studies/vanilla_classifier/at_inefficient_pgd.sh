#!/bin/bash

n_samples=${1:-2048}
echo "Using ${n_samples} samples"

checkpoint="checkpoints/rn50_madry_robustness_linf_at_8.pth"
basecommand="
--n-samples=${n_samples}
--batch-size=512
--classifier=networks.resnet50
--classifier-input-normalization
--input=${checkpoint}
--no-logit-diff-loss
"

if [ -z ${2+x} ]; then echo "Using default device"; else basecommand="$basecommand --device=$2"; fi

function eval {
  local bs="norm=linf epsilon=0.031372549 \
    n_inner_points=999 \
    n_boundary_points=1 \
    optimizer=sklearn \
    adversarial_attack_settings=\"ATTACK_SETTINGS\""

  PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py \
    $basecommand \
    --no-clean-evaluation \
    --decision-boundary-binarization="${bs/ATTACK_SETTINGS/$1}"
}

function evaladv {
  PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py \
    $basecommand \
    --no-clean-evaluation \
    --adversarial-attack="$1"
}

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "+Normal adversarial evaluation+"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] PGD, 200 steps, lr = 0.0011372549"
evaladv "norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "+Changing number of steps while keeping step size fixed+"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] PGD, 200 steps, lr = 0.0011372549"
eval "norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] PGD, 100 steps, lr = 0.0011372549"
eval "norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=100"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] PGD, 50 steps, lr = 0.0011372549"
eval "norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=50"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] PGD, 25 steps, lr = 0.0011372549"
eval "norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=25"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] PGD, 12 steps, lr = 0.0011372549"
eval "norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=12"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -


printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] PGD, 6 steps, lr = 0.0011372549"
eval "norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=6"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -


printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] PGD, 6 steps, lr = 0.0011372549"
eval "norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=3"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -


printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] PGD, 6 steps, lr = 0.0011372549"
eval "norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=1"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "+Changing number of steps and step size while maintaining their ratio+"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] PGD, 100 steps, lr = 0.00078431372"
eval "norm=linf epsilon=0.031372549 step_size=0.00078431372 n_steps=100"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] PGD, 50 steps, lr = 0.00156862745"
eval "norm=linf epsilon=0.031372549 step_size=0.00156862745 n_steps=50"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] PGD, 25 steps, lr = 0.0031372549"
eval "norm=linf epsilon=0.031372549 step_size=0.0031372549 n_steps=25"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] PGD, 12 steps, lr = 0.0065359477"
eval "norm=linf epsilon=0.031372549 step_size=0.0065359477 n_steps=12"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] PGD, 6 steps, lr = 0.01307189541"
eval "norm=linf epsilon=0.031372549 step_size=0.01307189541 n_steps=6"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] PGD, 3 steps, lr = 0.02614379083"
eval "norm=linf epsilon=0.031372549 step_size=0.02614379083 n_steps=3"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] PGD, 1 steps, lr = 0.031372549"
eval "norm=linf epsilon=0.031372549 step_size=0.031372549 n_steps=1"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -