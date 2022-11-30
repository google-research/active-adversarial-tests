#!/bin/bash

checkpoint="$1"

basecommand='
--n-samples=512
--batch-size=512
'
basecommand="${basecommand} --input=${checkpoint}"

if [ -z ${2+x} ]; then echo "Using default device"; else basecommand="$basecommand --device=$2"; fi

#echo
#printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
#echo "[Linf, 8/255] Adding additive Gaussian noise as a pre-processing step"
#PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py $basecommand \
#  --adversarial-attack="norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200" \
#  --classifier-input-noise=0.1 \
#  --no-logit-diff \
#  --decision-boundary-binarization="norm=linf epsilon=0.031372549 n_inner_points=49 n_boundary_points=10 \
#      optimizer=sklearn adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200\""
## --logit-matching="n_steps=2000 step_size=0.0011372549" \
## --model-destruction="adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200\" n_probes=10 stddev=1.0" \
#printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

echo
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] Adding additive Gaussian noise as a pre-processing step"
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py $basecommand \
  --adversarial-attack="norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200 n_averages=10" \
  --classifier-input-noise=0.1 \
  --no-logit-diff \
  --decision-boundary-binarization="norm=linf epsilon=0.031372549 n_inner_points=49 n_boundary_points=10 \
      optimizer=sklearn adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200 n_averages=10\""
# --logit-matching="n_steps=2000 step_size=0.0011372549" \
# --model-destruction="adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200\" n_probes=10 stddev=1.0" \
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' '

echo
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] Adding additive Gaussian noise as a pre-processing step"
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py $basecommand \
  --adversarial-attack="norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200 n_averages=100" \
  --classifier-input-noise=0.1 \
  --no-logit-diff \
  --decision-boundary-binarization="norm=linf epsilon=0.031372549 n_inner_points=49 n_boundary_points=10 \
      optimizer=sklearn adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200 n_averages=100\""
# --logit-matching="n_steps=2000 step_size=0.0011372549" \
# --model-destruction="adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200\" n_probes=10 stddev=1.0" \
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

echo
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] Adding additive Gaussian noise to gradients"
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py $basecommand \
  --adversarial-attack="norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200" \
  --classifier-gradient-noise=0.1 \
  --no-logit-diff \
  --decision-boundary-binarization="norm=linf epsilon=0.031372549 n_inner_points=49 n_boundary_points=10 \
      optimizer=sklearn adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200\""
#--logit-matching="n_steps=2000 step_size=0.0011372549" \
# --model-destruction="adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200\" n_probes=10 stddev=1.0" \
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -


echo
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] Adding additive Gaussian noise to gradients"
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py $basecommand \
  --adversarial-attack="norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200 n_averages=10" \
  --classifier-gradient-noise=0.1 \
  --no-logit-diff \
  --decision-boundary-binarization="norm=linf epsilon=0.031372549 n_inner_points=49 n_boundary_points=10 \
      optimizer=sklearn adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200 n_averages=10\""
#--logit-matching="n_steps=2000 step_size=0.0011372549" \
# --model-destruction="adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200\" n_probes=10 stddev=1.0" \
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

echo
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] Adding additive Gaussian noise to gradients"
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py $basecommand \
  --adversarial-attack="norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200 n_averages=100" \
  --classifier-gradient-noise=0.1 \
  --no-logit-diff \
  --decision-boundary-binarization="norm=linf epsilon=0.031372549 n_inner_points=49 n_boundary_points=10 \
      optimizer=sklearn adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200 n_averages=100\""
#--logit-matching="n_steps=2000 step_size=0.0011372549" \
# --model-destruction="adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200\" n_probes=10 stddev=1.0" \
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
