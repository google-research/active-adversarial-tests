#!/bin/bash

checkpoint="$1"

basecommand='
--n-samples=512
--batch-size=512
'
basecommand="${basecommand} --input=${checkpoint}"

if [ -z ${2+x} ]; then echo "Using default device"; else basecommand="$basecommand --device=$2"; fi

echo
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] Gaussian blur"
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py $basecommand \
  #--adversarial-attack="norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200" \
  #--logit-matching="n_steps=2000 step_size=0.0011372549" \
  #--classifier-input-gaussian-blur-stddev=0.5 \
  --decision-boundary-binarization="norm=linf epsilon=0.031372549 n_inner_points=49 n_boundary_points=10 \
      optimizer=sklearn adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200\""
  # --model-destruction="adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200\" n_probes=10 stddev=1.0" \
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

echo
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] JPEG compression (quality = 80)"
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py $basecommand \
  --adversarial-attack="norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200" \
  --logit-matching="n_steps=2000 step_size=0.0011372549" \
  --classifier-input-jpeg-quality=80 \
  --decision-boundary-binarization="norm=linf epsilon=0.031372549 n_inner_points=49 n_boundary_points=10 \
      optimizer=sklearn adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200\""
  # --model-destruction="adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200\" n_probes=10 stddev=1.0" \
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -