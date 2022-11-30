#!/bin/bash

checkpoint="$1"

basecommand='
--n-samples=2048
--batch-size=512
'
basecommand="${basecommand} --input=${checkpoint}"

if [ -z ${2+x} ]; then echo "Using default device"; else basecommand="$basecommand --device=$2"; fi

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] Clean evaluation"
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py $basecommand \
  --batch-size=2048 \
  --decision-boundary-binarization="norm=linf epsilon=0.031372549 n_inner_points=1 n_boundary_points=999 \
      optimizer=sklearn adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=1\""
  #--adversarial-attack="norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200"
  # --model-destruction="adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200\" n_probes=10 stddev=1.0" \
exit
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
#--adversarial-attack="norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200" \
#  --logit-matching="n_steps=2000 step_size=0.0011372549" \

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 6/255] Clean evaluation"
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py $basecommand \
  --decision-boundary-binarization="norm=linf epsilon=0.02352941176 n_inner_points=49 n_boundary_points=1 \
      optimizer=sklearn adversarial_attack_settings=\"norm=linf epsilon=0.02352941176 step_size=0.0011372549 n_steps=200\""
  #--adversarial-attack="norm=linf epsilon=0.02352941176 step_size=0.0011372549 n_steps=200"
  # --model-destruction="adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200\" n_probes=10 stddev=1.0" \
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

#--adversarial-attack="norm=linf epsilon=0.02352941176 step_size=0.0011372549 n_steps=200" \
#  --logit-matching="n_steps=2000 step_size=0.0011372549" \

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 4/255] Clean evaluation"
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py $basecommand \
  --decision-boundary-binarization="norm=linf epsilon=0.0156862745 n_inner_points=49 n_boundary_points=1 \
      optimizer=sklearn adversarial_attack_settings=\"norm=linf epsilon=0.0156862745 step_size=0.0011372549 n_steps=200\""
  #--adversarial-attack="norm=linf epsilon=0.0156862745 step_size=0.0011372549 n_steps=200" \
  # --model-destruction="adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200\" n_probes=10 stddev=1.0" \
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

#--adversarial-attack="norm=linf epsilon=0.0156862745 step_size=0.0011372549 n_steps=200" \
#  --logit-matching="n_steps=2000 step_size=0.0011372549" \

