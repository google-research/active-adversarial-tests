#!/bin/bash

checkpoint="$1"

basecommand='
--n-samples=2048
--batch-size=512
'
basecommand="${basecommand} --input=${checkpoint}"

if [ -z ${2+x} ]; then echo "Using default device"; else basecommand="$basecommand --device=$2"; fi
epsilon=${3:-8}
attack=${4:-pgd}
echo "Attack: $attack"

if [[ "$attack" == "thermometer-lspgd" ]]; then
  nsteps=100
  stepsize=0.01
else
  nsteps=10 # 200
  stepsize=0.005 # 0.0011372549
fi
#--classifier="networks.differentiable_16_thermometer_encoding_cifar_resnet18" \
if [[ "$epsilon" == "8" ]]; then
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "[Linf, 8/255] Thermometer encoding (${checkpoint}) evaluation"
  PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py $basecommand \
    --classifier="networks.differentiable_16_thermometer_encoding_cifar_wideresnet344" \
    --decision-boundary-binarization="norm=linf epsilon=0.031372549 n_inner_points=49 n_boundary_points=1 \
      optimizer=sklearn adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=$stepsize n_steps=$nsteps attack=$attack\"" \
    --no-logit-diff-loss \
    --adversarial-attack="norm=linf epsilon=0.031372549 step_size=$stepsize n_steps=$nsteps attack=$attack"
    #--logit-matching="n_steps=2000 step_size=0.0011372549" \
    #    --model-destruction="adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200\" n_probes=10 stddev=1.0" \
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
elif [[ "$epsilon" == "6" ]]; then
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "[Linf, 6/255] Thermometer encoding (${checkpoint}) evaluation"
  PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py $basecommand \
    --classifier="networks.differentiable_16_thermometer_encoding_cifar_resnet18" \
    --decision-boundary-binarization="norm=linf epsilon=0.02352941176 n_inner_points=49 n_boundary_points=1 \
      optimizer=sklearn adversarial_attack_settings=\"norm=linf epsilon=0.02352941176 step_size=$stepsize n_steps=$nsteps attack=$attack\"" \
    --no-logit-diff-loss \
    --adversarial-attack="norm=linf epsilon=0.02352941176 step_size=$stepsize n_steps=$nsteps attack=$attack"
    #--logit-matching="n_steps=2000 step_size=0.0011372549" \
    # --model-destruction="adversarial_attack_settings=\"norm=linf epsilon=0.02352941176 step_size=0.0011372549 n_steps=200\" n_probes=10 stddev=1.0" \
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
elif [[ "$epsilon" == "4" ]]; then
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "[Linf, 4/255] Thermometer encoding (${checkpoint}) evaluation"
  PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py $basecommand \
    --classifier="networks.differentiable_16_thermometer_encoding_cifar_resnet18" \
    --decision-boundary-binarization="norm=linf epsilon=0.0156862745 n_inner_points=49 n_boundary_points=1 \
      optimizer=sklearn adversarial_attack_settings=\"norm=linf epsilon=0.0156862745 step_size=$stepsize n_steps=$nsteps attack=$attack\"" \
    --no-logit-diff-loss \
    --adversarial-attack="norm=linf epsilon=0.0156862745 step_size=$stepsize n_steps=$nsteps attack=$attack"
    #--logit-matching="n_steps=2000 step_size=0.0011372549" \
    # --model-destruction="adversarial_attack_settings=\"norm=linf epsilon=0.0156862745 step_size=0.0011372549 n_steps=200\" n_probes=10 stddev=1.0" \
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
else
  echo "Unknown epsilon value: $epsilon"
fi

