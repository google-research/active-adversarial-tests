
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] Clean evaluation"

epsilon="0.031372549"
nsteps="200"

# 2.5 / nsteps = 0.0125
stepsizevalues=(
  $(echo "print($epsilon)" | python3)
  $(echo "print($epsilon * 0.5)" | python3)
  $(echo "print($epsilon * 0.25)" | python3)
  $(echo "print($epsilon * 0.125)" | python3)
  $(echo "print($epsilon * 0.0625)" | python3)
  $(echo "print($epsilon * 0.0125)" | python3) # good value, equals eps/nsteps*2.5
  $(echo "print($epsilon * 0.003125)" | python3)
  $(echo "print($epsilon * 0.0015625)" | python3)
  $(echo "print($epsilon * 0.00078125)" | python3)
)

for stepsize in "${stepsizevalues[@]}"; do
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "$nsteps-step PGD (step size $stepsize)"
  advattack="norm=linf epsilon=$epsilon step_size=$stepsize n_steps=$nsteps"
  PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py \
    --decision-boundary-binarization="norm=linf epsilon=$epsilon n_inner_points=999 n_boundary_points=1 \
      optimizer=sklearn adversarial_attack_settings=\"$advattack\"" \
    --adversarial-attack="$advattack" \
    --no-logit-diff-loss \
    --no-logit-diff-loss \
    --n-samples=2048 \
    --batch-size=512 \
    --input=$1 \
    --classifier=${2:-networks.cifar_resnet18} \
    $3
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
done