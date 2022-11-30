
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[Linf, 8/255] Clean evaluation"

epsilon="0.031372549"
nsteps="200"

nstepsvalues=( 1 5 10 15 20)

for nsteps in "${nstepsvalues[@]}"; do
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "$nsteps-step PGD"
  advattack="norm=linf epsilon=$epsilon step_size=0.0011372549 n_steps=$nsteps"
  PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py \
    --decision-boundary-binarization="norm=linf epsilon=$epsilon n_inner_points=999 n_boundary_points=1 \
        optimizer=sklearn adversarial_attack_settings=\"$advattack\"" \
    --adversarial-attack="$advattack" \
    --no-clean-evaluation \
    --no-logit-diff-loss \
    --n-samples=2048 \
    --batch-size=512 \
    --input=$1 \
    --classifier=${2:-networks.cifar_resnet18} \
    $3
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
done