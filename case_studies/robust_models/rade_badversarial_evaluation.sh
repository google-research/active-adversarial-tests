nsamples=${1:-512}
ninner=9999
nboundary=1

#kwargs=""
kwargs="--dbl-sample-from-corners"

echo "#samples: $nsamples"
echo "kwargs: $kwargs"

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "eps=8/255, $nboundary boundary, $ninner inner points PGD"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
advattack="norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200 attack=pgd"
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py \
  --n-samples=$nsamples \
  --batch-size=512 \
  --decision-boundary-binarization="norm=linf epsilon=0.031372549 n_inner_points=$ninner n_boundary_points=$nboundary optimizer=sklearn adversarial_attack_settings=\"$advattack\"" \
  --classifier=dm_networks.preactresnet_18 \
  --no-logit-diff-loss  \
  --input=checkpoints/rade_helper_rn18_linf.pt \
  --adversarial-attack="$advattack" \
  $kwargs

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "eps=8/255, $nboundary boundary, $ninner inner points APGD"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
advattack="norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200 attack=autopgd"
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py \
  --n-samples=$nsamples \
  --batch-size=512 \
  --decision-boundary-binarization="norm=linf epsilon=0.031372549 n_inner_points=$ninner n_boundary_points=$nboundary optimizer=sklearn adversarial_attack_settings=\"$advattack\"" \
  --classifier=dm_networks.preactresnet_18 \
  --no-logit-diff-loss  \
  --input=checkpoints/rade_helper_rn18_linf.pt \
  --adversarial-attack="$advattack" \
  $kwargs

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "eps=8/255, $nboundary boundary, $ninner inner points APGD+"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
advattack="norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200 attack=autopgd+"
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py \
  --n-samples=$nsamples \
  --batch-size=512 \
  --decision-boundary-binarization="norm=linf epsilon=0.031372549 n_inner_points=$ninner n_boundary_points=$nboundary optimizer=sklearn adversarial_attack_settings=\"$advattack\"" \
  --classifier=dm_networks.preactresnet_18 \
  --no-logit-diff-loss  \
  --input=checkpoints/rade_helper_rn18_linf.pt \
  --adversarial-attack="$advattack" \
  $kwargs

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "eps=8/255, $nboundary boundary, $ninner inner points FAB"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
advattack="norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200 attack=fab"
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py \
  --n-samples=$nsamples \
  --batch-size=512 \
  --decision-boundary-binarization="norm=linf epsilon=0.031372549 n_inner_points=$ninner n_boundary_points=$nboundary optimizer=sklearn adversarial_attack_settings=\"$advattack\"" \
  --classifier=dm_networks.preactresnet_18 \
  --no-logit-diff-loss  \
  --input=checkpoints/rade_helper_rn18_linf.pt \
  --adversarial-attack="$advattack" \
  $kwargs