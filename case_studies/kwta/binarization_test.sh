nsamples=${1:-512}
epsilon=${2:-8}
attack=${3:-pgd}
sparsity=${4:-1}
echo "Attack: $attack"
echo "Epsilon: $epsilon"
echo "#samples: $nsamples"
echo "Sparsity: $sparsity"

if [[ "$sparsity" == "1" ]]; then
  checkpoint="checkpoints/kwta_sparse_resnet18_0.1.pth"
elif [[ "$sparsity" == "2" ]]; then
  checkpoint="checkpoints/kwta_sparse_resnet18_0.2.pth"
else
  echo "invalid sparsity"
  exit -1
fi

if [[ "$attack" == "kwta" ]]; then
  kwargs="--no-ce-loss"
else
  kwargs="--no-logit-diff-loss"
fi

kwargs="$kwargs --dbl-sample-from-corners"

echo "kwargs: $kwargs"

if [[ "$epsilon" == "8" ]]; then
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "[Linf, 8/255] kWTA (${checkpoint}) evaluation"
  PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py $basecommand \
    --classifier="case_studies.kwta.resnet.sparse_resnet18_0$sparsity"  \
    --decision-boundary-binarization="norm=linf epsilon=0.031372549 n_inner_points=999 n_boundary_points=1 \
      optimizer=sklearn adversarial_attack_settings=\"norm=linf epsilon=0.031372549 step_size=0.0011372549 n_steps=200 attack=$attack\"" \
    --input=$checkpoint \
    $kwargs
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
elif [[ "$epsilon" == "6" ]]; then
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "[Linf, 6/255] kWTA (${checkpoint}) evaluation"
  PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py $basecommand \
    --classifier="case_studies.kwta.resnet.sparse_resnet18_0$sparsity"  \
    --decision-boundary-binarization="norm=linf epsilon=0.02352941176 n_inner_points=999 n_boundary_points=1 \
      optimizer=sklearn adversarial_attack_settings=\"norm=linf epsilon=0.02352941176 step_size=0.0011372549 n_steps=200 attack=$attack\"" \
    --input=$checkpoint \
    $kwargs
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
elif [[ "$epsilon" == "4" ]]; then
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "[Linf, 4/255] kWTA (${checkpoint}) evaluation"
  PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py $basecommand \
    --classifier="case_studies.kwta.resnet.sparse_resnet18_0$sparsity"  \
    --decision-boundary-binarization="norm=linf epsilon=0.0156862745 n_inner_points=999 n_boundary_points=1 \
      optimizer=sklearn adversarial_attack_settings=\"norm=linf epsilon=0.0156862745 step_size=0.0011372549 n_steps=200 attack=$attack\"" \
    --input=$checkpoint \
    $kwargs
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
else
  echo "Unknown epsilon value: $epsilon"
fi