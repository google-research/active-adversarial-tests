nsamples=${1:-512}
epsilon=${2:-8}

if [[ "$3" == "deterministic" ]]; then
  deterministic="--deterministic"
else
  deterministic=""
fi


# kwargs=""
kwargs="--sample-from-corners"

echo "#samples: $nsamples"
echo "Deterministic?: $deterministic"
echo "Epsilon: $epsilon"
echo "kwargs: $kwargs"

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary point, 999 inner (Original attack)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$(pwd) python case_studies/mixup/binarization_test.py \
  --eps=$epsilon \
  --n-samples=$nsamples \
  --attack=pgd \
  $deterministic \
  $kwargs

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary point, 999 inner (Adaptive attack)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$(pwd) python case_studies/mixup/binarization_test.py \
  --eps=$epsilon \
  --n-samples=$nsamples \
  --attack=adaptive-pgd \
  $deterministic \
  $kwargs