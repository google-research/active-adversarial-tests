nsamples=${1:-512}
epsilon=${2:-8}

kwargs=""
kwargs="--sample-from-corners"

echo "#samples: $nsamples"
echo "epsilon: $epsilon"
echo "kwargs: $kwargs"

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary point, 999 inner (Original attack)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python case_studies/error_correcting_codes/binarization_test.py \
  --n-samples=$nsamples \
  --eps=$epsilon \
  $kwargs

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary point, 999 inner (Adaptive attack)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python case_studies/error_correcting_codes/binarization_test.py \
  --n-samples=$nsamples \
  --adaptive-attack \
  --pgd-n-steps=200 \
  --pgd-step-size=0.50 \
  --eps=$epsilon \
  $kwargs
