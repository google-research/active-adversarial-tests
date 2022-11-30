nsamples=${1:-512}
direction=${2:-normal}
echo "Evaluating on $nsamples samples"
echo "Direction: $direction"
multinoise=${3:-}


if [[ "$direction" == "normal" ]]; then
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "1 boundary point, 999 inner"
  echo "Original attack"
  echo "Normal test"
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

  TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python case_studies/odds/original/binarization_test.py \
    --n-samples=$nsamples \
    --n-boundary=1 \
    --n-inner=999 \
    --dont-verify-training-data \
    --attack=original \
    $multinoise

  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "1 boundary point, 999 inner"
  echo "Adaptive attack"
  echo "Normal test"
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

  TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python case_studies/odds/original/binarization_test.py \
    --n-samples=$nsamples \
    --n-boundary=1 \
    --n-inner=999 \
    --dont-verify-training-data \
    --attack=adaptive \
     $multinoise

  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "1 boundary point, 999 inner"
  echo "Adaptive attack w/ EOT"
  echo "Normal test"
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

  TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python case_studies/odds/original/binarization_test.py \
    --n-samples=$nsamples \
    --n-boundary=1 \
    --n-inner=999 \
    --dont-verify-training-data \
    --attack=adaptive-eot \
     $multinoise
else
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "1 boundary point, 999 inner"
  echo "Original attack"
  echo "Inverted test"
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

  TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python case_studies/odds/original/binarization_test.py \
    --n-samples=$nsamples \
    --n-boundary=1 \
    --n-inner=999 \
    --dont-verify-training-data \
    --attack=original \
    --inverted-test \
     $multinoise

  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "1 boundary point, 999 inner"
  echo "Adaptive attack"
  echo "Inverted test"
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

  TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python case_studies/odds/original/binarization_test.py \
    --n-samples=$nsamples \
    --n-boundary=1 \
    --n-inner=999 \
    --dont-verify-training-data \
    --attack=adaptive \
    --inverted-test \
     $multinoise

  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "1 boundary point, 999 inner"
  echo "Adaptive attack w/ EOT"
  echo "Inverted test"
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

  TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python case_studies/odds/original/binarization_test.py \
    --n-samples=$nsamples \
    --n-boundary=1 \
    --n-inner=999 \
    --dont-verify-training-data \
    --attack=adaptive-eot \
    --inverted-test \
     $multinoise
fi