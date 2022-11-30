nsamples=${1:-2048}
epsilon=${2:-8}

# kwargs=""
kwargs="--sample-from-corners"


echo "Epsilon: $epsilon"
echo "#samples: $nsamples"
echo "kwargs: $kwargs"

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary point, 999 inner (Original attack)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python case_studies/thermometer/original/binarization_test.py \
  --cifar-path=data/cifar-10-batches-py/test_batch \
  --n-samples=$nsamples \
  --n-boundary=1 \
  --n-inner=999 \
  --decision-boundary-closeness=0.999 \
  --epsilon=$epsilon \
  --attack=original \
  $kwargs

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary point, 999 inner (Modified attack)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python case_studies/thermometer/original/binarization_test.py \
  --cifar-path=data/cifar-10-batches-py/test_batch \
  --n-samples=$nsamples \
  --n-boundary=1 \
  --n-inner=999 \
  --decision-boundary-closeness=0.999 \
  --epsilon=$epsilon \
  --attack=modified \
  $kwargs

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary point, 999 inner (Adaptive attack)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python case_studies/thermometer/original/binarization_test.py \
  --cifar-path=data/cifar-10-batches-py/test_batch \
  --n-samples=$nsamples \
  --n-boundary=1 \
  --n-inner=999 \
  --decision-boundary-closeness=0.999 \
  --epsilon=$epsilon \
  --attack=adaptive \
  $kwargs