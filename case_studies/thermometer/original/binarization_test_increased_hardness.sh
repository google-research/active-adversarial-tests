attack=${1:-original}
nsamples=${2:-2048}
mode=$3

if [ -z ${mode+x} ]; then
  echo "No hardness mode specified. Choose from: ninner, gap"
  exit -1
fi

echo "Attack: $attack, #Samples: $nsamples"
echo ""

if [[ "$mode" == "ninner" ]]; then
  for ninner in 49 99 199 299 399 499 599 699 799 899 999 1999; do
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo "1 boundary point, $ninner inner"
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python case_studies/thermometer/original/binarization_test.py \
      --cifar-path=data/cifar-10-batches-py/test_batch \
      --n-samples=$nsamples \
      --n-boundary=1 \
      --n-inner=$ninner \
      --decision-boundary-closeness=0.999 \
      --attack=$attack
  done
elif [[ "$mode" == "gap" ]]; then
  for closeness in 0.999 0.95 0.90 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo "1 boundary point, 999 inner, closeness of $closeness"
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python case_studies/thermometer/original/binarization_test.py \
      --cifar-path=data/cifar-10-batches-py/test_batch \
      --n-samples=$nsamples \
      --n-boundary=1 \
      --n-inner=999 \
      --attack=$attack \
      --decision-boundary-closeness=$closeness
  done
fi