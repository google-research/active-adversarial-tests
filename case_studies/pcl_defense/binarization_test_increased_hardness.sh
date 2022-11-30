attack=${1:-pgd}
nsamples=${2:-2048}
mode=$3
epsilon=${4:-8}

if [ -z ${mode+x} ]; then
  echo "No hardness mode specified. Choose from: ninner, gap"
  exit -1
fi

echo "Attack: $attack, #Samples: $nsamples, epsilon: $epsilon"
echo ""

if [[ "$mode" == "ninner" ]]; then
  for ninner in 49 99 199 299 399 499 599 699 799 899 999 1999; do
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo "1 boundary point, $ninner inner"
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    PYTHONPATH=$(pwd) python3 case_studies/pcl_defense/robustness.py \
      --binarization-test \
      --epsilon=$epsilon \
      --attack=$attack \
      --num-samples-test=$nsamples \
      --n-boundary-points=1 \
      --n-inner-points=999
  done
elif [[ "$mode" == "gap" ]]; then
  for closeness in 0.9999 0.999 0.95 0.90 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo "1 boundary point, 1999 inner, closeness of $closeness"
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    PYTHONPATH=$(pwd) python3 case_studies/pcl_defense/robustness.py \
      --binarization-test \
      --epsilon=$epsilon \
      --attack=$attack \
      --num-samples-test=$nsamples \
      --n-boundary-points=1 \
      --n-inner-points=1999 \
      --decision-boundary-closeness=$closeness
      # -n-inner-points was 999 before 25.02.2022
  done
fi