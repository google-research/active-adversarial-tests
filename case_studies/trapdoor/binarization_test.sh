n_samples=${1:-512}
epsilon=${2:-0.031}

# for eps=0.01
#thresholds="0.27039704 0.3084671 0.24682844 0.22648834 0.25642416 0.2449155 0.25744236 0.2869493 0.2991438 0.2467549"
#thresholds="0.32685438 0.3048646 0.21139874 -0.011124074 0.26301256 0.25698307 0.25040865 0.18050945 0.3116589 0.16479838"
#thresholds="0.2704116 0.30847666 0.2505051 0.18937282 0.25757647 0.24697195 0.25848407 0.28757182 0.2991565 0.24589166"

thresholds="0.3268544 0.30486462 0.21139881 0.23486444 0.26301256 0.25698304 0.24667358 0.17656253 0.31165892 0.16479836"

kwargs=""
#kwargs="--sample-from-corners"

if [[ "$epsilon" == "0.01" ]]; then
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "eps=0.01, Normal test (1 boundary, 999 inner points), Normal PGD"
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/trapdoor/binarization_test.py \
    --n-samples=$n_samples \
    --epsilon=0.01 \
    --pgd-steps=100 \
    --thresholds $thresholds \
    --pgd-step-size=0.00078431372 \
    $kwargs


  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "eps=0.01, Inverted test (1 boundary, 999 inner points), Normal PGD"
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/trapdoor/binarization_test.py \
    --n-samples=$n_samples \
    --epsilon=0.01 \
    --pgd-steps=100 \
    --thresholds $thresholds \
    --pgd-step-size=0.00078431372 \
    --inverted-test \
    $kwargs

  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "eps=0.01, Normal test (1 boundary, 999 inner points), Orthogonal PGD"
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/trapdoor/binarization_test.py \
    --n-samples=$n_samples \
    --epsilon=0.01 \
    --attack=orthogonal \
    --thresholds $thresholds \
    --pgd-steps=500 \
    --pgd-step-size=0.01 \
    $kwargs

  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "eps=0.01, Inverted test (1 boundary, 999 inner points), Orthogonal PGD"
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/trapdoor/binarization_test.py \
    --n-samples=$n_samples \
    --epsilon=0.01 \
    --attack=orthogonal \
    --thresholds $thresholds \
    --pgd-steps=1000 \
    --pgd-step-size=0.05 \
    --inverted-test \
    $kwargs
elif [[ "$epsilon" == "0.031" ]]; then
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "eps=0.031, Normal test (1 boundary, 999 inner points), Normal PGD"
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/trapdoor/binarization_test.py \
    --n-samples=$n_samples \
    --epsilon=0.031 \
    --pgd-steps=100 \
    --thresholds $thresholds \
    --pgd-step-size=0.00078431372\
    $kwargs

  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "eps=0.031, Inverted test (1 boundary, 999 inner points), Normal PGD"
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/trapdoor/binarization_test.py \
    --n-samples=$n_samples \
    --epsilon=0.031 \
    --pgd-steps=100 \
    --thresholds $thresholds \
    --pgd-step-size=0.00078431372 \
    --inverted-test \
    $kwargs

  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "eps=0.031, Normal test (1 boundary, 999 inner points), Orthogonal PGD"
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/trapdoor/binarization_test.py \
    --n-samples=$n_samples \
    --epsilon=0.031 \
    --attack=orthogonal \
    --thresholds $thresholds \
    --pgd-steps=750 \
    --pgd-step-size=0.015 \
    $kwargs

  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "eps=0.031, Inverted test (1 boundary, 999 inner points), Orthogonal PGD"
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/trapdoor/binarization_test.py \
    --n-samples=$n_samples \
    --epsilon=0.031 \
    --attack=orthogonal \
    --thresholds $thresholds \
    --pgd-steps=1000 \
    --pgd-step-size=0.05 \
    --inverted-test \
    $kwargs
fi