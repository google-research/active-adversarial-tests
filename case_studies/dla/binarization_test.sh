n_samples=512

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Normal test (1 boundary, 999 inner points), normal PGD"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) \
  python case_studies/dla/binarization_test.py  \
  --adversarial-attack=pgd \
  --epsilon=0.01 \
  --step-size=0.001 \
  --n-steps=200 \
  --n-inner-points=999 \
  --n-boundary-points=1 \
  --n-samples=$n_samples \
  --batch-size=2048

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Inverted test (1 boundary, 999 inner points), normal PGD"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) \
  python case_studies/dla/binarization_test.py  \
  --adversarial-attack=pgd \
  --epsilon=0.01 \
  --step-size=0.001 \
  --n-steps=200 \
  --n-inner-points=999 \
  --n-boundary-points=1 \
  --n-samples=$n_samples \
  --batch-size=2048 \
  --inverted-test

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Normal test (1 boundary, 999 inner points), joined PGD"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) \
  python case_studies/dla/binarization_test.py  \
  --adversarial-attack=joined-pgd \
  --epsilon=0.01 \
  --step-size=0.001 \
  --n-steps=200 \
  --n-inner-points=999 \
  --n-boundary-points=1 \
  --n-samples=$n_samples \
  --batch-size=2048

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Inverted test (1 boundary, 999 inner points), joined PGD"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) \
  python case_studies/dla/binarization_test.py  \
  --adversarial-attack=joined-pgd \
  --epsilon=0.01 \
  --step-size=0.001 \
  --n-steps=200 \
  --n-inner-points=999 \
  --n-boundary-points=1 \
  --n-samples=$n_samples \
  --batch-size=2048 \
  --inverted-test

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Normal test (1 boundary, 999 inner points), selective PGD"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) \
  python case_studies/dla/binarization_test.py  \
  --adversarial-attack=selective-pgd \
  --epsilon=0.01 \
  --step-size=0.001 \
  --n-steps=200 \
  --n-inner-points=999 \
  --n-boundary-points=1 \
  --n-samples=$n_samples \
  --batch-size=2048

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Inverted test (1 boundary, 999 inner points), selective PGD"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) \
  python case_studies/dla/binarization_test.py  \
  --adversarial-attack=selective-pgd \
  --epsilon=0.01 \
  --step-size=0.001 \
  --n-steps=200 \
  --n-inner-points=999 \
  --n-boundary-points=1 \
  --n-samples=$n_samples \
  --batch-size=2048 \
  --inverted-test