n_samples=512

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "epsilon = 0.01, FGSM"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) \
  python case_studies/dla/adversarial_evaluation.py  \
  --adversarial-attack=pgd \
  --epsilon=0.01 \
  --step-size=0.01 \
  --n-steps=1 \
  --n-samples=$n_samples \
  --batch-size=256

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "epsilon = 0.01, PGD"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) \
  python case_studies/dla/adversarial_evaluation.py  \
  --adversarial-attack=pgd \
  --epsilon=0.01 \
  --step-size=0.001 \
  --n-steps=200 \
  --n-samples=$n_samples \
  --batch-size=256