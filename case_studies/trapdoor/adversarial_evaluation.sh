n_samples=${1:-512}

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Normal/Naive PGD (FPR = 5%)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/trapdoor/adversarial_evaluation.py \
  --n-samples=$n_samples \
  --epsilon=0.01 \
  --pgd-steps=100 \
  --pgd-step-size=0.0007843137254901962 \
  --fpr-threshold=0.05 \
  --attack=naive

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Selective PGD (FPR = 5%)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/trapdoor/adversarial_evaluation.py \
  --n-samples=$n_samples \
  --epsilon=0.01 \
  --pgd-steps=500 \
  --pgd-step-size=0.025 \
  --fpr-threshold=0.05 \
  --attack=selective

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Orthogonal PGD (FPR = 5%)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/trapdoor/adversarial_evaluation.py \
  --n-samples=$n_samples \
  --epsilon=0.01 \
  --pgd-steps=1000 \
  --pgd-step-size=0.025 \
  --fpr-threshold=0.05 \
  --attack=orthogonal

echo ""

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Normal/Naive PGD (FPR = 50%)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/trapdoor/adversarial_evaluation.py \
  --n-samples=$n_samples \
  --epsilon=0.01 \
  --pgd-steps=100 \
  --pgd-step-size=0.0007843137254901962 \
  --fpr-threshold=0.50 \
  --attack=naive

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Selective PGD (FPR = 50%)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/trapdoor/adversarial_evaluation.py \
  --n-samples=$n_samples \
  --epsilon=0.01 \
  --pgd-steps=500 \
  --pgd-step-size=0.025 \
  --fpr-threshold=0.50 \
  --attack=selective

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Orthogonal PGD (FPR = 50%)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/trapdoor/adversarial_evaluation.py \
  --n-samples=$n_samples \
  --epsilon=0.01 \
  --pgd-steps=500 \
  --pgd-step-size=0.025 \
  --fpr-threshold=0.50 \
  --attack=orthogonal