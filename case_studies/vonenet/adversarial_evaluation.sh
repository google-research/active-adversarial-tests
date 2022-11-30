printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "50 step PGD, step size = 0.1"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/vonenet/adversarial_evaluation.py \
  --in-path=/home/rolandz/imagenet_dataset/ \
  --n-samples=512 \
  --batch-size=64 \
  --attack=pgd \
  --epsilon=1 \
  --step-size=0.1 \
  --n-steps=50 \
  --ensemble-size=1 \
  --differentiable-replacement

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "50 step PGD, differentiable, step size = 0.1"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/vonenet/adversarial_evaluation.py \
  --in-path=/home/rolandz/imagenet_dataset/ \
  --n-samples=512 \
  --batch-size=64 \
  --attack=pgd \
  --epsilon=1 \
  --step-size=0.1 \
  --n-steps=50 \
  --ensemble-size=1 \
  --differentiable-replacement \
  --deterministic-replacement

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "50 step stable PGD, step size = 0.1"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/vonenet/adversarial_evaluation.py \
  --in-path=/home/rolandz/imagenet_dataset/ \
  --n-samples=512 \
  --batch-size=64 \
  --attack=pgd \
  --epsilon=1 \
  --step-size=0.1 \
  --n-steps=50 \
  --ensemble-size=1 \
  --stable-gradients \
  --differentiable-replacement

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "50 step stable PGD, deterministic, step size = 0.1"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/vonenet/adversarial_evaluation.py \
  --in-path=/home/rolandz/imagenet_dataset/ \
  --n-samples=512 \
  --batch-size=64 \
  --attack=pgd \
  --epsilon=1 \
  --step-size=0.1 \
  --n-steps=50 \
  --ensemble-size=1 \
  --stable-gradients \
  --differentiable-replacement \
  --deterministic-replacement

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "500 step stable PGD, step size = 0.01"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/vonenet/adversarial_evaluation.py \
  --in-path=/home/rolandz/imagenet_dataset/ \
  --n-samples=512 \
  --batch-size=64 \
  --attack=pgd \
  --epsilon=1 \
  --step-size=0.01 \
  --n-steps=500 \
  --ensemble-size=1 \
  --stable-gradients \
  --differentiable-replacement

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "500 step stable PGD, deterministic, step size = 0.01"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/vonenet/adversarial_evaluation.py \
  --in-path=/home/rolandz/imagenet_dataset/ \
  --n-samples=512 \
  --batch-size=64 \
  --attack=pgd \
  --epsilon=1 \
  --step-size=0.01 \
  --n-steps=500 \
  --ensemble-size=1 \
  --stable-gradients \
  --differentiable-replacement \
  --deterministic-replacement

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1000 step stable PGD, step size = 0.005"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/vonenet/adversarial_evaluation.py \
  --in-path=/home/rolandz/imagenet_dataset/ \
  --n-samples=512 \
  --batch-size=64 \
  --attack=pgd \
  --epsilon=1 \
  --step-size=0.005 \
  --n-steps=1000 \
  --ensemble-size=1 \
  --stable-gradients \
  --differentiable-replacement

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1000 step stable PGD, deterministic, step size = 0.005"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/vonenet/adversarial_evaluation.py \
  --in-path=/home/rolandz/imagenet_dataset/ \
  --n-samples=512 \
  --batch-size=64 \
  --attack=pgd \
  --epsilon=1 \
  --step-size=0.005 \
  --n-steps=1000 \
  --ensemble-size=1 \
  --stable-gradients \
  --differentiable-replacement \
  --deterministic-replacement