printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary, 1999 inner points, 50 step PGD, epsilon = 0.5"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/vonenet/binarization_test.py \
  --in-path=/home/rolandz/imagenet_dataset/ \
  --n-samples=2048 \
  --batch-size=64 \
  --attack=pgd \
  --epsilon=0.5 \
  --n-inner-points=1999 \
  --step-size=0.1 \
  --n-steps=50 \
  --ensemble-size=1 \
  --differentiable-replacement \
  #--stable-gradients \


printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary, 1999 inner points, 50 step PGD (stable), epsilon = 0.5"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' '
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/vonenet/binarization_test.py \
  --in-path=/home/rolandz/imagenet_dataset/ \
  --n-samples=2048 \
  --batch-size=64 \
  --attack=pgd \
  --epsilon=0.5 \
  --n-inner-points=1999 \
  --step-size=0.1 \
  --n-steps=50 \
  --ensemble-size=1 \
  --differentiable-replacement \
  --stable-gradients

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary, 1999 inner points, 50 step PGD, epsilon = 1"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/vonenet/binarization_test.py \
  --in-path=/home/rolandz/imagenet_dataset/ \
  --n-samples=2048 \
  --batch-size=64 \
  --attack=pgd \
  --epsilon=1 \
  --n-inner-points=1999 \
  --step-size=0.1 \
  --n-steps=50 \
  --ensemble-size=1 \
  --differentiable-replacement \
  #--stable-gradients \


printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary, 1999 inner points, 50 step PGD (stable), epsilon = 1"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/vonenet/binarization_test.py \
  --in-path=/home/rolandz/imagenet_dataset/ \
  --n-samples=2048 \
  --batch-size=64 \
  --attack=pgd \
  --epsilon=1 \
  --n-inner-points=1999 \
  --step-size=0.1 \
  --n-steps=50 \
  --ensemble-size=1 \
  --differentiable-replacement \
  --stable-gradients

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary, 1999 inner points, 50 step PGD, epsilon = 2"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/vonenet/binarization_test.py \
  --in-path=/home/rolandz/imagenet_dataset/ \
  --n-samples=2048 \
  --batch-size=64 \
  --attack=pgd \
  --epsilon=2 \
  --n-inner-points=1999 \
  --step-size=0.1 \
  --n-steps=50 \
  --ensemble-size=1 \
  --differentiable-replacement \
  #--stable-gradients \


printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary, 1999 inner points, 50 step PGD (stable), epsilon = 2"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/vonenet/binarization_test.py \
  --in-path=/home/rolandz/imagenet_dataset/ \
  --n-samples=2048 \
  --batch-size=64 \
  --attack=pgd \
  --epsilon=2 \
  --n-inner-points=1999 \
  --step-size=0.1 \
  --n-steps=50 \
  --ensemble-size=1 \
  --differentiable-replacement \
  --stable-gradients

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary, 1999 inner points, 50 step PGD, epsilon = 4"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/vonenet/binarization_test.py \
  --in-path=/home/rolandz/imagenet_dataset/ \
  --n-samples=2048 \
  --batch-size=64 \
  --attack=pgd \
  --epsilon=4 \
  --n-inner-points=1999 \
  --step-size=0.1 \
  --n-steps=50 \
  --ensemble-size=1 \
  --differentiable-replacement \
  #--stable-gradients \


printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary, 1999 inner points, 50 step PGD (stable), epsilon = 4"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/vonenet/binarization_test.py \
  --in-path=/home/rolandz/imagenet_dataset/ \
  --n-samples=2048 \
  --batch-size=64 \
  --attack=pgd \
  --epsilon=4 \
  --n-inner-points=1999 \
  --step-size=0.1 \
  --n-steps=50 \
  --ensemble-size=1 \
  --differentiable-replacement \
  --stable-gradients