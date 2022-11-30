# strong bim/bim:
#fpr1="0.2630430452371708"
#fpr5="0.05742787427778243"
#fpr10="0.006814145480778491"

# weak bim/bim2:
fpr1="0.8909015048587877"
fpr5="0.38117096263305317"
fpr10="0.05692284412332819"

nsamples=${1:-512}

echo "Testing for FPR5"

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Normal Test (naive attack)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd)/../../ ../../venv3.8tf/bin/python binarization_test.py \
  --dataset_name cifar10 \
  --model_name resnet  \
  --n-samples=$nsamples \
  --detector-threshold=${fpr5} \
  --attack=bim \
  --detector-attack=bim

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Inverted Test (naive attack)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd)/../../ ../../venv3.8tf/bin/python binarization_test.py \
  --dataset_name cifar10 \
  --model_name resnet  \
  --n-samples=$nsamples \
  --detector-threshold=${fpr5} \
  --attack=bim \
  --detector-attack=bim \
  --inverted-test


printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Normal Test (adaptive attack)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd)/../../ ../../venv3.8tf/bin/python binarization_test.py \
  --dataset_name cifar10 \
  --model_name resnet  \
  --n-samples=$nsamples \
  --detector-threshold=${fpr5} \
  --attack=fma \
  --detector-attack=bim

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Inverted Test (adaptive attack)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd)/../../ ../../venv3.8tf/bin/python binarization_test.py \
  --dataset_name cifar10 \
  --model_name resnet  \
  --n-samples=$nsamples \
  --detector-threshold=${fpr5} \
  --attack=fma \
  --detector-attack=bim \
  --inverted-test