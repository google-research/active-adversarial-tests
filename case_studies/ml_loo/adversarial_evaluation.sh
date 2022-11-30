# strong bim/bim:
#fpr1="0.2630430452371708"
#fpr5="0.05742787427778243"
#fpr10="0.006814145480778491"

# weak bim/bim2:
fpr1="0.8909015048587877"
fpr5="0.38117096263305317"
fpr10="0.05692284412332819"

nsamples=${1:-512}
fpr=${2:-5}

case $fpr in
  1)
    echo "Evaluating at @FPR=1"
    threshold=${fpr1}
    ;;
  5)
    echo "Evaluating at @FPR=5"
    threshold=${fpr5}
    ;;
  10)
    echo "Evaluating at @FPR=10"
    threshold=${fpr10}
    ;;
esac

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Naive attack (FPR$fpr)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 ../../venv3.8tf/bin/python adversarial_evaluation.py \
  --dataset_name cifar10 \
  --model_name resnet \
  --detector-attack=bim2 \
  --attack=bim2 \
  --detector-threshold=${threshold} \
  --batch-size=1 \
  --n-samples=512

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Adaptive attack (FPR$fpr)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 \../../venv3.8tf/bin/python adversarial_evaluation.py \
  --dataset_name cifar10 \
  --model_name resnet \
  --detector-attack=bim2 \
  --attack=fma \
  --detector-threshold=${threshold} \
  --batch-size=1 \
  --n-samples=$nsamples
