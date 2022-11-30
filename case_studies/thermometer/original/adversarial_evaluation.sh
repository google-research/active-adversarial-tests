nsamples=2048

epsilon=${1:-8}

echo "Original attack"
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python case_studies/thermometer/original/robustness_evaluation.py \
  --cifar-path=data/cifar-10-batches-py/test_batch \
  --end=-1 \
  --batch-size=512 \
  --epsilon=$epsilon \
  --attack=original

echo "Adaptive attack"
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python case_studies/thermometer/original/robustness_evaluation.py \
  --cifar-path=data/cifar-10-batches-py/test_batch \
  --end=-1 \
  --batch-size=256 \
  --epsilon=$epsilon \
  --attack=adaptive