nsamples=128

epsilon=${1:-0.05}

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python case_studies/inputtransformations/adversarial_evaluation.py \
  --imagenet-path=/imagenet_dataset/ \
  --batch-size=128 \
  --n-samples=$nsamples \
  --epsilon=$epsilon \
  --defense=jpeg \
  --pgd-steps=256