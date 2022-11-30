#!/bin/sh
nsamples=${1:-512}

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python \
  ./case_studies/jarn/adversarial_evaluation.py \
  --model_dir=checkpoints/jarn/modelJARN_cifar10_b64_beta_1.000_gamma_1.000_disc_update_steps20_l5bc32_imgpert_advdelay140000_tanhencact_zeromeaninput_160000steps \
  --data_path=data/cifar-10-batches-py/ \
  --eval_batch_size=512 \
  --num_steps=1 \
  --num_eval_examples=$nsamples \
  --attack=pgd
