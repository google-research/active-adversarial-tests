#!/bin/bash

TF_CPP_MIN_LOG_LEVEL=3 ./venv3.8tf/bin/python case_studies/bat/adversarial_evaluation.py \
  --num_classes=10 \
  --ckpt_path=checkpoints/bat/mosa_eps4/checkpoint-200 \
  --dataset=CIFAR \
  --loss_type=xent \
  --margin=50.0 \
  --epsilon=8.0 \
  --num_steps=100 \
  --step_size=2.0 \
  --random_start=True \
  --batch_size=256 \
  --n_samples=2048
