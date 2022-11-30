#! /bin/bash

python eval.py  --cuda_device=3 \
--num_classes=10 \
--ckpt_path=models/mc_eps4/checkpoint-200 \
--dataset=CIFAR \
--loss_type=xent \
--margin=50.0 \
--epsilon=8.0 \
--num_steps=100 \
--step_size=2.0 \
--random_start=True \
--targeted=False \
--batch_size=100 \
