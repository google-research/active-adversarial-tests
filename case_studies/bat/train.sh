#! /bin/bash

python train.py --cuda_device=0 \
--model_dir=/tmp \
--dataset=CIFAR \
--num_classes=10 \
--max_epoch=201 \
--decay_epoch1=100 \
--decay_epoch2=150 \
--start_epoch=0 \
--restore_ckpt_path='' \
--loss_type=xent \
--margin=50.0 \
--epsilon=8.0 \
--num_steps=1 \
--step_size=8.0 \
--random_start=True \
--targeted=True \
--target_type=MC \
--save_epochs=20 \
--eval_epochs=10 \
--multi=9.0 \
