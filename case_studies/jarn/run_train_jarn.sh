#!/bin/sh

python train_jarn.py --img_random_pert --eval_adv_attack --sep_opt_version 1 --beta 1 --disc_layers 5 --disc_base_channels 32 --disc_update_steps 20 --steps_before_adv_opt 140000 --step_size_schedule 0,0.1 80000,0.01 120000,0.001 --train_steps 160000 -b 64