#!/usr/bin/env bash

SEED1=$1
SEED2=$2

for t in 100; do
  for adv_eps in 0.031373; do
    for seed in $SEED1; do
      for data_seed in $SEED2; do

        PYTHONPATH=$(pwd) python case_studies/diffpure/eval_sde_adv.py --exp ./case_studies/diffpure/exp_results \
          --config ./case_studies/diffpure/configs/cifar10.yml \
          -i cifar10-robust_adv-$t-eps$adv_eps-64x1-bm0-t0-end1e-5-cont-eot20 \
          --t $t \
          --adv_eps $adv_eps \
          --adv_batch_size 64 \
          --num_sub 5 \
          --domain cifar10 \
          --classifier_name cifar10-wideresnet-28-10 \
          --seed $seed \
          --data_seed $data_seed \
          --diffusion_type sde \
          --score_type score_sde \
          --attack_version rand \
          --eot_iter 20 \
          --batch-size 512 \
          --n_boundary_points=1 \
          --n_inner_points=999
      done
    done
  done
done
