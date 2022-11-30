#!/bin/sh

nsamples=${1:-512}
echo "#samples: $nsamples"

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "eps=8/255, 1 boundary, 999 inner points (original PGD)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python \
  ./case_studies/jarn/binarization_test.py \
  --model_dir=checkpoints/jarn/modelJARN_cifar10_b64_beta_1.000_gamma_1.000_disc_update_steps20_l5bc32_imgpert_advdelay140000_tanhencact_zeromeaninput_160000steps \
  --data_path=data/cifar-10-batches-py/ \
  --eval_batch_size=512 \
  --num_steps=20 \
  --num_eval_examples=$nsamples \
  --attack=pgd \
  --step_size=2.0

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "eps=8/255, 1 boundary, 999 inner points (modified PGD)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python \
  ./case_studies/jarn/binarization_test.py \
  --model_dir=checkpoints/jarn/modelJARN_cifar10_b64_beta_1.000_gamma_1.000_disc_update_steps20_l5bc32_imgpert_advdelay140000_tanhencact_zeromeaninput_160000steps \
  --data_path=data/cifar-10-batches-py/ \
  --eval_batch_size=512 \
  --num_steps=500 \
  --num_eval_examples=$nsamples \
  --attack=pgd-ld \
  --step_size=1.0 \
  --random_start

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "eps=8/255, 1 boundary, 999 inner points (APGD)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python \
  ./case_studies/jarn/binarization_test.py \
  --model_dir=checkpoints/jarn/modelJARN_cifar10_b64_beta_1.000_gamma_1.000_disc_update_steps20_l5bc32_imgpert_advdelay140000_tanhencact_zeromeaninput_160000steps \
  --data_path=data/cifar-10-batches-py/ \
  --eval_batch_size=512 \
  --num_steps=500 \
  --num_eval_examples=$nsamples \
  --attack=apgd \
  --step_size=1.0 \
  --random_start

