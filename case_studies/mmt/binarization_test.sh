nsamples=${1:-512}
epsilon=${2:-8}

kwargs=""
kwargs="--sample_from_corners=True"

echo "#samples: $nsamples"
echo "epsilon: $epsilon"
echo "kwargs: $kwargs"


printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary point, 999 inner (Original attack)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$(pwd) venv3.8tf/bin/python case_studies/mmt/binarization_test_iterative.py --mean_var=10 --batch_size=50 \
  --attack_method_for_advtrain='MadryEtAl' \
  --dataset=cifar10 --use_ball=True \
  --use_MMLDA=True --use_advtrain=False --epoch=200 \
  --use_BN=True --normalize_output_for_ball=False --use_random=False \
  --adv_ratio=1.0 --use_target=False \
  --checkpoint=checkpoints/mmt_mmc_rn110.h5 \
  --n_samples=$nsamples \
  --attack_method='MadryEtAl' \
  --num_iter=50 \
  --epsilon=$epsilon \
  $kwargs


printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary point, 999 inner (Adaptive attack)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$(pwd) venv3.8tf/bin/python case_studies/mmt/binarization_test_iterative.py --mean_var=10 --batch_size=50 \
  --attack_method_for_advtrain='MadryEtAl' \
  --dataset=cifar10 --use_ball=True \
  --use_MMLDA=True --use_advtrain=False --epoch=200 \
  --use_BN=True --normalize_output_for_ball=False --use_random=False \
  --adv_ratio=1.0 --use_target=False \
  --checkpoint=checkpoints/mmt_mmc_rn110.h5 \
  --n_samples=$nsamples \
  --attack_method='Adaptive' \
  --num_iter=50 \
  --epsilon=$epsilon \
  $kwargs