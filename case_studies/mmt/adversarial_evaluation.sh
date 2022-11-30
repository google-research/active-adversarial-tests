TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$(pwd) venv3.8tf/bin/python case_studies/mmt/advtest_iterative.py --mean_var=10 --batch_size=50 \
  --attack_method_for_advtrain='MadryEtAl' \
  --dataset=cifar10 --target=False --use_ball=True \
  --use_MMLDA=True --use_advtrain=False --epoch=200 \
  --use_BN=True --normalize_output_for_ball=False --use_random=False \
  --adv_ratio=1.0 --use_target=False \
  --checkpoint=checkpoints/mmt_mmc_rn110.h5 \
  --n_samples=512 \
  --attack_method='MadryEtAl' \
  --num_iter=50

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$(pwd) venv3.8tf/bin/python case_studies/mmt/advtest_iterative.py --mean_var=10 --batch_size=50 \
  --attack_method_for_advtrain='MadryEtAl' \
  --dataset=cifar10 --target=False --use_ball=True \
  --use_MMLDA=True --use_advtrain=False --epoch=200 \
  --use_BN=True --normalize_output_for_ball=False --use_random=False \
  --adv_ratio=1.0 --use_target=False \
  --checkpoint=checkpoints/mmt_mmc_rn110.h5 \
  --n_samples=512 \
  --attack_method='Adaptive' \
  --num_iter=50