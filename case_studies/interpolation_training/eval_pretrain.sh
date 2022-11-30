export PYTHONPATH=./:$PYTHONPATH
checkpoint_dir=./pre_trained_adv_interp_models/
CUDA_VISIBLE_DEVICES=1 python eval.py \
    --model_dir=$checkpoint_dir \
    --init_model_pass=latest \
    --attack=True \
    --attack_method_list=natural-fgsm-pgd-cw \
    --batch_size_test=80 \
    --resume