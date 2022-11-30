export PYTHONPATH=./:$PYTHONPATH
python case_studies/interpolation_training/eval.py \
    --model-path=checkpoints/interpolation_training_linf_200_epochs.pth \
    --init_model_pass=latest \
    --attack=True \
    --attack_method_list=natural-fgsm-pgd-cw \
    --batch_size_test=80 \
    --resume