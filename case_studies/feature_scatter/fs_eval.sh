epsilon=${1:-8}
attack=${2-pgd-autopgddlr+}

export PYTHONPATH=./:$PYTHONPATH
python3 case_studies/feature_scatter/fs_eval.py \
    --model-path=checkpoints/feature_scattering_linf_200_epochs.pth \
    --init_model_pass=latest \
    --attack=True \
    --attack_method_list=$attack \
    --dataset=cifar10 \
    --batch_size_test=256 \
    --resume \
    --epsilon=$epsilon

#natural-fgsm-pgd-cw