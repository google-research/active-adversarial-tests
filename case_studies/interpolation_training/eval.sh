epsilon=${1:-8}
attack=${2-pgd-cw}

export PYTHONPATH=./:$PYTHONPATH
python  case_studies/interpolation_training/eval.py \
    --model-path=checkpoints/adversarial_inperpolation_linf_200_epochs.pth \
    --init_model_pass=latest \
    --attack=True \
    --attack_method_list=$attack \
    --batch_size_test=128 \
    --epsilon=$epsilon \
    --resume