epsilon=${1:-8}
attack=${2-pgd-autopgddlr+}
nsamples=${3:-2048}
#attack=${2-pgd-autopgddlr-autopgddlrt-autopgddlrplus}
export PYTHONPATH=./:$PYTHONPATH

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Using epsilon = $epsilon and few steps (20)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

python3 case_studies/feature_scatter/fs_eval.py \
    --model-path=checkpoints/feature_scattering_linf_200_epochs.pth \
    --init_model_pass=latest \
    --attack=True \
    --attack_method_list=$attack \
    --dataset=cifar10 \
    --batch_size_test=1 \
    --binarization-test \
    --num_samples_test=$nsamples \
    --n-inner-points=999 \
    --n-boundary-points=1 \
    --resume \
    --epsilon=$epsilon

exit

# autopgddlr-autopgdce-autopgddlr+-autopgdce+

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Using epsilon = $epsilon and more steps (200)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -


python3 case_studies/feature_scatter/fs_eval.py \
    --model-path=checkpoints/feature_scattering_linf_200_epochs.pth \
    --init_model_pass=latest \
    --attack=True \
    --attack_method_list=$attack \
    --dataset=cifar10 \
    --batch_size_test=1 \
    --binarization-test \
    --num_samples_test=$nsamples \
    --n-inner-points=999 \
    --n-boundary-points=1 \
    --resume \
    --epsilon=$epsilon \
    --more-steps
