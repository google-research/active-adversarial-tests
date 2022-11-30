nsamples=${1:-512}
epsilon=${2:-8}

# kwargs=""
kwargs="--sample-from-corners"
echo "Epsilon: $epsilon"
echo "#samples: $nsamples"
echo "kwargs: $kwargs"

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Using epsilon = $epsilon and few steps (20)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

PYTHONPATH=$(pwd) python3 case_studies/feature_scatter/fs_eval.py \
    --model-path=checkpoints/feature_scattering_linf_200_epochs.pth \
    --init_model_pass=latest \
    --attack=True \
    --attack_method_list=pgd-autopgddlr \
    --dataset=cifar10 \
    --batch_size_test=1 \
    --binarization-test \
    --num_samples_test=$nsamples \
    --n-inner-points=9999 \
    --n-boundary-points=1 \
    --resume \
    --epsilon=$epsilon \
    $kwargs
  #9999

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Using epsilon = $epsilon and more steps (200)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

PYTHONPATH=$(pwd) python3 case_studies/feature_scatter/fs_eval.py \
    --model-path=checkpoints/feature_scattering_linf_200_epochs.pth \
    --init_model_pass=latest \
    --attack=True \
    --attack_method_list=pgd-autopgddlr \
    --dataset=cifar10 \
    --batch_size_test=1 \
    --binarization-test \
    --num_samples_test=$nsamples \
    --n-inner-points=9999 \
    --n-boundary-points=1 \
    --resume \
    --epsilon=$epsilon \
    --more-steps \
    $kwargs