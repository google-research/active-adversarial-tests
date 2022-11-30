nsamples=${1:-512}
epsilon=${2:-8}
mode=${3:-train}

kwargs=""
kwargs="--sample-from-corners"

echo "Attacking $nsamples with epsilon = $epsilon and model inference = $mode"
echo "kwargs: $kwargs"

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary point, 999 inner (their attack parameters)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python \
  case_studies/curriculum_at/binarization_test.py \
  --step_size=2 \
  --num_steps=20 \
  --loss_func=xent \
  --n_boundary_points=1 \
  --n_inner_points=999 \
  --epsilon=$epsilon \
  --inference_mode=$mode \
  --num_eval_examples=$nsamples \
  $kwargs

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary point, 999 inner (modified attack parameters)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python \
  case_studies/curriculum_at/binarization_test.py \
  --random_start \
  --step_size=0.5 \
  --num_steps=50 \
  --loss_func=logit-diff \
  --n_boundary_points=1 \
  --n_inner_points=999 \
  --epsilon=$epsilon \
  --inference_mode=$mode \
  --num_eval_examples=$nsamples\
  $kwargs