nsamples=10000

echo "Running attacks with epsilon=128/255"

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Using their attack parameters and train mode (default)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python \
  case_studies/curriculum_at/adversarial_evaluation.py \
  --step_size=2 \
  --num_steps=20 \
  --loss_func=xent \
  --inference_mode=train \
  --epsilon=128 \
  --num_eval_examples=$nsamples

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Using adapted attack parameters and train mode (default)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python \
  case_studies/curriculum_at/adversarial_evaluation.py \
  --random_start \
  --step_size=0.5 \
  --num_steps=75 \
  --loss_func=logit-diff \
  --inference_mode=train \
  --epsilon=128 \
  --num_eval_examples=$nsamples

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Using their attack parameters and eval mode (modified)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python \
  case_studies/curriculum_at/adversarial_evaluation.py \
  --step_size=2 \
  --num_steps=20 \
  --loss_func=xent \
  --inference_mode=eval \
  --epsilon=128 \
  --num_eval_examples=$nsamples

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Using adapted attack parameters and eval mode (modified)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) ./venv3.8tf/bin/python \
  case_studies/curriculum_at/adversarial_evaluation.py \
  --random_start \
  --step_size=0.5 \
  --num_steps=75 \
  --loss_func=logit-diff \
  --inference_mode=eval \
  --epsilon=128 \
  --num_eval_examples=$nsamples