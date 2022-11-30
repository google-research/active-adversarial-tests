nsamples=${1:-512}

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Original attack"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) venv3.8tf/bin/python case_studies/odds/original/adversarial_evaluation.py \
  --n-samples=$nsamples --attack=original

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Adaptive attack"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) venv3.8tf/bin/python case_studies/odds/original/adversarial_evaluation.py \
  --n-samples=$nsamples --attack=adaptive

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Adaptive attack w/ EOT attack"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd) venv3.8tf/bin/python case_studies/odds/original/adversarial_evaluation.py \
  --n-samples=$nsamples --attack=adaptive-eot --batch-size=128