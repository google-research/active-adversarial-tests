nsamples=${1:-512}

#kwargs=""
kwargs="--sample-from-corners"

echo "#samples: $nsamples"
echo "kwargs: $kwargs"

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary point, 999 inner (Original attack)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd):case_studies/empir/ ./venv3.8tf/bin/python case_studies/empir/examples/cifar10_binarization_test.py \
  --ensembleThree \
  --attack=3 \
  --abits=2 \
  --wbits=4 \
  --abits2=2 \
  --wbits2=2 \
  --model_path1=case_studies/empir/weights/Model1/ \
  --model_path2=case_studies/empir/weights/Model2/ \
  --model_path3=case_studies/empir/weights/Model3/ \
  --nb_samples=$nsamples \
  $kwargs

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "1 boundary point, 999 inner (Adaptive attack)"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$PYTHONPATH:$(pwd):case_studies/empir/ ./venv3.8tf/bin/python case_studies/empir/examples/cifar10_binarization_test.py \
  --ensembleThree  \
  --attack=3 \
  --abits=2 \
  --wbits=4 \
  --abits2=2 \
  --wbits2=2 \
  --model_path1=case_studies/empir/weights/Model1/ \
  --model_path2=case_studies/empir/weights/Model2/ \
  --model_path3=case_studies/empir/weights/Model3/ \
  --nb_samples=$nsamples \
  --robust-attack \
  $kwargs
