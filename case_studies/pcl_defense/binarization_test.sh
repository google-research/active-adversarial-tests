nsamples=${1:-512}
attack=${2:-pgd}
epsilon=${3:-8}

# kwargs=""
kwargs="--sample-from-corners"

echo "Attack: $attack"
echo "Epsilon: $epsilon"
echo "#samples: $nsamples"
echo "kwargs: $kwargs"

PYTHONPATH=$(pwd) python3 case_studies/pcl_defense/robustness.py \
  --binarization-test \
  --epsilon=$epsilon \
  --n-boundary-points=1 \
  --n-inner-points=999 \
  --attack=$attack \
  --num-samples-test=$nsamples \
  $kwargs
