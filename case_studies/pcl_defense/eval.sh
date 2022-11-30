
echo "Baseline:"
#python3 case_studies/pcl_defense/robustness.py --baseline
#python3 case_studies/pcl_defense/robustness.py --baseline --binarization-test --epsilon=4

echo "Model w/ their defense"
#python3 case_studies/pcl_defense/robustness.py
#python3 case_studies/pcl_defense/robustness.py --binarization-test --epsilon=8 --n-boundary-points=1 --n-inner-points=999
#python3 case_studies/pcl_defense/robustness.py --binarization-test --epsilon=6
#python3 case_studies/pcl_defense/robustness.py --binarization-test --epsilon=4


PYTHONPATH=$(pwd) python3 case_studies/pcl_defense/robustness.py \
  --binarization-test --epsilon=8 --n-boundary-points=1 \
  --n-inner-points=999
PYTHONPATH=$(pwd) python3 case_studies/pcl_defense/robustness.py \
  --binarization-test --epsilon=8 --n-boundary-points=1 --n-inner-points=999 \
  --attack=autopgd

#PYTHONPATH=$(pwd) python3 case_studies/pcl_defense/robustness.py \
#  --binarization-test --epsilon=8 --n-boundary-points=1 \
#  --n-inner-points=999 --use-autopgd-boundary-adversarials
#PYTHONPATH=$(pwd) python3 case_studies/pcl_defense/robustness.py \
#  --binarization-test --epsilon=8 --n-boundary-points=1 --n-inner-points=999 \
#  --use-autopgd-boundary-adversarials --attack=autopgd