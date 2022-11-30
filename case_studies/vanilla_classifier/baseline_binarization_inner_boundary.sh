function run_experiment {
  nboundary="$1"
  ninner="$2"
  epsilon="$3"
  stepsize="$4"

  PYTHONPATH=$PYTHONPATH:$(pwd) python case_studies/evaluate_classifier.py \
    --n-samples=2048 \
    --batch-size=512 \
    --input=checkpoints/mrn18_200_epochs.pth  \
    --decision-boundary-binarization="norm=linf epsilon=$epsilon n_inner_points=$ninner \
      n_boundary_points=$nboundary optimizer=sklearn adversarial_attack_settings=\"norm=linf epsilon=$epsilon \
      step_size=$stepsize n_steps=200\"" \
    --no-ce-loss
}

#run_experiment 1 49 0.031372549 0.0011372549
#run_experiment 1 149 0.031372549 0.0011372549
#run_experiment 1 249 0.031372549 0.0011372549
#run_experiment 1 499 0.031372549 0.0011372549
#run_experiment 1 1499 0.031372549 0.0011372549
#run_experiment 1 1999 0.031372549 0.0011372549
#run_experiment 1 2499 0.031372549 0.0011372549
#run_experiment 1 2999 0.031372549 0.0011372549