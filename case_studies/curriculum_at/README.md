# Get Fooled for the Right Reason
Official repository for the NeurIPS 2021 paper Get Fooled for the Right Reason: Improving Adversarial Robustness through a Teacher-guided Curriculum Learning Approach

## Dependencies
1. Tensorflow 1.14.0
2. Python 3.7

## Datasets
CIFAR10: https://www.cs.toronto.edu/~kriz/cifar.html

## Models
`modelGTP_cifar10`: https://www.dropbox.com/sh/29n2lt08ypjdw67/AABSZlD8nTM08E-bcZv1mdkOa?dl=0

## Usage
1. Install dependencies with `pip install -r requirements.txt`. Prefarably, create an anaconda environment.
2. Download and save datasets in `datasets/` folder.
3. Download and save model in `models/` folder.
4. Run the `python eval_attack.py`
5. The evaluation results will be stored in `attack_log` directory.

### Note
Using a GPU is highly recommended.


## Code overview
- `model_new.py`: contains code for IGAM model architectures.
- `cifar10_input.py` provides utility functions and classes for loading the CIFAR10 dataset.
- `PGD_attack.py`: generates adversarial examples and save them in `attacks/`.
- `run_attack.py`: evaluates model on adversarial examples from `attacks/`.
- `config_attack.py`: parameters for adversarial example evaluation.
- `eval_attack.py`: runs **FGSM, PGD-5, PGD-10, PGD-20** attacks and logs the results in `attack_log` directory. However, you can get results for any attack by modifying the `num_steps` flag in the code.

## Acknowledgements

Useful code bases we used in our work:
- https://github.com/MadryLab/cifar10_challenge (for adversarial example generation and evaluation)
- https://github.com/ashafahi/free_adv_train (for model code)
