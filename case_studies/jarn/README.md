# JARN_ICLR2020
This is our Tensorflow implementation of Jacobian Adversarially Regularized Networks (JARN). 

**Jacobian Adversarially Regularized Networks for Robustness (ICLR 2020)**<br>
*Alvin Chan, Yi Tay, Yew Soon Ong, Jie Fu*<br>
https://arxiv.org/abs/1912.10185

TL;DR: We show that training classifiers to produce salient input Jacobian matrices with a GAN-like regularization can boost adversarial robustness.


## Dependencies
1. Tensorflow 1.14.0
2. Python 3.7


## Usage
1. Install dependencies with `pip install -r requirements.txt`.
2. Run JARN training and evaluation with `sh run_train_jarn.sh`. Final evaluation output is saved in `attack_log`.


## Code overview
- `train_jarn.py`: trains the JARN model and subsequently evaluate on adversarial examples.
- `pgd_attack.py`: generates adversarial examples and save them in `attacks/`.
- `run_attack.py`: evaluates model on adversarial examples from `attacks/`.
- `config.py`: training parameters for JARN.
- `config_attack.py`: parameters for adversarial example evaluation.
- `model_jarn.py`: contains code for JARN model architectures.
- `cifar10_input.py` provides utility functions and classes for loading the CIFAR10 dataset.


## Citation
If you find our repository useful, please consider citing our paper:

```
@article{chan2019jacobian,
  title={Jacobian Adversarially Regularized Networks for Robustness},
  author={Chan, Alvin and Tay, Yi and Ong, Yew Soon and Fu, Jie},
  journal={arXiv preprint arXiv:1912.10185},
  year={2019}
}
```


## Acknowledgements

Useful code bases we used in our work:
- https://github.com/MadryLab/cifar10_challenge (for adversarial example generation and evaluation)